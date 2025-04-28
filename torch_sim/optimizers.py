"""Optimizers for geometry relaxations.

This module provides optimization algorithms for atomic structures in a batched format,
enabling efficient relaxation of multiple atomic structures simultaneously. It includes
several gradient-based methods with support for both atomic position and unit cell
optimization.

The module offers:

* Standard gradient descent for atomic positions
* Gradient descent with unit cell optimization
* FIRE (Fast Inertial Relaxation Engine) optimization with unit cell parameters
* FIRE optimization with Frechet cell parameterization for improved cell relaxation

"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import warnings
import numpy as np # Need numpy for vdot equivalent if not using torch.sum(f*v)

import torch_sim.math as tsm
from torch_sim.state import DeformGradMixin, SimState
from torch_sim.typing import StateDict

from torch_sim.state import GroupedSimState # Make sure GroupedSimState is available

import logging

logger = logging.getLogger(__name__)


@dataclass
class GDState(SimState):
    """State class for batched gradient descent optimization.

    This class extends SimState to store and track the evolution of system state
    during gradient descent optimization. It maintains the energies and forces
    needed to perform gradient-based structure relaxation in a batched manner.

    Attributes:
        positions (torch.Tensor): Atomic positions with shape [n_atoms, 3]
        masses (torch.Tensor): Atomic masses with shape [n_atoms]
        cell (torch.Tensor): Unit cell vectors with shape [n_batches, 3, 3]
        pbc (bool): Whether to use periodic boundary conditions
        atomic_numbers (torch.Tensor): Atomic numbers with shape [n_atoms]
        batch (torch.Tensor): Batch indices with shape [n_atoms]
        forces (torch.Tensor): Forces acting on atoms with shape [n_atoms, 3]
        energy (torch.Tensor): Potential energy with shape [n_batches]
    """

    forces: torch.Tensor
    energy: torch.Tensor


def gradient_descent(
    model: torch.nn.Module,
    *,
    lr: torch.Tensor | float = 0.01,
) -> tuple[
    Callable[[StateDict | SimState], GDState],
    Callable[[GDState], GDState],
]:
    """Initialize a batched gradient descent optimization.

    Creates an optimizer that performs standard gradient descent on atomic positions
    for multiple systems in parallel. The optimizer updates atomic positions based on
    forces computed by the provided model. The cell is not optimized with this optimizer.

    Args:
        model (torch.nn.Module): Model that computes energies and forces
        lr (torch.Tensor | float): Learning rate(s) for optimization. Can be a single
            float applied to all batches or a tensor with shape [n_batches] for
            batch-specific rates

    Returns:
        tuple: A pair of functions:
            - Initialization function that creates the initial BatchedGDState
            - Update function that performs one gradient descent step

    Notes:
        The learning rate controls the step size during optimization. Larger values can
        speed up convergence but may cause instability in the optimization process.
    """
    device, dtype = model.device, model.dtype

    def gd_init(
        state: SimState | StateDict,
        **kwargs: Any,
    ) -> GDState:
        """Initialize the batched gradient descent optimization state.

        Args:
            state: SimState containing positions, masses, cell, etc.
            kwargs: Additional keyword arguments to override state attributes

        Returns:
            Initialized BatchedGDState with forces and energy
        """
        if not isinstance(state, SimState):
            state = SimState(**state)

        atomic_numbers = kwargs.get("atomic_numbers", state.atomic_numbers)

        # Get initial forces and energy from model
        model_output = model(state)
        energy = model_output["energy"]
        forces = model_output["forces"]

        return GDState(
            positions=state.positions,
            forces=forces,
            energy=energy,
            masses=state.masses,
            cell=state.cell,
            pbc=state.pbc,
            atomic_numbers=atomic_numbers,
            batch=state.batch,
        )

    def gd_step(state: GDState, lr: torch.Tensor = lr) -> GDState:
        """Perform one gradient descent optimization step to update the
        atomic positions. The cell is not optimized.

        Args:
            state: Current optimization state
            lr: Learning rate(s) to use for this step, overriding the default

        Returns:
            Updated GDState after one optimization step
        """
        # Get per-atom learning rates by mapping batch learning rates to atoms
        if isinstance(lr, float):
            lr = torch.full((state.n_batches,), lr, device=device, dtype=dtype)

        atom_lr = lr[state.batch].unsqueeze(-1)  # shape: (total_atoms, 1)

        # Update positions using forces and per-atom learning rates
        state.positions = state.positions + atom_lr * state.forces

        # Get updated forces and energy from model
        model_output = model(state)

        # Update state with new forces and energy
        state.forces = model_output["forces"]
        state.energy = model_output["energy"]

        return state

    return gd_init, gd_step


@dataclass
class UnitCellGDState(GDState, DeformGradMixin):
    """State class for batched gradient descent optimization with unit cell.

    Extends GDState to include unit cell optimization parameters and stress
    information. This class maintains the state variables needed for simultaneously
    optimizing atomic positions and unit cell parameters.

    Attributes:
        # Inherited from GDState
        positions (torch.Tensor): Atomic positions with shape [n_atoms, 3]
        masses (torch.Tensor): Atomic masses with shape [n_atoms]
        cell (torch.Tensor): Unit cell vectors with shape [n_batches, 3, 3]
        pbc (bool): Whether to use periodic boundary conditions
        atomic_numbers (torch.Tensor): Atomic numbers with shape [n_atoms]
        batch (torch.Tensor): Batch indices with shape [n_atoms]
        forces (torch.Tensor): Forces acting on atoms with shape [n_atoms, 3]
        energy (torch.Tensor): Potential energy with shape [n_batches]

        # Additional attributes for cell optimization
        stress (torch.Tensor): Stress tensor with shape [n_batches, 3, 3]
        reference_cell (torch.Tensor): Reference unit cells with shape
            [n_batches, 3, 3]
        cell_factor (torch.Tensor): Scaling factor for cell optimization with shape
            [n_batches, 1, 1]
        hydrostatic_strain (bool): Whether to only allow hydrostatic deformation
        constant_volume (bool): Whether to maintain constant volume
        pressure (torch.Tensor): Applied pressure tensor with shape [n_batches, 3, 3]
        cell_positions (torch.Tensor): Cell positions with shape [n_batches, 3, 3]
        cell_forces (torch.Tensor): Cell forces with shape [n_batches, 3, 3]
        cell_masses (torch.Tensor): Cell masses with shape [n_batches, 3]
    """

    # Required attributes not in BatchedGDState
    reference_cell: torch.Tensor
    cell_factor: torch.Tensor
    hydrostatic_strain: bool
    constant_volume: bool
    pressure: torch.Tensor
    stress: torch.Tensor

    # Cell attributes
    cell_positions: torch.Tensor
    cell_forces: torch.Tensor
    cell_masses: torch.Tensor


def unit_cell_gradient_descent(  # noqa: PLR0915, C901
    model: torch.nn.Module,
    *,
    positions_lr: float = 0.01,
    cell_lr: float = 0.1,
    cell_factor: float | torch.Tensor | None = None,
    hydrostatic_strain: bool = False,
    constant_volume: bool = False,
    scalar_pressure: float = 0.0,
) -> tuple[
    Callable[[SimState | StateDict], UnitCellGDState],
    Callable[[UnitCellGDState], UnitCellGDState],
]:
    """Initialize a batched gradient descent optimization with unit cell parameters.

    Creates an optimizer that performs gradient descent on both atomic positions and
    unit cell parameters for multiple systems in parallel. Supports constraints on cell
    deformation and applied external pressure.

    This optimizer extends standard gradient descent to simultaneously optimize
    both atomic coordinates and unit cell parameters based on forces and stress
    computed by the provided model.

    Args:
        model (torch.nn.Module): Model that computes energies, forces, and stress
        positions_lr (float): Learning rate for atomic positions optimization. Default
            is 0.01.
        cell_lr (float): Learning rate for unit cell optimization. Default is 0.1.
        cell_factor (float | torch.Tensor | None): Scaling factor for cell
            optimization. If None, defaults to number of atoms per batch
        hydrostatic_strain (bool): Whether to only allow hydrostatic deformation
            (isotropic scaling). Default is False.
        constant_volume (bool): Whether to maintain constant volume during optimization
            Default is False.
        scalar_pressure (float): Applied external pressure in GPa. Default is 0.0.

    Returns:
        tuple: A pair of functions:
            - Initialization function that creates a BatchedUnitCellGDState
            - Update function that performs one gradient descent step with cell
                optimization

    Notes:
        - To fix the cell and only optimize atomic positions, set both
          constant_volume=True and hydrostatic_strain=True
        - The cell_factor parameter controls the relative scale of atomic vs cell
          optimization
        - Larger values for positions_lr and cell_lr can speed up convergence but
          may cause instability in the optimization process
    """
    device, dtype = model.device, model.dtype

    def gd_init(
        state: SimState,
        cell_factor: float | torch.Tensor | None = cell_factor,
        hydrostatic_strain: bool = hydrostatic_strain,  # noqa: FBT001
        constant_volume: bool = constant_volume,  # noqa: FBT001
        scalar_pressure: float = scalar_pressure,
    ) -> UnitCellGDState:
        """Initialize the batched gradient descent optimization state with unit cell.

        Args:
            state: Initial system state containing positions, masses, cell, etc.
            cell_factor: Scaling factor for cell optimization (default: number of atoms)
            hydrostatic_strain: Whether to only allow hydrostatic deformation
            constant_volume: Whether to maintain constant volume
            scalar_pressure: Applied pressure in GPa
            **kwargs: Additional keyword arguments for state initialization

        Returns:
            Initial UnitCellGDState with system configuration and forces
        """
        if not isinstance(state, SimState):
            state = SimState(**state)

        # Setup cell_factor
        if cell_factor is None:
            # Count atoms per batch
            _, counts = torch.unique(state.batch, return_counts=True)
            cell_factor = counts.to(dtype=dtype)

        if isinstance(cell_factor, int | float):
            # Use same factor for all batches
            cell_factor = torch.full(
                (state.n_batches,), cell_factor, device=device, dtype=dtype
            )

        # Reshape to (n_batches, 1, 1) for broadcasting
        cell_factor = cell_factor.view(-1, 1, 1)

        scalar_pressure = torch.full(
            (state.n_batches, 1, 1), scalar_pressure, device=device, dtype=dtype
        )
        # Setup pressure tensor
        pressure = scalar_pressure * torch.eye(3, device=device)

        # Get initial forces and energy from model
        model_output = model(state)
        energy = model_output["energy"]
        forces = model_output["forces"]
        stress = model_output["stress"]  # Already shape: (n_batches, 3, 3)

        # Create cell masses
        cell_masses = torch.ones(
            (state.n_batches, 3), device=device, dtype=dtype
        )  # One mass per cell DOF

        # Get current deformation gradient
        cur_deform_grad = DeformGradMixin._deform_grad(  # noqa: SLF001
            state.row_vector_cell, state.row_vector_cell
        )

        # Calculate cell positions
        cell_factor_expanded = cell_factor.expand(
            state.n_batches, 3, 1
        )  # shape: (n_batches, 3, 1)
        cell_positions = (
            cur_deform_grad.reshape(state.n_batches, 3, 3) * cell_factor_expanded
        )  # shape: (n_batches, 3, 3)

        # Calculate virial
        volumes = torch.linalg.det(state.cell).view(-1, 1, 1)
        virial = -volumes * (stress + pressure)

        if hydrostatic_strain:
            diag_mean = torch.diagonal(virial, dim1=1, dim2=2).mean(dim=1, keepdim=True)
            virial = diag_mean.unsqueeze(-1) * torch.eye(3, device=device).unsqueeze(
                0
            ).expand(state.n_batches, -1, -1)

        if constant_volume:
            diag_mean = torch.diagonal(virial, dim1=1, dim2=2).mean(dim=1, keepdim=True)
            virial = virial - diag_mean.unsqueeze(-1) * torch.eye(
                3, device=device
            ).unsqueeze(0).expand(state.n_batches, -1, -1)

        return UnitCellGDState(
            positions=state.positions,
            forces=forces,
            energy=energy,
            stress=stress,
            masses=state.masses,
            cell=state.cell,
            pbc=state.pbc,
            reference_cell=state.cell.clone(),
            cell_factor=cell_factor,
            hydrostatic_strain=hydrostatic_strain,
            constant_volume=constant_volume,
            pressure=pressure,
            atomic_numbers=state.atomic_numbers,
            batch=state.batch,
            cell_positions=cell_positions,
            cell_forces=virial / cell_factor,
            cell_masses=cell_masses,
        )

    def gd_step(
        state: UnitCellGDState,
        positions_lr: torch.Tensor = positions_lr,
        cell_lr: torch.Tensor = cell_lr,
    ) -> UnitCellGDState:
        """Perform one gradient descent optimization step with unit cell.

        Updates both atomic positions and cell parameters based on forces and stress.

        Args:
            state: Current optimization state
            positions_lr: Learning rate for atomic positions optimization
            cell_lr: Learning rate for unit cell optimization

        Returns:
            Updated UnitCellGDState after one optimization step
        """
        # Get dimensions
        n_batches = state.n_batches

        # Get per-atom learning rates by mapping batch learning rates to atoms
        if isinstance(positions_lr, float):
            positions_lr = torch.full(
                (state.n_batches,), positions_lr, device=device, dtype=dtype
            )

        if isinstance(cell_lr, float):
            cell_lr = torch.full((state.n_batches,), cell_lr, device=device, dtype=dtype)

        # Get current deformation gradient
        cur_deform_grad = state.deform_grad()

        # Calculate cell positions from deformation gradient
        cell_factor_expanded = state.cell_factor.expand(n_batches, 3, 1)
        cell_positions = (
            cur_deform_grad.reshape(n_batches, 3, 3) * cell_factor_expanded
        )  # shape: (n_batches, 3, 3)

        # Get per-atom and per-cell learning rates
        atom_wise_lr = positions_lr[state.batch].unsqueeze(-1)
        cell_wise_lr = cell_lr.view(-1, 1, 1)  # shape: (n_batches, 1, 1)

        # Update atomic and cell positions
        atomic_positions_new = state.positions + atom_wise_lr * state.forces
        cell_positions_new = cell_positions + cell_wise_lr * state.cell_forces

        # Update cell with deformation gradient
        cell_update = cell_positions_new / cell_factor_expanded
        new_row_vector_cell = torch.bmm(
            state.reference_row_vector_cell, cell_update.transpose(-2, -1)
        )

        # Update state
        state.positions = atomic_positions_new
        state.row_vector_cell = new_row_vector_cell

        # Get new forces and energy
        model_output = model(state)

        state.energy = model_output["energy"]
        state.forces = model_output["forces"]
        state.stress = model_output["stress"]

        # Calculate virial for cell forces
        volumes = torch.linalg.det(new_row_vector_cell).view(-1, 1, 1)
        virial = -volumes * (state.stress + state.pressure)
        if state.hydrostatic_strain:
            diag_mean = torch.diagonal(virial, dim1=1, dim2=2).mean(dim=1, keepdim=True)
            virial = diag_mean.unsqueeze(-1) * torch.eye(3, device=device).unsqueeze(
                0
            ).expand(n_batches, -1, -1)
        if state.constant_volume:
            diag_mean = torch.diagonal(virial, dim1=1, dim2=2).mean(dim=1, keepdim=True)
            virial = virial - diag_mean.unsqueeze(-1) * torch.eye(
                3, device=device
            ).unsqueeze(0).expand(n_batches, -1, -1)

        # Update cell forces
        state.cell_positions = cell_positions_new
        state.cell_forces = virial / state.cell_factor

        return state

    return gd_init, gd_step


@dataclass
class FireState(GroupedSimState):
    """State information for batched FIRE optimization.

    This class extends SimState to store and track the system state during FIRE
    (Fast Inertial Relaxation Engine) optimization. It maintains the atomic
    parameters along with their velocities and forces for structure relaxation using
    the FIRE algorithm.

    Attributes:
        # Inherited from SimState
        positions (torch.Tensor): Atomic positions with shape [n_atoms, 3]
        masses (torch.Tensor): Atomic masses with shape [n_atoms]
        cell (torch.Tensor): Unit cell vectors with shape [n_batches, 3, 3]
        pbc (bool): Whether to use periodic boundary conditions
        atomic_numbers (torch.Tensor): Atomic numbers with shape [n_atoms]
        batch (torch.Tensor): Batch indices with shape [n_atoms]

        # Atomic quantities
        forces (torch.Tensor): Forces on atoms with shape [n_atoms, 3]
        velocities (torch.Tensor): Atomic velocities with shape [n_atoms, 3]
        energy (torch.Tensor): Energy per batch with shape [n_batches]

        # FIRE optimization parameters
        dt (torch.Tensor): Current timestep per batch with shape [n_batches]
        alpha (torch.Tensor): Current mixing parameter per batch with shape [n_batches]
        n_pos (torch.Tensor): Number of positive power steps per batch with shape
            [n_batches]

        # Add batch_group inherited from GroupedSimState
        # batch_group: torch.Tensor # Inherited implicitly

    Properties:
        momenta (torch.Tensor): Atomwise momenta of the system with shape [n_atoms, 3],
            calculated as velocities * masses
    """

    # Required attributes not in SimState
    forces: torch.Tensor
    energy: torch.Tensor
    velocities: torch.Tensor

    # FIRE algorithm parameters
    dt: torch.Tensor
    alpha: torch.Tensor
    n_pos: torch.Tensor


def fire(
    model: torch.nn.Module,
    *,
    dt_max: float = 1.0,
    dt_start: float = 0.1,
    n_min: int = 5,
    f_inc: float = 1.1,
    f_dec: float = 0.5,
    alpha_start: float = 0.1,
    f_alpha: float = 0.99,
) -> tuple[
    FireState,
    Callable[[FireState], FireState],
]:
    """Initialize a batched FIRE optimization.

    Creates an optimizer that performs FIRE (Fast Inertial Relaxation Engine)
    optimization on atomic positions.

    Args:
        model (torch.nn.Module): Model that computes energies, forces, and stress
        dt_max (float): Maximum allowed timestep
        dt_start (float): Initial timestep
        n_min (int): Minimum steps before timestep increase
        f_inc (float): Factor for timestep increase when power is positive
        f_dec (float): Factor for timestep decrease when power is negative
        alpha_start (float): Initial velocity mixing parameter
        f_alpha (float): Factor for mixing parameter decrease

    Returns:
        tuple: A pair of functions:
            - Initialization function that creates a FireState
            - Update function that performs one FIRE optimization step

    Notes:
        - FIRE is generally more efficient than standard gradient descent for atomic
          structure optimization
        - The algorithm adaptively adjusts step sizes and mixing parameters based
          on the dot product of forces and velocities
    """
    device, dtype = model.device, model.dtype

    eps = 1e-8 if dtype == torch.float32 else 1e-16

    # Setup parameters
    params = [dt_max, dt_start, alpha_start, f_inc, f_dec, f_alpha, n_min]
    dt_max, dt_start, alpha_start, f_inc, f_dec, f_alpha, n_min = [
        torch.as_tensor(p, device=device, dtype=dtype) for p in params
    ]

    def fire_init(
        state: SimState | GroupedSimState | StateDict, # Allow GroupedSimState input
        dt_start: float = dt_start,
        alpha_start: float = alpha_start,
    ) -> FireState:
        """Initialize a batched FIRE optimization state.

        Args:
            state: Input state as SimState, GroupedSimState, or dict
            dt_start: Initial timestep per batch
            alpha_start: Initial mixing parameter per batch

        Returns:
            FireState with initialized optimization tensors
        """
        # Ensure state is a SimState or GroupedSimState object
        _state_dict = {}
        if isinstance(state, dict):
            _state_dict = state
            # Try creating GroupedSimState first if batch_group is present
            if 'batch_group' in state:
                try:
                    state = GroupedSimState(**state)
                except TypeError: # Fallback if other GroupedSimState args missing
                    state = SimState(**state)
            else:
                state = SimState(**state)

        # Determine batch_group for the output state
        if isinstance(state, GroupedSimState):
            batch_group = state.batch_group.clone()
        else: # Input was SimState or dict without batch_group
            # Create default group indices (all zeros)
            batch_group = torch.zeros(state.n_batches, device=state.device, dtype=torch.int64)

        # Get dimensions
        n_batches = state.n_batches

        # Get initial forces and energy from model
        # Use the original state object (SimState or GroupedSimState) for model call
        model_output = model(state)

        energy = model_output["energy"]  # [n_batches]
        forces = model_output["forces"]  # [n_total_atoms, 3]

        # Setup parameters
        dt_start_tensor = torch.full((n_batches,), dt_start.item(), device=device, dtype=dtype)
        alpha_start_tensor = torch.full((n_batches,), alpha_start.item(), device=device, dtype=dtype)

        n_pos = torch.zeros((n_batches,), device=device, dtype=torch.int32)

        # Create initial state
        return FireState(
            # Copy SimState attributes
            positions=state.positions.clone(),
            masses=state.masses.clone(),
            cell=state.cell.clone(),
            atomic_numbers=state.atomic_numbers.clone(),
            batch=state.batch.clone(),
            pbc=state.pbc,
            batch_group=batch_group, # Add batch_group
            # New attributes
            velocities=torch.zeros_like(state.positions),
            forces=forces,
            energy=energy,
            # Optimization attributes
            dt=dt_start_tensor,
            alpha=alpha_start_tensor,
            n_pos=n_pos,
        )

    def fire_step(
        state: FireState,
        alpha_start: float = alpha_start,
        dt_start: float = dt_start,
    ) -> FireState:
        """Perform one FIRE optimization step for batched atomic systems.

        Implements one step of the Fast Inertial Relaxation Engine (FIRE) algorithm for
        optimizing atomic positions in a batched setting. Uses velocity Verlet
        integration with adaptive velocity mixing.

        Args:
            state: Current optimization state containing atomic parameters
            alpha_start: Initial mixing parameter for velocity update
            dt_start: Initial timestep for velocity Verlet integration

        Returns:
            Updated state after performing one FIRE step
        """
        n_batches = state.n_batches

        # Setup parameters
        dt_start = torch.full((n_batches,), dt_start, device=device, dtype=dtype)
        alpha_start = torch.full((n_batches,), alpha_start, device=device, dtype=dtype)

        # Velocity Verlet first half step (v += 0.5*a*dt)
        atom_wise_dt = state.dt[state.batch].unsqueeze(-1)
        state.velocities += 0.5 * atom_wise_dt * state.forces / state.masses.unsqueeze(-1)

        # Split positions and forces into atomic and cell components
        atomic_positions = state.positions  # shape: (n_atoms, 3)

        # Update atomic positions
        atomic_positions_new = atomic_positions + atom_wise_dt * state.velocities

        # Update state with new positions and cell
        state.positions = atomic_positions_new

        # Get new forces, energy, and stress
        results = model(state)
        state.energy = results["energy"]
        state.forces = results["forces"]

        # Velocity Verlet first half step (v += 0.5*a*dt)
        state.velocities += 0.5 * atom_wise_dt * state.forces / state.masses.unsqueeze(-1)

        # Calculate power (F·V) for atoms
        atomic_power = (state.forces * state.velocities).sum(dim=1)  # [n_atoms]
        atomic_power_per_batch = torch.zeros(
            n_batches, device=device, dtype=atomic_power.dtype
        )
        atomic_power_per_batch.scatter_add_(
            dim=0, index=state.batch, src=atomic_power
        )  # [n_batches]

        # Calculate power for cell DOFs
        batch_power = atomic_power_per_batch

        for batch_idx in range(n_batches):
            # FIRE specific updates
            if batch_power[batch_idx] > 0:  # Power is positive
                state.n_pos[batch_idx] += 1
                if state.n_pos[batch_idx] > n_min:
                    state.dt[batch_idx] = min(state.dt[batch_idx] * f_inc, dt_max)
                    state.alpha[batch_idx] = state.alpha[batch_idx] * f_alpha
            else:  # Power is negative
                state.n_pos[batch_idx] = 0
                state.dt[batch_idx] = state.dt[batch_idx] * f_dec
                state.alpha[batch_idx] = alpha_start[batch_idx]
                # Reset velocities for both atoms and cell
                state.velocities[state.batch == batch_idx] = 0

        # Mix velocity and force direction using FIRE for atoms
        v_norm = torch.norm(state.velocities, dim=1, keepdim=True)
        f_norm = torch.norm(state.forces, dim=1, keepdim=True)
        # Avoid division by zero
        # mask = f_norm > 1e-10
        # state.velocity = torch.where(
        #     mask,
        #     (1.0 - state.alpha) * state.velocity
        #     + state.alpha * state.forces * v_norm / f_norm,
        #     state.velocity,
        # )
        atom_wise_alpha = state.alpha[state.batch].unsqueeze(-1)
        state.velocities = (
            1.0 - atom_wise_alpha
        ) * state.velocities + atom_wise_alpha * state.forces * v_norm / (f_norm + eps)

        return state

    return fire_init, fire_step


@dataclass
class UnitCellFireState(SimState, DeformGradMixin):
    """State information for batched FIRE optimization with unit cell degrees of
    freedom.

    This class extends SimState to store and track the system state during FIRE
    (Fast Inertial Relaxation Engine) optimization. It maintains both atomic and cell
    parameters along with their velocities and forces for structure relaxation using
    the FIRE algorithm.

    Attributes:
        # Inherited from SimState
        positions (torch.Tensor): Atomic positions with shape [n_atoms, 3]
        masses (torch.Tensor): Atomic masses with shape [n_atoms]
        cell (torch.Tensor): Unit cell vectors with shape [n_batches, 3, 3]
        pbc (bool): Whether to use periodic boundary conditions
        atomic_numbers (torch.Tensor): Atomic numbers with shape [n_atoms]
        batch (torch.Tensor): Batch indices with shape [n_atoms]

        # Atomic quantities
        forces (torch.Tensor): Forces on atoms with shape [n_atoms, 3]
        velocities (torch.Tensor): Atomic velocities with shape [n_atoms, 3]
        energy (torch.Tensor): Energy per batch with shape [n_batches]
        stress (torch.Tensor): Stress tensor with shape [n_batches, 3, 3]

        # Cell quantities
        cell_positions (torch.Tensor): Cell positions with shape [n_batches, 3, 3]
        cell_velocities (torch.Tensor): Cell velocities with shape [n_batches, 3, 3]
        cell_forces (torch.Tensor): Cell forces with shape [n_batches, 3, 3]
        cell_masses (torch.Tensor): Cell masses with shape [n_batches, 3]

        # Cell optimization parameters
        reference_cell (torch.Tensor): Original unit cells with shape [n_batches, 3, 3]
        cell_factor (torch.Tensor): Cell optimization scaling factor with shape
            [n_batches, 1, 1]
        pressure (torch.Tensor): Applied pressure tensor with shape [n_batches, 3, 3]
        hydrostatic_strain (bool): Whether to only allow hydrostatic deformation
        constant_volume (bool): Whether to maintain constant volume

        # FIRE optimization parameters
        dt (torch.Tensor): Current timestep per batch with shape [n_batches]
        alpha (torch.Tensor): Current mixing parameter per batch with shape [n_batches]
        n_pos (torch.Tensor): Number of positive power steps per batch with shape
            [n_batches]

    Properties:
        momenta (torch.Tensor): Atomwise momenta of the system with shape [n_atoms, 3],
            calculated as velocities * masses
    """

    # Required attributes not in SimState
    forces: torch.Tensor
    energy: torch.Tensor
    stress: torch.Tensor
    velocities: torch.Tensor

    # Cell attributes
    cell_positions: torch.Tensor
    cell_velocities: torch.Tensor
    cell_forces: torch.Tensor
    cell_masses: torch.Tensor

    # Optimization-specific attributes
    reference_cell: torch.Tensor
    cell_factor: torch.Tensor
    pressure: torch.Tensor
    hydrostatic_strain: bool
    constant_volume: bool

    # FIRE algorithm parameters
    dt: torch.Tensor
    alpha: torch.Tensor
    n_pos: torch.Tensor


def unit_cell_fire(  # noqa: C901, PLR0915
    model: torch.nn.Module,
    *,
    dt_max: float = 1.0,
    dt_start: float = 0.1,
    n_min: int = 5,
    f_inc: float = 1.1,
    f_dec: float = 0.5,
    alpha_start: float = 0.1,
    f_alpha: float = 0.99,
    cell_factor: float | None = None,
    hydrostatic_strain: bool = False,
    constant_volume: bool = False,
    scalar_pressure: float = 0.0,
) -> tuple[
    UnitCellFireState,
    Callable[[UnitCellFireState], UnitCellFireState],
]:
    """Initialize a batched FIRE optimization with unit cell degrees of freedom.

    Creates an optimizer that performs FIRE (Fast Inertial Relaxation Engine)
    optimization on both atomic positions and unit cell parameters for multiple systems
    in parallel. FIRE combines molecular dynamics with velocity damping and adjustment
    of time steps to efficiently find local minima.

    Args:
        model (torch.nn.Module): Model that computes energies, forces, and stress
        dt_max (float): Maximum allowed timestep
        dt_start (float): Initial timestep
        n_min (int): Minimum steps before timestep increase
        f_inc (float): Factor for timestep increase when power is positive
        f_dec (float): Factor for timestep decrease when power is negative
        alpha_start (float): Initial velocity mixing parameter
        f_alpha (float): Factor for mixing parameter decrease
        cell_factor (float | None): Scaling factor for cell optimization.
            If None, defaults to number of atoms per batch
        hydrostatic_strain (bool): Whether to only allow hydrostatic deformation
            (isotropic scaling)
        constant_volume (bool): Whether to maintain constant volume during optimization
        scalar_pressure (float): Applied external pressure in GPa

    Returns:
        tuple: A pair of functions:
            - Initialization function that creates a BatchedUnitCellFireState
            - Update function that performs one FIRE optimization step

    Notes:
        - FIRE is generally more efficient than standard gradient descent for atomic
          structure optimization
        - The algorithm adaptively adjusts step sizes and mixing parameters based
          on the dot product of forces and velocities
        - To fix the cell and only optimize atomic positions, set both
          constant_volume=True and hydrostatic_strain=True
        - The cell_factor parameter controls the relative scale of atomic vs cell
          optimization
    """
    device, dtype = model.device, model.dtype

    eps = 1e-8 if dtype == torch.float32 else 1e-16

    # Setup parameters
    params = [dt_max, dt_start, alpha_start, f_inc, f_dec, f_alpha, n_min]
    dt_max, dt_start, alpha_start, f_inc, f_dec, f_alpha, n_min = [
        torch.as_tensor(p, device=device, dtype=dtype) for p in params
    ]

    def fire_init(
        state: SimState | StateDict,
        cell_factor: torch.Tensor | None = cell_factor,
        scalar_pressure: float = scalar_pressure,
        dt_start: float = dt_start,
        alpha_start: float = alpha_start,
    ) -> UnitCellFireState:
        """Initialize a batched FIRE optimization state with unit cell.

        Args:
            state: Input state as SimState object or state parameter dict
            cell_factor: Cell optimization scaling factor. If None, uses atoms per batch.
                Single value or tensor of shape [n_batches].
            scalar_pressure: Applied pressure in energy units
            dt_start: Initial timestep per batch
            alpha_start: Initial mixing parameter per batch

        Returns:
            UnitCellFireState with initialized optimization tensors
        """
        if not isinstance(state, SimState):
            state = SimState(**state)

        # Get dimensions
        n_batches = state.n_batches

        # Setup cell_factor
        if cell_factor is None:
            # Count atoms per batch
            _, counts = torch.unique(state.batch, return_counts=True)
            cell_factor = counts.to(dtype=dtype)

        if isinstance(cell_factor, int | float):
            # Use same factor for all batches
            cell_factor = torch.full(
                (state.n_batches,), cell_factor, device=device, dtype=dtype
            )

        # Reshape to (n_batches, 1, 1) for broadcasting
        cell_factor = cell_factor.view(-1, 1, 1)

        # Setup pressure tensor
        pressure = scalar_pressure * torch.eye(3, device=device, dtype=dtype)
        pressure = pressure.unsqueeze(0).expand(n_batches, -1, -1)

        # Get initial forces and energy from model
        model_output = model(state)

        energy = model_output["energy"]  # [n_batches]
        forces = model_output["forces"]  # [n_total_atoms, 3]
        stress = model_output["stress"]  # [n_batches, 3, 3]

        volumes = torch.linalg.det(state.cell).view(-1, 1, 1)
        virial = -volumes * (stress + pressure)  # P is P_ext * I

        if hydrostatic_strain:
            diag_mean = torch.diagonal(virial, dim1=1, dim2=2).mean(dim=1, keepdim=True)
            virial = diag_mean.unsqueeze(-1) * torch.eye(3, device=device).unsqueeze(
                0
            ).expand(n_batches, -1, -1)

        if constant_volume:
            diag_mean = torch.diagonal(virial, dim1=1, dim2=2).mean(dim=1, keepdim=True)
            virial = virial - diag_mean.unsqueeze(-1) * torch.eye(
                3, device=device
            ).unsqueeze(0).expand(n_batches, -1, -1)

        cell_forces = virial / cell_factor

        # Sum masses per batch using segment_reduce
        # TODO (AG): check this
        batch_counts = torch.bincount(state.batch)

        cell_masses = torch.segment_reduce(
            state.masses, reduce="sum", lengths=batch_counts
        )  # shape: (n_batches,)
        cell_masses = cell_masses.unsqueeze(-1).expand(-1, 3)  # shape: (n_batches, 3)

        # Setup parameters
        dt_start = torch.full((n_batches,), dt_start, device=device, dtype=dtype)
        alpha_start = torch.full((n_batches,), alpha_start, device=device, dtype=dtype)

        n_pos = torch.zeros((n_batches,), device=device, dtype=torch.int32)

        # Create initial state
        return UnitCellFireState(
            # Copy SimState attributes
            positions=state.positions.clone(),
            masses=state.masses.clone(),
            cell=state.cell.clone(),
            atomic_numbers=state.atomic_numbers.clone(),
            batch=state.batch.clone(),
            pbc=state.pbc,
            # New attributes
            velocities=torch.zeros_like(state.positions),
            forces=forces,
            energy=energy,
            stress=stress,
            # Cell attributes
            cell_positions=torch.zeros(n_batches, 3, 3, device=device, dtype=dtype),
            cell_velocities=torch.zeros(n_batches, 3, 3, device=device, dtype=dtype),
            cell_forces=cell_forces,
            cell_masses=cell_masses,
            # Optimization attributes
            reference_cell=state.cell.clone(),
            cell_factor=cell_factor,
            pressure=pressure,
            dt=dt_start,
            alpha=alpha_start,
            n_pos=n_pos,
            hydrostatic_strain=hydrostatic_strain,
            constant_volume=constant_volume,
        )

    def fire_step(  # noqa: PLR0915
        state: UnitCellFireState,
        alpha_start: float = alpha_start,
        dt_start: float = dt_start,
    ) -> UnitCellFireState:
        """Perform one FIRE optimization step for batched atomic systems with unit cell
        optimization.

        Implements one step of the Fast Inertial Relaxation Engine (FIRE) algorithm for
        optimizing atomic positions and unit cell parameters in a batched setting. Uses
        velocity Verlet integration with adaptive velocity mixing.

        Args:
            state: Current optimization state containing atomic and cell parameters
            alpha_start: Initial mixing parameter for velocity update
            dt_start: Initial timestep for velocity Verlet integration

        Returns:
            Updated state after performing one FIRE step
        """
        n_batches = state.n_batches

        # Setup parameters
        dt_start = torch.full((n_batches,), dt_start, device=device, dtype=dtype)
        alpha_start = torch.full((n_batches,), alpha_start, device=device, dtype=dtype)

        # Calculate current deformation gradient
        cur_deform_grad = torch.transpose(
            torch.linalg.solve(state.reference_cell, state.cell), 1, 2
        )  # shape: (n_batches, 3, 3)

        # Calculate cell positions from deformation gradient
        cell_factor_expanded = state.cell_factor.expand(n_batches, 3, 1)
        cell_positions = cur_deform_grad * cell_factor_expanded

        # Velocity Verlet first half step (v += 0.5*a*dt)
        atom_wise_dt = state.dt[state.batch].unsqueeze(-1)
        cell_wise_dt = state.dt.unsqueeze(-1).unsqueeze(-1)

        state.velocities += 0.5 * atom_wise_dt * state.forces / state.masses.unsqueeze(-1)
        state.cell_velocities += (
            0.5 * cell_wise_dt * state.cell_forces / state.cell_masses.unsqueeze(-1)
        )

        # Split positions and forces into atomic and cell components
        atomic_positions = state.positions  # shape: (n_atoms, 3)

        # Update atomic and cell positions
        atomic_positions_new = atomic_positions + atom_wise_dt * state.velocities
        cell_positions_new = cell_positions + cell_wise_dt * state.cell_velocities

        # Update cell with deformation gradient
        cell_update = cell_positions_new / cell_factor_expanded
        new_cell = torch.bmm(state.reference_cell, cell_update.transpose(1, 2))

        # Update state with new positions and cell
        state.positions = atomic_positions_new
        state.cell_positions = cell_positions_new
        state.cell = new_cell

        # Get new forces, energy, and stress
        results = model(state)
        state.energy = results["energy"]
        forces = results["forces"]
        stress = results["stress"]

        state.forces = forces
        state.stress = stress
        # Calculate virial
        volumes = torch.linalg.det(new_cell).view(-1, 1, 1)
        virial = -volumes * (stress + state.pressure)
        if state.hydrostatic_strain:
            diag_mean = torch.diagonal(virial, dim1=1, dim2=2).mean(dim=1, keepdim=True)
            virial = diag_mean.unsqueeze(-1) * torch.eye(3, device=device).unsqueeze(
                0
            ).expand(n_batches, -1, -1)
        if state.constant_volume:
            diag_mean = torch.diagonal(virial, dim1=1, dim2=2).mean(dim=1, keepdim=True)
            virial = virial - diag_mean.unsqueeze(-1) * torch.eye(
                3, device=device
            ).unsqueeze(0).expand(n_batches, -1, -1)

        state.cell_forces = virial / state.cell_factor

        # Velocity Verlet first half step (v += 0.5*a*dt)
        state.velocities += 0.5 * atom_wise_dt * state.forces / state.masses.unsqueeze(-1)
        state.cell_velocities += (
            0.5 * cell_wise_dt * state.cell_forces / state.cell_masses.unsqueeze(-1)
        )

        # Calculate power (F·V) for atoms
        atomic_power = (state.forces * state.velocities).sum(dim=1)  # [n_atoms]
        atomic_power_per_batch = torch.zeros(
            n_batches, device=device, dtype=atomic_power.dtype
        )
        atomic_power_per_batch.scatter_add_(
            dim=0, index=state.batch, src=atomic_power
        )  # [n_batches]

        # Calculate power for cell DOFs
        cell_power = (state.cell_forces * state.cell_velocities).sum(
            dim=(1, 2)
        )  # [n_batches]
        batch_power = atomic_power_per_batch + cell_power

        for batch_idx in range(n_batches):
            # FIRE specific updates
            if batch_power[batch_idx] > 0:  # Power is positive
                state.n_pos[batch_idx] += 1
                if state.n_pos[batch_idx] > n_min:
                    state.dt[batch_idx] = min(state.dt[batch_idx] * f_inc, dt_max)
                    state.alpha[batch_idx] = state.alpha[batch_idx] * f_alpha
            else:  # Power is negative
                state.n_pos[batch_idx] = 0
                state.dt[batch_idx] = state.dt[batch_idx] * f_dec
                state.alpha[batch_idx] = alpha_start[batch_idx]
                # Reset velocities for both atoms and cell
                state.velocities[state.batch == batch_idx] = 0
                state.cell_velocities[batch_idx] = 0

        # Mix velocity and force direction using FIRE for atoms
        v_norm = torch.norm(state.velocities, dim=1, keepdim=True)
        f_norm = torch.norm(state.forces, dim=1, keepdim=True)
        # Avoid division by zero
        # mask = f_norm > 1e-10
        # state.velocity = torch.where(
        #     mask,
        #     (1.0 - state.alpha) * state.velocity
        #     + state.alpha * state.forces * v_norm / f_norm,
        #     state.velocity,
        # )
        batch_wise_alpha = state.alpha[state.batch].unsqueeze(-1)
        state.velocities = (
            1.0 - batch_wise_alpha
        ) * state.velocities + batch_wise_alpha * state.forces * v_norm / (f_norm + eps)

        # Mix velocity and force direction for cell DOFs
        cell_v_norm = torch.norm(state.cell_velocities, dim=(1, 2), keepdim=True)
        cell_f_norm = torch.norm(state.cell_forces, dim=(1, 2), keepdim=True)
        cell_wise_alpha = state.alpha.unsqueeze(-1).unsqueeze(-1)
        cell_mask = cell_f_norm > eps
        state.cell_velocities = torch.where(
            cell_mask,
            (1.0 - cell_wise_alpha) * state.cell_velocities
            + cell_wise_alpha * state.cell_forces * cell_v_norm / cell_f_norm,
            state.cell_velocities,
        )

        return state

    return fire_init, fire_step


@dataclass
class FrechetCellFIREState(SimState, DeformGradMixin):
    """State class for batched FIRE optimization with Frechet cell derivatives.

    This class extends SimState to store and track the system state during FIRE
    optimization with matrix logarithm parameterization for cell degrees of freedom.
    This parameterization provides improved handling of cell deformations during
    optimization.

    Attributes:
        # Inherited from SimState
        positions (torch.Tensor): Atomic positions with shape [n_atoms, 3]
        masses (torch.Tensor): Atomic masses with shape [n_atoms]
        cell (torch.Tensor): Unit cell vectors with shape [n_batches, 3, 3]
        pbc (bool): Whether to use periodic boundary conditions
        atomic_numbers (torch.Tensor): Atomic numbers with shape [n_atoms]
        batch (torch.Tensor): Batch indices with shape [n_atoms]

        # Additional atomic quantities
        forces (torch.Tensor): Forces on atoms with shape [n_atoms, 3]
        energy (torch.Tensor): Energy per batch with shape [n_batches]
        velocities (torch.Tensor): Atomic velocities with shape [n_atoms, 3]
        stress (torch.Tensor): Stress tensor with shape [n_batches, 3, 3]

        # Optimization-specific attributes
        reference_cell (torch.Tensor): Original unit cell with shape [n_batches, 3, 3]
        cell_factor (torch.Tensor): Scaling factor for cell optimization with shape
            [n_batches, 1, 1]
        pressure (torch.Tensor): Applied pressure tensor with shape [n_batches, 3, 3]
        hydrostatic_strain (bool): Whether to only allow hydrostatic deformation
        constant_volume (bool): Whether to maintain constant volume

        # Cell attributes using log parameterization
        cell_positions (torch.Tensor): Cell positions using log parameterization with
            shape [n_batches, 3, 3]
        cell_velocities (torch.Tensor): Cell velocities with shape [n_batches, 3, 3]
        cell_forces (torch.Tensor): Cell forces with shape [n_batches, 3, 3]
        cell_masses (torch.Tensor): Cell masses with shape [n_batches, 3]

        # FIRE algorithm parameters
        dt (torch.Tensor): Current timestep per batch with shape [n_batches]
        alpha (torch.Tensor): Current mixing parameter per batch with shape [n_batches]
        n_pos (torch.Tensor): Number of positive power steps per batch with shape
            [n_batches]

    Properties:
        momenta (torch.Tensor): Atomwise momenta of the system with shape [n_atoms, 3],
            calculated as velocities * masses
    """

    # Required attributes not in SimState
    forces: torch.Tensor
    energy: torch.Tensor
    velocities: torch.Tensor
    stress: torch.Tensor

    # Optimization-specific attributes
    reference_cell: torch.Tensor
    cell_factor: torch.Tensor
    pressure: torch.Tensor
    hydrostatic_strain: bool
    constant_volume: bool

    # Cell attributes
    cell_positions: torch.Tensor
    cell_velocities: torch.Tensor
    cell_forces: torch.Tensor
    cell_masses: torch.Tensor

    # FIRE algorithm parameters
    dt: torch.Tensor
    alpha: torch.Tensor
    n_pos: torch.Tensor


def frechet_cell_fire(  # noqa: C901, PLR0915
    model: torch.nn.Module,
    *,
    dt_max: float = 1.0,
    dt_start: float = 0.1,
    n_min: int = 5,
    f_inc: float = 1.1,
    f_dec: float = 0.5,
    alpha_start: float = 0.1,
    f_alpha: float = 0.99,
    cell_factor: float | None = None,
    hydrostatic_strain: bool = False,
    constant_volume: bool = False,
    scalar_pressure: float = 0.0,
) -> tuple[
    FrechetCellFIREState,
    Callable[[FrechetCellFIREState], FrechetCellFIREState],
]:
    """Initialize a batched FIRE optimization with Frechet cell parameterization.

    Creates an optimizer that performs FIRE optimization on both atomic positions and
    unit cell parameters using matrix logarithm parameterization for cell degrees of
    freedom. This parameterization provides forces consistent with numerical
    derivatives of the potential energy with respect to cell variables, resulting in
    more robust cell optimization.

    Args:
        model (torch.nn.Module): Model that computes energies, forces, and stress.
        dt_max (float): Maximum allowed timestep
        dt_start (float): Initial timestep
        n_min (int): Minimum steps before timestep increase
        f_inc (float): Factor for timestep increase when power is positive
        f_dec (float): Factor for timestep decrease when power is negative
        alpha_start (float): Initial velocity mixing parameter
        f_alpha (float): Factor for mixing parameter decrease
        cell_factor (float | None): Scaling factor for cell optimization.
            If None, defaults to number of atoms per batch
        hydrostatic_strain (bool): Whether to only allow hydrostatic deformation
            (isotropic scaling)
        constant_volume (bool): Whether to maintain constant volume during optimization
        scalar_pressure (float): Applied external pressure in GPa

    Returns:
        tuple: A pair of functions:
            - Initialization function that creates a FrechetCellFIREState
            - Update function that performs one FIRE step with Frechet derivatives

    Notes:
        - Frechet cell parameterization uses matrix logarithm to represent cell
          deformations, which provides improved numerical properties for cell
          optimization
        - This method generally performs better than standard unit cell optimization
          for cases with large cell deformations
        - To fix the cell and only optimize atomic positions, set both
          constant_volume=True and hydrostatic_strain=True
    """
    device, dtype = model.device, model.dtype

    eps = 1e-8 if dtype == torch.float32 else 1e-16

    # Setup parameters
    params = [dt_max, dt_start, alpha_start, f_inc, f_dec, f_alpha, n_min]
    dt_max, dt_start, alpha_start, f_inc, f_dec, f_alpha, n_min = [
        torch.as_tensor(p, device=device, dtype=dtype) for p in params
    ]

    def fire_init(
        state: SimState | StateDict,
        cell_factor: torch.Tensor | None = cell_factor,
        scalar_pressure: float = scalar_pressure,
        dt_start: float = dt_start,
        alpha_start: float = alpha_start,
    ) -> FrechetCellFIREState:
        """Initialize a batched FIRE optimization state with Frechet cell
        parameterization.

        Args:
            state: Input state as SimState object or state parameter dict
            cell_factor: Cell optimization scaling factor. If None, uses atoms per batch.
                         Single value or tensor of shape [n_batches].
            scalar_pressure: Applied pressure in energy units
            dt_start: Initial timestep per batch
            alpha_start: Initial mixing parameter per batch

        Returns:
            FrechetCellFIREState with initialized optimization tensors
        """
        if not isinstance(state, SimState):
            state = SimState(**state)

        # Get dimensions
        n_batches = state.n_batches

        # Setup cell_factor
        if cell_factor is None:
            # Count atoms per batch
            _, counts = torch.unique(state.batch, return_counts=True)
            cell_factor = counts.to(dtype=dtype)

        if isinstance(cell_factor, int | float):
            # Use same factor for all batches
            cell_factor = torch.full(
                (state.n_batches,), cell_factor, device=device, dtype=dtype
            )

        # Reshape to (n_batches, 1, 1) for broadcasting
        cell_factor = cell_factor.view(-1, 1, 1)

        # Setup pressure tensor
        pressure = scalar_pressure * torch.eye(3, device=device, dtype=dtype)
        pressure = pressure.unsqueeze(0).expand(n_batches, -1, -1)

        # Get initial forces and energy from model
        model_output = model(state)

        energy = model_output["energy"]  # [n_batches]
        forces = model_output["forces"]  # [n_total_atoms, 3]
        stress = model_output["stress"]  # [n_batches, 3, 3]

        # Calculate initial cell positions using matrix logarithm
        # Calculate current deformation gradient (identity matrix at start)
        cur_deform_grad = DeformGradMixin._deform_grad(  # noqa: SLF001
            state.row_vector_cell, state.row_vector_cell
        )  # shape: (n_batches, 3, 3)

        # For identity matrix, logm gives zero matrix
        # Initialize cell positions to zeros
        cell_positions = torch.zeros((n_batches, 3, 3), device=device, dtype=dtype)

        # Calculate virial for cell forces
        volumes = torch.linalg.det(state.cell).view(-1, 1, 1)
        virial = -volumes * (stress + pressure)  # P is P_ext * I

        if hydrostatic_strain:
            diag_mean = torch.diagonal(virial, dim1=1, dim2=2).mean(dim=1, keepdim=True)
            virial = diag_mean.unsqueeze(-1) * torch.eye(3, device=device).unsqueeze(
                0
            ).expand(n_batches, -1, -1)

        if constant_volume:
            diag_mean = torch.diagonal(virial, dim1=1, dim2=2).mean(dim=1, keepdim=True)
            virial = virial - diag_mean.unsqueeze(-1) * torch.eye(
                3, device=device
            ).unsqueeze(0).expand(n_batches, -1, -1)

        # Calculate UCF-style cell gradient
        ucf_cell_grad = torch.zeros_like(virial)
        for b in range(n_batches):
            ucf_cell_grad[b] = virial[b] @ torch.linalg.inv(cur_deform_grad[b].T)
        # Calculate cell forces using Frechet derivative approach (all zeros for identity)
        cell_forces = ucf_cell_grad / cell_factor

        # Sum masses per batch
        batch_counts = torch.bincount(state.batch)
        cell_masses = torch.segment_reduce(
            state.masses, reduce="sum", lengths=batch_counts
        )  # shape: (n_batches,)
        cell_masses = cell_masses.unsqueeze(-1).expand(-1, 3)  # shape: (n_batches, 3)

        # Setup parameters
        dt_start = torch.full((n_batches,), dt_start, device=device, dtype=dtype)
        alpha_start = torch.full((n_batches,), alpha_start, device=device, dtype=dtype)
        n_pos = torch.zeros((n_batches,), device=device, dtype=torch.int32)

        # Create initial state
        return FrechetCellFIREState(
            # Copy SimState attributes
            positions=state.positions,
            masses=state.masses,
            cell=state.cell,
            atomic_numbers=state.atomic_numbers,
            batch=state.batch,
            pbc=state.pbc,
            # New attributes
            velocities=torch.zeros_like(state.positions),
            forces=forces,
            energy=energy,
            stress=stress,
            # Cell attributes
            cell_positions=cell_positions,
            cell_velocities=torch.zeros((n_batches, 3, 3), device=device, dtype=dtype),
            cell_forces=cell_forces,
            cell_masses=cell_masses,
            # Optimization attributes
            reference_cell=state.cell.clone(),
            cell_factor=cell_factor,
            pressure=pressure,
            dt=dt_start,
            alpha=alpha_start,
            n_pos=n_pos,
            hydrostatic_strain=hydrostatic_strain,
            constant_volume=constant_volume,
        )

    def fire_step(  # noqa: PLR0915
        state: FrechetCellFIREState,
        alpha_start: float = alpha_start,
        dt_start: float = dt_start,
    ) -> FrechetCellFIREState:
        """Perform one FIRE optimization step for batched atomic systems with
        Frechet cell parameterization.

        Implements one step of the Fast Inertial Relaxation Engine (FIRE)
        algorithm for optimizing atomic positions and unit cell parameters
        using matrix logarithm parameterization for the cell degrees of freedom.

        Args:
            state: Current optimization state containing atomic and cell parameters
            alpha_start: Initial mixing parameter for velocity update
            dt_start: Initial timestep for velocity Verlet integration

        Returns:
            Updated state after performing one FIRE step with Frechet cell derivatives
        """
        n_batches = state.n_batches

        # Setup parameters
        dt_start = torch.full((n_batches,), dt_start, device=device, dtype=dtype)
        alpha_start = torch.full((n_batches,), alpha_start, device=device, dtype=dtype)

        # Calculate current deformation gradient
        cur_deform_grad = state.deform_grad()  # shape: (n_batches, 3, 3)

        # Calculate log of deformation gradient
        deform_grad_log = torch.zeros_like(cur_deform_grad)
        for b in range(n_batches):
            deform_grad_log[b] = tsm.matrix_log_33(cur_deform_grad[b])

        # Scale to get cell positions
        cell_positions = deform_grad_log * state.cell_factor

        # Velocity Verlet first half step (v += 0.5*a*dt)
        atom_wise_dt = state.dt[state.batch].unsqueeze(-1)
        cell_wise_dt = state.dt.unsqueeze(-1).unsqueeze(-1)

        state.velocities += 0.5 * atom_wise_dt * state.forces / state.masses.unsqueeze(-1)
        state.cell_velocities += (
            0.5 * cell_wise_dt * state.cell_forces / state.cell_masses.unsqueeze(-1)
        )

        # Split positions and forces into atomic and cell components
        atomic_positions = state.positions  # shape: (n_atoms, 3)

        # Update atomic and cell positions
        atomic_positions_new = atomic_positions + atom_wise_dt * state.velocities
        cell_positions_new = cell_positions + cell_wise_dt * state.cell_velocities

        # Convert cell positions to deformation gradient
        deform_grad_log_new = cell_positions_new / state.cell_factor

        # deform_grad_new = torch.zeros_like(deform_grad_log_new)
        # for b in range(n_batches):
        #    deform_grad_new[b] = expm.apply(deform_grad_log_new[b])

        deform_grad_new = torch.matrix_exp(deform_grad_log_new)

        # Update cell with deformation gradient
        new_row_vector_cell = torch.bmm(
            state.reference_row_vector_cell, deform_grad_new.transpose(1, 2)
        )

        # Update state with new positions and cell
        state.positions = atomic_positions_new
        state.row_vector_cell = new_row_vector_cell
        state.cell_positions = cell_positions_new

        # Get new forces and energy
        results = model(state)
        state.energy = results["energy"]

        # Combine new atomic forces and cell forces
        forces = results["forces"]
        stress = results["stress"]

        state.forces = forces
        state.stress = stress

        # Calculate virial
        volumes = torch.linalg.det(state.cell).view(-1, 1, 1)
        virial = -volumes * (stress + state.pressure)  # P is P_ext * I
        if state.hydrostatic_strain:
            diag_mean = torch.diagonal(virial, dim1=1, dim2=2).mean(dim=1, keepdim=True)
            virial = diag_mean.unsqueeze(-1) * torch.eye(3, device=device).unsqueeze(
                0
            ).expand(n_batches, -1, -1)
        if state.constant_volume:
            diag_mean = torch.diagonal(virial, dim1=1, dim2=2).mean(dim=1, keepdim=True)
            virial = virial - diag_mean.unsqueeze(-1) * torch.eye(
                3, device=device
            ).unsqueeze(0).expand(n_batches, -1, -1)

        # Perform batched matrix multiplication
        ucf_cell_grad = torch.bmm(
            virial, torch.linalg.inv(torch.transpose(deform_grad_new, 1, 2))
        )

        # Pre-compute all 9 direction matrices
        directions = torch.zeros((9, 3, 3), device=device, dtype=dtype)
        for idx, (mu, nu) in enumerate([(i, j) for i in range(3) for j in range(3)]):
            directions[idx, mu, nu] = 1.0

        # Calculate cell forces batch by batch
        cell_forces = torch.zeros_like(ucf_cell_grad)
        for b in range(n_batches):
            # Calculate all 9 Frechet derivatives at once
            expm_derivs = torch.stack(
                [
                    tsm.expm_frechet(
                        deform_grad_log_new[b], direction, compute_expm=False
                    )
                    for direction in directions
                ]
            )

            # Calculate all 9 cell forces components
            forces_flat = torch.sum(
                expm_derivs * ucf_cell_grad[b].unsqueeze(0), dim=(1, 2)
            )
            cell_forces[b] = forces_flat.reshape(3, 3)

        # Scale by cell_factor
        cell_forces = cell_forces / state.cell_factor
        state.cell_forces = cell_forces

        # Velocity Verlet second half step (v += 0.5*a*dt)
        state.velocities += 0.5 * atom_wise_dt * state.forces / state.masses.unsqueeze(-1)
        state.cell_velocities += (
            0.5 * cell_wise_dt * state.cell_forces / state.cell_masses.unsqueeze(-1)
        )

        # Calculate power (F·V) for atoms
        atomic_power = (state.forces * state.velocities).sum(dim=1)  # [n_atoms]
        atomic_power_per_batch = torch.zeros(
            n_batches, device=device, dtype=atomic_power.dtype
        )
        atomic_power_per_batch.scatter_add_(
            dim=0, index=state.batch, src=atomic_power
        )  # [n_batches]

        # Calculate power for cell DOFs
        cell_power = (state.cell_forces * state.cell_velocities).sum(
            dim=(1, 2)
        )  # [n_batches]
        batch_power = atomic_power_per_batch + cell_power

        # FIRE updates for each batch
        for batch_idx in range(n_batches):
            # FIRE specific updates
            if batch_power[batch_idx] > 0:
                # Power is positive
                state.n_pos[batch_idx] += 1
                if state.n_pos[batch_idx] > n_min:
                    state.dt[batch_idx] = min(state.dt[batch_idx] * f_inc, dt_max)
                    state.alpha[batch_idx] = state.alpha[batch_idx] * f_alpha
            else:
                # Power is negative
                state.n_pos[batch_idx] = 0
                state.dt[batch_idx] = state.dt[batch_idx] * f_dec
                state.alpha[batch_idx] = alpha_start[batch_idx]
                # Reset velocities for both atoms and cell
                state.velocities[state.batch == batch_idx] = 0
                state.cell_velocities[batch_idx] = 0

        # Mix velocity and force direction using FIRE for atoms
        v_norm = torch.norm(state.velocities, dim=1, keepdim=True)
        f_norm = torch.norm(state.forces, dim=1, keepdim=True)
        batch_wise_alpha = state.alpha[state.batch].unsqueeze(-1)
        state.velocities = (
            1.0 - batch_wise_alpha
        ) * state.velocities + batch_wise_alpha * state.forces * v_norm / (f_norm + eps)

        # Mix velocity and force direction for cell DOFs
        cell_v_norm = torch.norm(state.cell_velocities, dim=(1, 2), keepdim=True)
        cell_f_norm = torch.norm(state.cell_forces, dim=(1, 2), keepdim=True)
        cell_wise_alpha = state.alpha.unsqueeze(-1).unsqueeze(-1)
        cell_mask = cell_f_norm > eps
        state.cell_velocities = torch.where(
            cell_mask,
            (1.0 - cell_wise_alpha) * state.cell_velocities
            + cell_wise_alpha * state.cell_forces * cell_v_norm / cell_f_norm,
            state.cell_velocities,
        )

        return state

    return fire_init, fire_step



@dataclass
class ASEFireState(GroupedSimState):
    """State information for batched FIRE optimization mimicking ASE's implementation.

    Uses global parameters (dt, alpha, Nsteps) for decision logic, mirroring
    ASE's behavior on single systems, but applies updates to batched tensors.

    Attributes:
        # Inherited from GroupedSimState
        positions (torch.Tensor): Atomic positions [n_atoms, 3]
        masses (torch.Tensor): Atomic masses [n_atoms]
        cell (torch.Tensor): Unit cell vectors [n_batches, 3, 3]
        pbc (bool): Periodic boundary conditions flag
        atomic_numbers (torch.Tensor): Atomic numbers [n_atoms]
        batch (torch.Tensor): Batch indices [n_atoms]
        batch_group (torch.Tensor): Group indices [n_batches]

        # Atomic quantities
        forces (torch.Tensor): Forces on atoms [n_atoms, 3]
        velocities (torch.Tensor): Atomic velocities [n_atoms, 3]
        energy (torch.Tensor): Energy per batch [n_batches]

        # ASE FIRE optimization parameters (stored per state instance)
        # Note: ASE uses single floats/ints; we store them, but logic uses them globally.
        dt: float
        alpha: float # Corresponds to ASE's 'a'
        n_steps: int # Corresponds to ASE's 'Nsteps'

        # Optional state for downhill_check
        downhill_check: bool
        energy_last: torch.Tensor | None = None    # Shape [n_batches]
        positions_last: torch.Tensor | None = None # Shape [n_atoms, 3]
        velocities_last: torch.Tensor | None = None # Shape [n_atoms, 3]

    """
    forces: torch.Tensor
    energy: torch.Tensor
    velocities: torch.Tensor

    # ASE-like global state parameters
    dt: float
    alpha: float # Corresponds to ASE's 'a'
    n_steps: int # Corresponds to ASE's 'Nsteps'

    # Optional state for downhill_check
    downhill_check: bool
    energy_last: torch.Tensor | None = None
    positions_last: torch.Tensor | None = None
    velocities_last: torch.Tensor | None = None


def ase_fire(
    model: torch.nn.Module,
    *,
    # ASE FIRE specific parameters with defaults
    dt: float = 0.1,
    maxstep: float = 0.2,
    dtmax: float = 1.0,
    n_min: int = 5,      # ASE Nmin
    f_inc: float = 1.1,    # ASE finc
    f_dec: float = 0.5,    # ASE fdec
    a_start: float = 0.1,  # ASE astart (alpha_start)
    f_a: float = 0.99,     # ASE fa (f_alpha)
    a: float | None = None,# ASE a (alpha), if None, use a_start initially
    downhill_check: bool = False,
    # position_reset_callback: Optional[Callable] = None, # Not directly transferable
) -> tuple[
    Callable[[SimState | GroupedSimState | StateDict], ASEFireState],
    Callable[[ASEFireState], ASEFireState],
]:
    """Initialize a batched FIRE optimization mimicking ASE's logic.

    Creates an optimizer that performs FIRE optimization on atomic positions,
    closely following the algorithm structure in `ase.optimize.FIRE`.

    Args:
        model (torch.nn.Module): Model that computes energies and forces.
        dt (float): Initial time step (ASE dt). Default is 0.1.
        maxstep (float): Max distance atom can move per step. Default is 0.2.
        dtmax (float): Maximum time step. Default is 1.0.
        n_min (int): Min steps before timestep increase (ASE Nmin). Default is 5.
        f_inc (float): Factor for dt increase (ASE finc). Default is 1.1.
        f_dec (float): Factor for dt decrease (ASE fdec). Default is 0.5.
        a_start (float): Initial mixing parameter (ASE astart). Default is 0.1.
        f_a (float): Factor for mixing parameter decrease (ASE fa). Default is 0.99.
        a (float | None): Current mixing parameter (ASE a). If None, uses `a_start`.
                          Default is None.
        downhill_check (bool): Whether to use explicit energy check for downhill
                               steps. Default is False.

    Returns:
        tuple: A pair of functions:
            - Initialization function `ase_fire_init`.
            - Update function `ase_fire_step`.
    """
    device, dtype = model.device, model.dtype
    eps = torch.finfo(dtype).eps # Use dtype-specific epsilon

    # Use a_start if a is not provided
    initial_alpha = a if a is not None else a_start

    def ase_fire_init(
        state: SimState | GroupedSimState | StateDict,
        # Allow overriding initial params if needed, though usually fixed
        dt_init: float = dt,
        alpha_init: float = initial_alpha,
    ) -> ASEFireState:
        """Initialize the ASE-like FIRE optimization state."""
        if isinstance(state, dict):
             # TODO: Decide how to handle dict input - require batch_group?
             # For now, assume it can be converted to SimState first
             state = SimState(**state) # Potential loss of batch_group if dict lacks it

        # Ensure state is GroupedSimState
        if not isinstance(state, GroupedSimState):
            batch_group = torch.zeros(state.n_batches, device=state.device, dtype=torch.int64)
            state = GroupedSimState(**vars(state), batch_group=batch_group)

        # Get initial forces and energy from model
        model_output = model(state)
        energy = model_output["energy"]
        forces = model_output["forces"]

        # Initial ASE state variables
        initial_velocities = torch.zeros_like(state.positions)
        n_steps_init = 0

        # Store initial state for downhill_check if enabled
        e_last_init = energy.clone() if downhill_check else None
        r_last_init = state.positions.clone() if downhill_check else None
        v_last_init = initial_velocities.clone() if downhill_check else None

        # Create initial ASEFireState
        return ASEFireState(
            # GroupedSimState attributes
            positions=state.positions.clone(),
            masses=state.masses.clone(),
            cell=state.cell.clone(),
            atomic_numbers=state.atomic_numbers.clone(),
            batch=state.batch.clone(),
            pbc=state.pbc,
            batch_group=state.batch_group.clone(),
            # New attributes
            velocities=initial_velocities,
            forces=forces,
            energy=energy,
            # ASE global state params
            dt=dt_init,
            alpha=alpha_init,
            n_steps=n_steps_init,
            # Downhill check state
            downhill_check=downhill_check,
            energy_last=e_last_init,
            positions_last=r_last_init,
            velocities_last=v_last_init,
        )

    def ase_fire_step(state: ASEFireState) -> ASEFireState:
        """Perform one ASE-like FIRE optimization step."""
        # --- Log optimizer state at start of step ---
        logger.debug(f"ASE_FIRE Step {state.n_steps} START: dt={state.dt:.6e}, alpha={state.alpha:.6e}, Nsteps={state.n_steps}")
        # ---------------------------------------------

        f = state.forces
        v = state.velocities
        current_dt = state.dt
        current_alpha = state.alpha
        current_n_steps = state.n_steps

        is_uphill = False
        if state.downhill_check:
            # Note: Requires model call *before* position update if downhill_check
            # is needed, unlike standard FIRE. This deviates slightly as we
            # calculate energy *after* the step for the *next* iteration.
            # For simplicity here, we use energy from the *previous* step.
            # A more exact implementation would need energy before the step.
            # Let's assume energy_last is sufficient for now.
            # Sticking to vf check only, as downhill_check=False is default.
            # If downhill_check is needed, this part requires rework.
            if downhill_check: # Re-check parameter passed to function
                warnings.warn("ASE-like downhill_check requires energy before step, "
                              "which is not readily available. Check disabled.", UserWarning)
                pass # Disable check for now due to implementation complexity


        # Calculate global power vf = sum(f * v)
        # torch.vdot sums all elements, equivalent to np.vdot(f.flatten(), v.flatten())
        # This matches ASE's np.vdot(f, v) behavior.
        vf = torch.vdot(f.flatten(), v.flatten())

        # --- Log power value ---
        logger.debug(f"ASE_FIRE Step {state.n_steps} CALC: vf={vf:.6e}")
        # ----------------------

        if vf > 0.0 and not is_uphill:
            # --- Downhill Step ---
            f_norm = torch.linalg.norm(f)
            v_norm = torch.linalg.norm(v)
            logger.debug(f"  Step {current_n_steps}: Downhill - vf={vf:.4e}, f_norm={f_norm:.4e}, v_norm={v_norm:.4e}")

            # Mix velocity
            if f_norm > eps and v_norm > eps:
                 v = (1.0 - current_alpha) * v + current_alpha * f * (v_norm / f_norm)
            # If f_norm or v_norm is zero, mixing term is zero, v remains (1.0 - a)*v
            elif f_norm <= eps or v_norm <= eps:
                 v = (1.0 - current_alpha) * v

            # Adapt dt and alpha if Nsteps > Nmin
            if current_n_steps > n_min:
                current_dt = min(current_dt * f_inc, dtmax)
                current_alpha *= f_a

            current_n_steps += 1
            logger.debug(f"  Step {current_n_steps} (next): New dt={current_dt:.4e}, New alpha={current_alpha:.4e}")

        else:
            # --- Uphill or Zero Power Step ---
            logger.debug(f"ASE_FIRE Step {current_n_steps}: Uphill/Zero - vf={vf:.6e}")
            v = torch.zeros_like(v)        # Reset velocities
            current_alpha = a_start        # Reset alpha to a_start
            current_dt *= f_dec            # Decrease dt
            current_n_steps = 0            # Reset Nsteps
            logger.debug(f"ASE_FIRE Step {current_n_steps} (RESET): New dt={current_dt:.6e}, New alpha={current_alpha:.6e}")

        # --- Update velocity: v += dt * f ---
        v += current_dt * f

        # --- Calculate displacement dr = dt * v ---
        dr = current_dt * v
        dr_norm_before = torch.linalg.norm(dr)

        # --- Apply global maxstep constraint ---
        normdr = torch.linalg.norm(dr)
        if normdr > maxstep:
            dr = maxstep * dr / normdr
        dr_norm_after = torch.linalg.norm(dr)
        logger.debug(f"  Step {state.n_steps}: Displacement norm: Before={dr_norm_before:.4e}, After={dr_norm_after:.4e} (maxstep={maxstep})")

        # --- Update state variables ---
        state.dt = current_dt
        state.alpha = current_alpha
        state.n_steps = current_n_steps
        state.velocities = v # Store updated velocities

        # --- Store state for downhill_check (if enabled) ---
        if state.downhill_check:
            # Store energy from *before* this step was taken
            # Cloning state.energy which holds the result from the *previous* model call
            state.energy_last = state.energy.clone()
            state.positions_last = state.positions.clone()
            # Store V *after* potential mixing but *before* the v += dt*f update?
            # ASE stores v *before* the step. 
            # Store v from the *start* of the step if downhill_check active?
            # For now, storing the final v before position update.
            state.velocities_last = state.velocities.clone()

        # --- Update positions ---
        state.positions += dr

        # --- Get new forces and energy for the *next* step ---
        results = model(state)
        state.energy = results["energy"]
        state.forces = results["forces"] # <-- ADD THIS LINE

        # --- Log optimizer state before storing updates ---
        logger.debug(f"ASE_FIRE Step {state.n_steps} END: dt={current_dt:.6e}, alpha={current_alpha:.6e}, Nsteps={current_n_steps}")
        # ------------------------------------------------

        return state

    return ase_fire_init, ase_fire_step
