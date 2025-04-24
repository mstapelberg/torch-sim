"""Nudged Elastic Band (NEB) workflow.

This module implements the Nudged Elastic Band method for finding minimum energy
paths between two given atomic configurations.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Literal
from contextlib import nullcontext

import torch

from torch_sim.autobatching import BinningAutoBatcher
from torch_sim.models.interface import ModelInterface
from torch_sim.optimizers import (
    FireState, fire, GDState, gradient_descent, FrechetCellFIREState, frechet_cell_fire
)
from torch_sim.state import SimState, concatenate_states, initialize_state
from torch_sim.trajectory import TorchSimTrajectory
from torch_sim.transforms import minimum_image_displacement
from torch_sim.typing import StateLike


logger = logging.getLogger(__name__)


@dataclass
class NEB:
    """Nudged Elastic Band (NEB) optimizer.

    Finds the minimum energy path (MEP) between an initial and final state using
    the NEB algorithm.

    Attributes:
        model (ModelInterface): The energy/force model (e.g., MACE) wrapped in a
            ModelInterface.
        n_images (int): Number of intermediate images between initial and final states.
        spring_constant (float): Spring constant connecting adjacent images.
        use_climbing_image (bool): Whether to use a climbing image.
        optimizer_type: Literal["fire", "gd", "frechet_cell_fire"]: Type of optimizer to use.
        optimizer_params: dict[str, Any]: Parameters for the optimizer.
        trajectory_filename (str | None): Filename for saving the trajectory.
        device (torch.device): Computation device.
        dtype (torch.dtype): Computation data type.
    """

    model: ModelInterface
    n_images: int
    spring_constant: float = 5.0  # eV/Ang^2, typical ASE default
    use_climbing_image: bool = False
    optimizer_type: Literal["fire", "gd", "frechet_cell_fire"] = "fire"
    optimizer_params: dict[str, Any] = field(default_factory=dict)
    trajectory_filename: str | None = None
    device: torch.device | None = None
    dtype: torch.dtype | None = None

    def __post_init__(self):
        """Initialize derived attributes."""
        if self.device is None:
            self.device = self.model.device
        if self.dtype is None:
            self.dtype = self.model.dtype

        # Initialize FIRE optimizer functions
        # self._fire_init, self._fire_step = fire(self.model, **self.fire_params)

        # Conditionally initialize optimizer functions and state type
        if self.optimizer_type == "fire":
            # TODO: Reinstate fire_params if needed, maybe via optimizer_params dict
            self._init_fn, self._step_fn = fire(self.model, **self.optimizer_params)
            self._OptimizerStateType = FireState
        elif self.optimizer_type == "frechet_cell_fire":
            # Initialize Frechet Cell FIRE, passing params. Ensure constant_volume=True is set by user.
            self._init_fn, self._step_fn = frechet_cell_fire(self.model, **self.optimizer_params)
            self._OptimizerStateType = FrechetCellFIREState
        elif self.optimizer_type == "gd":
            # Use .get() for lr with a default, in case user doesn't pass it
            self._init_fn, self._step_fn = gradient_descent(self.model, lr=self.optimizer_params.get('lr', 0.01))
            self._OptimizerStateType = GDState
        else:
            raise ValueError(f"Unsupported optimizer_type: {self.optimizer_type}")

    def _interpolate_path(self, initial_state: SimState, final_state: SimState) -> SimState:
        """Linearly interpolate the initial path between states.

        Args:
            initial_state: The starting SimState (single batch).
            final_state: The ending SimState (single batch).

        Returns:
            SimState: A single SimState containing all interpolated images, batched.

        Raises:
            ValueError: If initial and final states are incompatible (e.g., different
                number of atoms, atom types, pbc, or multiple batches).
        """
        # --- Input Validation ---
        if initial_state.n_batches != 1 or final_state.n_batches != 1:
            raise ValueError("Initial and final states must be single-batch SimStates.")
        if initial_state.n_atoms != final_state.n_atoms:
            raise ValueError(
                f"Initial ({initial_state.n_atoms}) and final ({final_state.n_atoms}) "
                "states must have the same number of atoms."
            )
        if not torch.equal(initial_state.atomic_numbers, final_state.atomic_numbers):
            # Comparing floats might be tricky, but atomic numbers should be exact
            raise ValueError("Initial and final states must have the same atom types.")
        if initial_state.pbc != final_state.pbc:
            # TODO: Could potentially support different PBCs, but complex for NEB.
            raise ValueError("Initial and final states must have the same PBC setting.")
        # For fixed-cell NEB, cells should ideally be identical. Warn if not?
        # if not torch.allclose(initial_state.cell, final_state.cell):
        #     logger.warning("Initial and final states have different cell shapes.")

        n_atoms_per_image = initial_state.n_atoms

        # --- Interpolation ---
        initial_pos = initial_state.positions
        final_pos = final_state.positions

        # Calculate displacement (simple subtraction, ignoring MIC for now)
        # TODO: Consider using MIC for displacement calculation if needed.
        # displacement = final_pos - initial_pos # Old simple subtraction

        # Calculate displacement using Minimum Image Convention
        displacement = minimum_image_displacement(
            dr = final_pos - initial_pos,
            cell = initial_state.cell[0], # Use cell from initial state
            pbc = initial_state.pbc
        )
        # Ensure shape is correct [n_atoms, 3]
        displacement = displacement.reshape(n_atoms_per_image, 3)

        # Generate interpolation factors (e.g., for n_images=3: 0.25, 0.5, 0.75)
        factors = torch.linspace(
            0.0, 1.0, steps=self.n_images + 2, device=self.device, dtype=self.dtype
        )[1:-1]  # Exclude 0.0 and 1.0
        factors = factors.view(-1, 1, 1)  # Shape: [n_images, 1, 1]

        # Calculate interpolated positions: initial + factor * displacement
        # Broadcasting: [N_atoms, 3] + [N_images, 1, 1] * [N_atoms, 3] -> [N_images, N_atoms, 3]
        interpolated_pos = initial_pos.unsqueeze(0) + factors * displacement.unsqueeze(0)

        # Reshape to [n_images * n_atoms_per_image, 3]
        all_positions = interpolated_pos.reshape(-1, 3)

        # --- Create Batched State ---
        # Repeat other attributes for each image
        all_atomic_numbers = initial_state.atomic_numbers.repeat(self.n_images)
        all_masses = initial_state.masses.repeat(self.n_images)
        # Use initial state's cell, repeated for each image
        all_cells = initial_state.cell.repeat(self.n_images, 1, 1) # Shape: [n_images, 3, 3]

        # Create batch tensor: [0, 0, ..., 1, 1, ..., n_images-1, ...]
        batch_indices = torch.arange(self.n_images, device=self.device, dtype=torch.int64)
        all_batch = torch.repeat_interleave(batch_indices, repeats=n_atoms_per_image)

        return SimState(
            positions=all_positions,
            atomic_numbers=all_atomic_numbers,
            masses=all_masses,
            cell=all_cells,
            pbc=initial_state.pbc,
            batch=all_batch,
        )

    def _calculate_neb_forces(
        self,
        path_state: SimState,
        true_forces: torch.Tensor,
        true_energies: torch.Tensor,
        initial_energy: torch.Tensor,
        final_energy: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate the NEB forces (true force perpendicular + spring force parallel).

        Args:
            path_state: SimState containing the full path (initial + intermediate + final).
                        Assumes batches are ordered 0=initial, 1..n=intermediate, n+1=final.
            true_forces: Tensor of shape [n_movable_atoms, 3] containing the forces from
                         the potential energy model for the *intermediate* images only.
            true_energies: Tensor of shape [n_images] containing the potential energies
                           for the *intermediate* images only.
            initial_energy: Potential energy of the initial state (scalar tensor).
            final_energy: Potential energy of the final state (scalar tensor).

        Returns:
            Tensor: NEB forces for the intermediate images, shape [n_movable_atoms, 3].
        """
        n_total_images = path_state.n_batches
        n_intermediate_images = n_total_images - 2
        assert n_intermediate_images == self.n_images
        n_atoms_per_image = path_state.n_atoms // n_total_images

        # --- Reshape inputs ---
        # Positions for all images: [n_total_images, n_atoms, 3]
        all_pos = path_state.positions.reshape(n_total_images, n_atoms_per_image, 3)
        # True forces for intermediate images: [n_images, n_atoms, 3]
        true_forces_reshaped = true_forces.reshape(
            n_intermediate_images, n_atoms_per_image, 3
        )
        # Cell vectors (assuming fixed cell for now, take from first batch)
        # Shape [3, 3]
        cell = path_state.cell[0]
        pbc = path_state.pbc

        # --- Get Energies for Tangent Calculation ---
        # Need energies for all images (0 to n+1)
        # Get endpoint energies - requires running the model on them!
        # TODO: This is inefficient! Energies should ideally be pre-calculated or passed.
        # For now, let's assume they are roughly equal to neighbors if not available.
        # A better approach is needed here. Maybe pass initial/final states?
        all_energies = torch.cat(
            [
                initial_energy.unsqueeze(0),
                true_energies,
                final_energy.unsqueeze(0),
            ]
        )

        # --- Calculate Tangents (tau) --- 
        tangents = torch.zeros_like(true_forces_reshaped)
        # Vectors between adjacent images (using MIC)
        # d[i] = R_{i+1} - R_i, shape [n_images+1, n_atoms, 3]
        displacements = minimum_image_displacement(
            dr = all_pos[1:] - all_pos[:-1],
            cell = cell,
            pbc = pbc,
        )
        # Ensure displacements shape matches expectations if MIC function altered it.
        displacements = displacements.reshape(n_total_images - 1, n_atoms_per_image, 3)

        for i in range(n_intermediate_images): # Loop over intermediate images 1 to N
            img_idx = i + 1 # Index in all_pos and all_energies

            E_prev = all_energies[img_idx - 1]
            E_curr = all_energies[img_idx]
            E_next = all_energies[img_idx + 1]

            d_im1_i = displacements[img_idx - 1] # R_i - R_{i-1}
            d_i_ip1 = displacements[img_idx]     # R_{i+1} - R_i

            # Tangent selection based on energy landscape (VTST/ASE style)
            if E_next > E_curr > E_prev:
                tau = d_i_ip1
            elif E_prev > E_curr > E_next:
                tau = d_im1_i
            else:
                # Saddle point or minimum: use higher energy side vector, normalized
                delta_E_prev = abs(E_curr - E_prev)
                delta_E_next = abs(E_next - E_curr)
                if E_next > E_prev:
                    tau = d_i_ip1 * delta_E_prev + d_im1_i * delta_E_next
                else:
                     # Corrected weighting for E_next <= E_prev case
                     tau = d_i_ip1 * delta_E_next + d_im1_i * delta_E_prev
                # Original formulation based just on position difference:
                # if E_next > E_prev:
                #     tau = d_i_ip1
                # else:
                #     tau = d_im1_i

            # Normalize the tangent vector (sum over atoms and dims)
            tau_norm = torch.sqrt((tau**2).sum(dim=(-1, -2), keepdim=True))
            # Avoid division by zero if tangent is zero (e.g., identical images)
            if tau_norm.item() > 1e-10:
                tangents[i] = tau / tau_norm
            # else: tangent remains zero

        # --- Calculate NEB Forces --- 
        neb_forces = torch.zeros_like(true_forces_reshaped)

        # Calculate perpendicular component of true force
        # F_perp = F_true - (F_true . tau) * tau
        # Dot product (sum over atoms and dims): [n_images]
        F_true_dot_tau = (true_forces_reshaped * tangents).sum(dim=(-1, -2), keepdim=True)
        F_perp = true_forces_reshaped - F_true_dot_tau * tangents

        # Calculate parallel component of spring force
        # F_spring_par = k * (|R_{i+1}-R_i| - |R_i-R_{i-1}|) * tau_i
        # Segment lengths (scalar magnitude per segment): [n_images+1]
        segment_lengths = torch.sqrt((displacements**2).sum(dim=(-1, -2)))
        # Spring force magnitude (scalar per intermediate image): [n_images]
        F_spring_mag = self.spring_constant * (segment_lengths[1:] - segment_lengths[:-1])
        # Project onto tangent: [n_images, 1, 1]
        F_spring_par = F_spring_mag.view(-1, 1, 1) * tangents

        # Combine: NEB force = F_perp + F_spring_par
        neb_forces = F_perp + F_spring_par

        # --- Handle Climbing Image --- 
        if self.use_climbing_image and n_intermediate_images > 0:
            # Find index of highest energy image among intermediates
            climbing_image_idx = torch.argmax(true_energies).item()
            # Invert the true force component along the tangent for this image
            # F_climb = F_true - 2 * (F_true . tau) * tau
            F_climb = true_forces_reshaped[climbing_image_idx] - \
                      2 * F_true_dot_tau[climbing_image_idx] * tangents[climbing_image_idx]
            neb_forces[climbing_image_idx] = F_climb


        logger.debug(f"  Max True Force Mag: {torch.sqrt((true_forces_reshaped**2).sum(dim=-1)).max().item():.4f}")
        logger.debug(f"  Max F_perp Mag: {torch.sqrt((F_perp**2).sum(dim=-1)).max().item():.4f}")
        logger.debug(f"  Max F_spring_par Mag: {torch.sqrt((F_spring_par**2).sum(dim=-1)).max().item():.4f}")
        logger.debug(f"  Max NEB Force Mag: {torch.sqrt((neb_forces**2).sum(dim=-1)).max().item():.4f}")

        # --- Reshape output --- 
        return neb_forces.reshape(-1, 3) # [n_movable_atoms, 3]

    def run(
        self,
        initial_system: StateLike,
        final_system: StateLike,
        max_steps: int = 100,
        fmax: float = 0.05,
        # TODO: add convergence criteria, batching options, output frequency etc.
    ) -> SimState:  # Or maybe return trajectory?
        """Run the NEB optimization.

        Args:
            initial_system: The starting configuration (ASE Atoms, SimState, etc.).
            final_system: The ending configuration.
            max_steps: Maximum number of optimization steps.
            fmax: Convergence criterion for the maximum NEB force per atom (eV/Ang).

        Returns:
            The optimized NEB path (SimState containing all images).
        """
        logger.info("Starting NEB optimization")

        # 1. Initialize initial and final states
        initial_state = initialize_state(initial_system, self.device, self.dtype)
        final_state = initialize_state(final_system, self.device, self.dtype)
        # TODO: Add checks (e.g., same number of atoms, atom types)

        # 1b. Calculate endpoint energies/forces (needed for tangent calculation)
        # Note: Forces aren't strictly needed here but model usually returns both
        logger.info("Calculating endpoint energies...")
        endpoint_states = concatenate_states([initial_state, final_state])
        endpoint_output = self.model(endpoint_states)
        initial_energy = endpoint_output["energy"][0]
        final_energy = endpoint_output["energy"][1]
        logger.info(f"Initial Energy: {initial_energy:.4f}, Final Energy: {final_energy:.4f}")

        # 2. Create initial interpolated path (movable images only)
        interpolated_images = self._interpolate_path(initial_state, final_state)

        # 3. Initialize FIRE optimizer state for the movable images
        # Use the generic initializer and state type
        opt_state: self._OptimizerStateType = self._init_fn(interpolated_images)

        # 4. Optimization loop
        logger.info(f"Running NEB for max {max_steps} steps or fmax < {fmax} eV/Ang.")

        # Context manager for trajectory writing
        traj_context = (
            TorchSimTrajectory(self.trajectory_filename, mode="w")
            if self.trajectory_filename
            else nullcontext() # Use a dummy context if no filename
        )

        with traj_context as traj:
            for step in range(max_steps):
                # a. Get current true forces and energies
                true_forces = opt_state.forces
                true_energies = opt_state.energy

                # b. Calculate NEB forces
                full_path_state_calc = concatenate_states(
                    [initial_state, opt_state, final_state]
                )
                neb_forces = self._calculate_neb_forces(
                    full_path_state_calc, true_forces, true_energies, initial_energy, final_energy
                )

                # c. Update the forces in the FIRE state object with NEB forces
                opt_state.forces = neb_forces

                # d. Perform FIRE optimization step
                # Use the generic step function
                opt_state = self._step_fn(opt_state)
                logger.debug(f"  Max True Force Mag (after step): {torch.sqrt((opt_state.forces**2).sum(dim=-1)).max().item():.4f}")

                # e. Write to trajectory (if enabled)
                if self.trajectory_filename is not None: # Use explicit check
                    # Create the full path state for writing (including endpoints)
                    current_full_path = concatenate_states(
                        [initial_state, opt_state, final_state]
                    )
                    # Write arrays directly using traj.write_arrays
                    data_to_write = {
                        "positions": current_full_path.positions
                    }
                    if step == 0: # Write static data only on the first step
                        # Assuming fixed cell NEB, cell is static
                        data_to_write["cell"] = current_full_path.cell
                        # These should also be static for the whole band
                        data_to_write["atomic_numbers"] = current_full_path.atomic_numbers
                        data_to_write["masses"] = current_full_path.masses
                        # Convert bool to tensor for saving
                        data_to_write["pbc"] = torch.tensor(current_full_path.pbc)
                        # Save the batch tensor to map atoms to images
                        data_to_write["image_indices"] = current_full_path.batch

                    traj.write_arrays(data_to_write, steps=step)

                # f. Check convergence
                max_force_magnitude = torch.sqrt((neb_forces**2).sum(dim=-1)).max()
                max_intermediate_energy = opt_state.energy.max()
                logger.info(
                    f"Step {step+1:4d}:  Max Force = {max_force_magnitude:.4f}   Max Energy = {max_intermediate_energy:.4f}"
                    # f"Energy = {fire_state.energy.mean():.4f} eV (mean per image), " # Removed mean energy for brevity
                )
                if max_force_magnitude < fmax:
                    logger.info("NEB optimization converged.")
                    break
            else:  # Loop finished without break
                logger.warning("NEB optimization did not converge within max_steps.")

        # 5. Return the final path (including endpoints)
        final_path_state = concatenate_states(
            [initial_state, opt_state, final_state]
        )
        return final_path_state 