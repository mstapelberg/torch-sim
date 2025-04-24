"""Nudged Elastic Band (NEB) workflow.

This module implements the Nudged Elastic Band method for finding minimum energy
paths between two given atomic configurations.
"""

import logging
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Literal

import torch

from torch_sim.models.interface import ModelInterface
from torch_sim.optimizers import (
    FireState,
    FrechetCellFIREState,
    GDState,
    fire,
    frechet_cell_fire,
    gradient_descent,
)
from torch_sim.state import SimState, concatenate_states, initialize_state
from torch_sim.trajectory import TorchSimTrajectory
from torch_sim.transforms import minimum_image_displacement
from torch_sim.typing import StateLike


logger = logging.getLogger(__name__)

# Add epsilon for numerical stability
_EPS = torch.finfo(torch.float64).eps


@dataclass
class NEB:
    """Nudged Elastic Band (NEB) optimizer.

    Finds the minimum energy path (MEP) between an initial and final state using
    the NEB algorithm.

    Attributes:
        model: The energy/force model (e.g., MACE) wrapped in a ModelInterface.
        n_images: Number of intermediate images between initial and final states.
        spring_constant: Spring constant connecting adjacent images (eV/Ang^2).
        use_climbing_image: Whether to use a climbing image.
        optimizer_type: Type of optimizer to use.
        optimizer_params: Parameters for the chosen optimizer.
        trajectory_filename: Optional filename for saving the NEB trajectory.
        device: Computation device (e.g., 'cpu', 'cuda'). If None, uses model device.
        dtype: Computation data type (e.g., torch.float32). If None, uses model dtype.
    """

    model: ModelInterface
    n_images: int
    spring_constant: float = 0.1  # eV/Ang^2, typical ASE default
    use_climbing_image: bool = False
    optimizer_type: Literal["fire", "gd", "frechet_cell_fire"] = "fire"
    optimizer_params: dict[str, Any] = field(default_factory=dict)
    trajectory_filename: str | None = None
    device: torch.device | None = None
    dtype: torch.dtype | None = None

    def __post_init__(self) -> None:
        """Initializes device, dtype, and optimizer functions after dataclass creation."""
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
            # Initialize Frechet Cell FIRE, passing params.
            # Ensure constant_volume=True is set by user.
            self._init_fn, self._step_fn = frechet_cell_fire(
                self.model, **self.optimizer_params
            )
            self._OptimizerStateType = FrechetCellFIREState
        elif self.optimizer_type == "gd":
            # Use .get() for lr with a default, in case user doesn't pass it
            self._init_fn, self._step_fn = gradient_descent(
                self.model, lr=self.optimizer_params.get("lr", 0.01)
            )
            self._OptimizerStateType = GDState
        else:
            raise ValueError(f"Unsupported optimizer_type: {self.optimizer_type}")

    def _interpolate_path(self, initial_state: SimState, final_state: SimState) -> SimState:
        """Linearly interpolate the initial path between states using MIC.

        Generates `n_images` intermediate states between the initial and final states
        by linear interpolation of atomic positions, respecting periodic boundary
        conditions via the Minimum Image Convention (MIC).

        Args:
            initial_state (SimState): The starting SimState (must be single-batch).
            final_state (SimState): The ending SimState (must be single-batch).

        Returns:
            SimState: A single SimState containing all interpolated intermediate
                images, batched together. The batch index corresponds to the image
                index (0 to n_images-1).

        Raises:
            ValueError: If initial and final states are incompatible (e.g., different
                number of atoms, atom types, PBC settings, or if they are not
                single-batch states).
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

        n_atoms_per_image = initial_state.n_atoms

        # --- Interpolation ---
        initial_pos = initial_state.positions
        final_pos = final_state.positions

        # Calculate displacement using Minimum Image Convention
        displacement = minimum_image_displacement(
            dr=final_pos - initial_pos,
            cell=initial_state.cell[0], # Use cell from initial state
            pbc=initial_state.pbc
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
        batch_indices = torch.arange(
            self.n_images, device=self.device, dtype=torch.int64
        )
        all_batch = torch.repeat_interleave(batch_indices, repeats=n_atoms_per_image)

        return SimState(
            positions=all_positions,
            atomic_numbers=all_atomic_numbers,
            masses=all_masses,
            cell=all_cells,
            pbc=initial_state.pbc,
            batch=all_batch,
        )

    def _compute_tangents( # noqa: C901
        self,
        all_pos: torch.Tensor, # Shape: [n_total_images, n_atoms, 3]
        all_energies: torch.Tensor, # Shape: [n_total_images]
        cell: torch.Tensor, # Shape: [3, 3]
        *, # Make pbc keyword-only
        pbc: bool,
    ) -> torch.Tensor:
        """Compute normalized tangent vectors for intermediate NEB images.

        Implements the improved tangent estimate of Henkelman and Jónsson (2000)
        to determine the local tangent direction at each intermediate image based
        on the positions and energies of its neighbors.

        Args:
            all_pos (torch.Tensor): Atomic configurations for all images in the path
                (initial + intermediate + final), shape [n_total_images, n_atoms, 3].
            all_energies (torch.Tensor): Potential energy of each image, shape
                [n_total_images].
            cell (torch.Tensor): Unit cell vectors (shape [3, 3]), assumed constant
                for the path.
            pbc (bool): Flag indicating if periodic boundary conditions are active.

        Returns:
            torch.Tensor: Normalized local tangent vectors for the intermediate
                images only, shape [n_images, n_atoms, 3]. Tangents are zero for
                numerically identical adjacent images.
        """
        n_total_images, n_atoms_per_image, _ = all_pos.shape
        n_intermediate_images = n_total_images - 2
        device = all_pos.device
        dtype = all_pos.dtype

        # Initialize tangents for intermediate images only
        tangents = torch.zeros(
            (n_intermediate_images, n_atoms_per_image, 3), device=device, dtype=dtype
        )

        # Calculate displacements between adjacent images using MIC
        # dR_forward[i] = R_{i+1} - R_i
        displacements = minimum_image_displacement(
            dr=all_pos[1:] - all_pos[:-1], cell=cell, pbc=pbc
        )
        # Ensure shape is correct after MIC if needed
        displacements = displacements.reshape(n_total_images - 1, n_atoms_per_image, 3)

        # Energy differences V_{i+1} - V_i
        dE_forward = all_energies[1:] - all_energies[:-1] # Shape: [n_total_images - 1]

        # Compute tangents for intermediate images (indices 1 to N in all_pos)
        for i in range(n_intermediate_images):
            img_idx = i + 1  # Index in all_pos, all_energies

            # Displacements adjacent to image `img_idx`
            # Note: displacements[k] is R_{k+1} - R_k
            dR_plus = displacements[img_idx]     # R_{i+1} - R_i (where i = img_idx)
            dR_minus = displacements[img_idx - 1]  # R_i - R_{i-1} (where i = img_idx)

            # Energy differences adjacent to image `img_idx`
            dE_plus = dE_forward[img_idx]     # V_{i+1} - V_i
            dE_minus = dE_forward[img_idx - 1]  # V_i - V_{i-1}

            # Select tangent based on energy profile (Henkelman & Jónsson criteria)
            tangent_i = torch.zeros_like(dR_plus)

            # Ascending segment (minimum)
            if dE_plus > 0 and dE_minus > 0:
                # Use ternary operator for simple assignment
                tangent_i = dR_plus if dE_plus > dE_minus else dR_minus
            # Descending segment (maximum)
            elif dE_plus < 0 and dE_minus < 0:
                # Weight by energy difference magnitude (originally absolute value based in ref code)
                # Simplified version: use lower energy difference direction
                # Let's use the reference code's logic more closely:
                # if abs(dE_plus) < abs(dE_minus): # Towards lower energy drop
                #     tangent_i = dR_plus
                # else:
                #     tangent_i = dR_minus
                # Alternative based on reference code logic:
                # Weight lower-energy direction
                # This implementation seems slightly different from reference, let's try that:
                abs_dE_plus = abs(dE_plus)
                abs_dE_minus = abs(dE_minus)
                if torch.isclose(abs_dE_plus, abs_dE_minus, rtol=1e-6):
                     # Symmetric max: bisect angle (normalize sum of unit vectors)
                     norm_plus = torch.linalg.norm(dR_plus)
                     norm_minus = torch.linalg.norm(dR_minus)
                     if norm_plus > _EPS and norm_minus > _EPS:
                         tangent_i = (dR_plus / norm_plus) + (dR_minus / norm_minus)
                     # Handle cases where one norm is zero (e.g., duplicate image)
                     elif norm_plus > _EPS:
                         tangent_i = dR_plus / norm_plus
                     elif norm_minus > _EPS:
                         tangent_i = dR_minus / norm_minus
                     # else: tangent_i remains zero if both norms are zero
                elif abs_dE_plus < abs_dE_minus:
                     tangent_i = dR_plus
                else:
                     tangent_i = dR_minus

            # Uphill slope
            elif dE_plus > 0 and dE_minus <= 0: # Modified condition slightly for plateaus
                tangent_i = dR_plus
            # Downhill slope
            elif dE_plus <= 0 and dE_minus > 0: # Modified condition slightly for plateaus
                tangent_i = dR_minus
            # Plateau or unexpected case (should ideally not happen in smooth path)
            # Fallback based on magnitude (consistent with reference code fallback)
            elif abs(dE_plus) > abs(dE_minus):
                tangent_i = dR_plus
            else:
                tangent_i = dR_minus


            # Normalize the tangent vector for the image
            # Sum over atoms and dims: [1]
            # Use torch.linalg.norm for clarity and potential stability
            norm_i = torch.linalg.norm(tangent_i)
            if norm_i > _EPS:
                 tangents[i] = tangent_i / norm_i
            # else: tangent remains zero if norm is too small

        return tangents

    def _calculate_neb_forces(
        self,
        path_state: SimState,
        true_forces: torch.Tensor,
        true_energies: torch.Tensor,
        initial_energy: torch.Tensor,
        final_energy: torch.Tensor,
        step: int,
    ) -> torch.Tensor:
        """Calculate the NEB forces for intermediate images.

        The NEB force is composed of the true force perpendicular to the path tangent
        and the spring force parallel to the path tangent. Handles climbing image
        force modification if enabled.

        Args:
            path_state (SimState): SimState containing the full path (initial +
                intermediate + final images). Batches are assumed to be ordered.
            true_forces (torch.Tensor): Forces from the potential energy model for
                the *intermediate* images only, shape [n_movable_atoms, 3].
            true_energies (torch.Tensor): Potential energies for the *intermediate*
                images only, shape [n_images].
            initial_energy (torch.Tensor): Potential energy of the initial state
                (scalar tensor).
            final_energy (torch.Tensor): Potential energy of the final state
                (scalar tensor).
            step (int): Current optimization step number (used for climbing image delay).

        Returns:
            torch.Tensor: Calculated NEB forces for the intermediate images, ready to
                be passed to the optimizer, shape [n_movable_atoms, 3].
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
        cell = path_state.cell[0] # Shape [3, 3]
        pbc = path_state.pbc

        # --- Get Energies for Tangent Calculation ---
        all_energies = torch.cat(
            [
                initial_energy.unsqueeze(0),
                true_energies,
                final_energy.unsqueeze(0),
            ]
        )

        # --- Calculate Tangents (tau) using the improved method ---
        # tangents shape: [n_images, n_atoms, 3]
        tangents = self._compute_tangents(all_pos, all_energies, cell, pbc=pbc)

        # --- Calculate Displacements for Spring Force ---
        # Recalculate here or reuse from _compute_tangents if efficient
        # For clarity, recalculate:
        displacements = minimum_image_displacement(
            dr=all_pos[1:] - all_pos[:-1], cell=cell, pbc=pbc
        )
        displacements = displacements.reshape(n_total_images - 1, n_atoms_per_image, 3)


        # --- Calculate NEB Force Components ---

        # 1. Perpendicular component of true force
        # F_perp = F_true - (F_true . tau) * tau
        # Dot product (sum over atoms and dims): [n_images]
        F_true_dot_tau = (true_forces_reshaped * tangents).sum(
            dim=(-1, -2), keepdim=True
        )
        F_perp = true_forces_reshaped - F_true_dot_tau * tangents

        # 2. Parallel component of spring force
        # F_spring_par = k * (|R_{i+1}-R_i| - |R_i-R_{i-1}|) * tau_i
        # Segment lengths (scalar magnitude per segment): [n_images+1]
        # segment_lengths = torch.sqrt((displacements**2).sum(dim=(-1, -2))) # Old way
        segment_lengths = torch.linalg.norm(
            displacements, dim=(-1, -2)
        ) # Cleaner way [n_total_images-1]
        # Spring force magnitude (scalar per intermediate image): [n_images]
        F_spring_mag = self.spring_constant * (
            segment_lengths[1:] - segment_lengths[:-1]
        )
        # Project onto tangent: [n_images, 1, 1] -> [n_images, n_atoms, 3]
        F_spring_par = F_spring_mag.view(-1, 1, 1) * tangents

        # --- Combine Components for NEB Force ---
        # Initial NEB force = F_perp + F_spring_par
        neb_forces = F_perp + F_spring_par

        # --- Handle Climbing Image ---
        climbing_delay_steps = 10 # Example value
        if self.use_climbing_image and n_intermediate_images > 0 and step >= climbing_delay_steps: # Check step number
             # Find index of highest energy image among intermediates
             climbing_image_idx = torch.argmax(true_energies).item() # Index from 0 to n_images-1
             # Calculate the climbing force: F_climb = F_true - 2 * (F_true . tau) * tau
             # This effectively inverts the component of the true force parallel to the tangent
             F_climb = true_forces_reshaped[climbing_image_idx] - (
                 2
                 * F_true_dot_tau[climbing_image_idx]
                 * tangents[climbing_image_idx]
             )
             # Replace the NEB force for the climbing image with F_climb
             # This overwrites the spring force component for this image, as required.
             neb_forces[climbing_image_idx] = F_climb

        # --- Logging (Optional) ---
        logger.debug(
            "  Max True Force Mag: "
            f"{torch.linalg.norm(true_forces_reshaped, dim=(-1,-2)).max().item():.4f}"
        )
        logger.debug(
            "  Max F_perp Mag: "
            f"{torch.linalg.norm(F_perp, dim=(-1,-2)).max().item():.4f}"
        )
        logger.debug(
            "  Max F_spring_par Mag: "
            f"{torch.linalg.norm(F_spring_par, dim=(-1,-2)).max().item():.4f}"
        )
        logger.debug(
            "  Max NEB Force Mag: "
            f"{torch.linalg.norm(neb_forces, dim=(-1,-2)).max().item():.4f}"
        )


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
        """Run the Nudged Elastic Band optimization.

        Optimizes the path between the initial and final systems to find the
        Minimum Energy Path (MEP).

        Args:
            initial_system (StateLike): The starting configuration (can be ASE Atoms,
                SimState, or other compatible format recognized by initialize_state).
            final_system (StateLike): The ending configuration.
            max_steps (int): Maximum number of optimization steps allowed.
            fmax (float): Convergence criterion based on the maximum NEB force component
                acting on any single atom across all intermediate images (in eV/Ang).

        Returns:
            SimState: The final optimized NEB path, including the initial,
                intermediate, and final images, concatenated into a single SimState.
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
        logger.info(
            f"Initial Energy: {initial_energy:.4f}, Final Energy: {final_energy:.4f}"
        )

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
                    full_path_state_calc,
                    true_forces,
                    true_energies,
                    initial_energy,
                    final_energy,
                    step=step,
                )

                # c. Update the forces in the FIRE state object with NEB forces
                opt_state.forces = neb_forces

                # d. Perform FIRE optimization step
                # Use the generic step function
                opt_state = self._step_fn(opt_state)
                logger.debug(
                    "  Max True Force Mag (after step): "
                    f"{torch.sqrt((opt_state.forces**2).sum(dim=-1)).max().item():.4f}"
                )

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
        return concatenate_states(
            [initial_state, opt_state, final_state]
        )
