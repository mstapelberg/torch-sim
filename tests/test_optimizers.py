import copy
from typing import get_args

import pytest
import torch

from torch_sim.optimizers import (
    MdFlavor,
    fire,
    frechet_cell_fire,
    gradient_descent,
    unit_cell_fire,
    unit_cell_gradient_descent,
)
from torch_sim.state import SimState, concatenate_states


def test_gradient_descent_optimization(
    ar_supercell_sim_state: SimState, lj_model: torch.nn.Module
) -> None:
    """Test that the Gradient Descent optimizer actually minimizes energy."""
    # Add some random displacement to positions
    perturbed_positions = (
        ar_supercell_sim_state.positions
        + torch.randn_like(ar_supercell_sim_state.positions) * 0.1
    )

    ar_supercell_sim_state.positions = perturbed_positions
    initial_state = ar_supercell_sim_state

    # Initialize Gradient Descent optimizer
    init_fn, update_fn = gradient_descent(
        model=lj_model,
        lr=0.01,
    )

    state = init_fn(ar_supercell_sim_state)

    # Run optimization for a few steps
    energies = [1000, state.energy.item()]
    while abs(energies[-2] - energies[-1]) > 1e-6:
        state = update_fn(state)
        energies.append(state.energy.item())

    energies = energies[1:]

    # Check that energy decreased
    assert energies[-1] < energies[0], (
        f"FIRE optimization should reduce energy "
        f"(initial: {energies[0]}, final: {energies[-1]})"
    )

    # Check force convergence
    max_force = torch.max(torch.norm(state.forces, dim=1))
    assert max_force < 0.2, f"Forces should be small after optimization, got {max_force=}"

    assert not torch.allclose(state.positions, initial_state.positions)


def test_unit_cell_gradient_descent_optimization(
    ar_supercell_sim_state: SimState, lj_model: torch.nn.Module
) -> None:
    """Test that the Gradient Descent optimizer actually minimizes energy."""
    # Add some random displacement to positions
    perturbed_positions = (
        ar_supercell_sim_state.positions
        + torch.randn_like(ar_supercell_sim_state.positions) * 0.1
    )

    ar_supercell_sim_state.positions = perturbed_positions
    initial_state = ar_supercell_sim_state

    # Initialize Gradient Descent optimizer
    init_fn, update_fn = unit_cell_gradient_descent(
        model=lj_model,
        positions_lr=0.01,
        cell_lr=0.1,
    )

    state = init_fn(ar_supercell_sim_state)

    # Run optimization for a few steps
    energies = [1000, state.energy.item()]
    while abs(energies[-2] - energies[-1]) > 1e-6:
        state = update_fn(state)
        energies.append(state.energy.item())

    energies = energies[1:]

    # Check that energy decreased
    assert energies[-1] < energies[0], (
        f"Gradient Descent optimization should reduce energy "
        f"(initial: {energies[0]}, final: {energies[-1]})"
    )

    # Check force convergence
    max_force = torch.max(torch.norm(state.forces, dim=1))
    pressure = torch.trace(state.stress.squeeze(0)) / 3.0
    assert pressure < 0.01, (
        f"Pressure should be small after optimization, got {pressure=}"
    )
    assert max_force < 0.2, f"Forces should be small after optimization, got {max_force=}"

    assert not torch.allclose(state.positions, initial_state.positions)
    assert not torch.allclose(state.cell, initial_state.cell)


@pytest.mark.parametrize("md_flavor", get_args(MdFlavor))
def test_fire_optimization(
    ar_supercell_sim_state: SimState, lj_model: torch.nn.Module, md_flavor: MdFlavor
) -> None:
    """Test that the FIRE optimizer actually minimizes energy."""
    # Add some random displacement to positions
    # Create a fresh copy for each test run to avoid interference

    current_positions = (
        ar_supercell_sim_state.positions.clone()
        + torch.randn_like(ar_supercell_sim_state.positions) * 0.1
    )

    current_sim_state = SimState(
        positions=current_positions,
        masses=ar_supercell_sim_state.masses.clone(),
        cell=ar_supercell_sim_state.cell.clone(),
        pbc=ar_supercell_sim_state.pbc,
        atomic_numbers=ar_supercell_sim_state.atomic_numbers.clone(),
        batch=ar_supercell_sim_state.batch.clone(),
    )

    initial_state_positions = current_sim_state.positions.clone()

    # Initialize FIRE optimizer
    init_fn, update_fn = fire(
        model=lj_model,
        dt_max=0.3,
        dt_start=0.1,
        md_flavor=md_flavor,
    )

    state = init_fn(current_sim_state)

    # Run optimization for a few steps
    energies = [1000, state.energy.item()]
    max_steps = 1000  # Add max step to prevent infinite loop
    steps_taken = 0
    while abs(energies[-2] - energies[-1]) > 1e-6 and steps_taken < max_steps:
        state = update_fn(state)
        energies.append(state.energy.item())
        steps_taken += 1

    if steps_taken == max_steps:
        print(f"FIRE optimization for {md_flavor=} did not converge in {max_steps} steps")

    energies = energies[1:]

    # Check that energy decreased
    assert energies[-1] < energies[0], (
        f"FIRE optimization for {md_flavor=} should reduce energy "
        f"(initial: {energies[0]}, final: {energies[-1]})"
    )

    # Check force convergence
    max_force = torch.max(torch.norm(state.forces, dim=1))
    # bumped up the tolerance to 0.3 to account for the fact that ase_fire is more lenient
    # in beginning steps
    assert max_force < 0.3, (
        f"{md_flavor=} forces should be small after optimization, got {max_force=}"
    )

    assert not torch.allclose(state.positions, initial_state_positions), (
        f"{md_flavor=} positions should have changed after optimization."
    )


@pytest.mark.parametrize("md_flavor", get_args(MdFlavor))
def test_unit_cell_fire_optimization(
    ar_supercell_sim_state: SimState, lj_model: torch.nn.Module, md_flavor: MdFlavor
) -> None:
    """Test that the FIRE optimizer actually minimizes energy."""
    print(f"\n--- Starting test_unit_cell_fire_optimization for {md_flavor=} ---")

    # Add random displacement to positions and cell
    current_positions = (
        ar_supercell_sim_state.positions.clone()
        + torch.randn_like(ar_supercell_sim_state.positions) * 0.1
    )
    current_cell = (
        ar_supercell_sim_state.cell.clone()
        + torch.randn_like(ar_supercell_sim_state.cell) * 0.01
    )  # Reduced cell perturbation slightly

    current_sim_state = SimState(
        positions=current_positions,
        masses=ar_supercell_sim_state.masses.clone(),
        cell=current_cell,
        pbc=ar_supercell_sim_state.pbc,
        atomic_numbers=ar_supercell_sim_state.atomic_numbers.clone(),
        batch=ar_supercell_sim_state.batch.clone(),
    )
    print(f"[{md_flavor}] Initial SimState created.")

    initial_state_positions = current_sim_state.positions.clone()
    initial_state_cell = current_sim_state.cell.clone()

    # Initialize FIRE optimizer
    print(f"[{md_flavor}] Initializing {md_flavor} optimizer...")
    init_fn, update_fn = unit_cell_fire(
        model=lj_model,
        dt_max=0.3,
        dt_start=0.1,
        md_flavor=md_flavor,
        # Add maxstep for ase_fire if not already default in optimizer
        # maxstep=0.2 # Assuming it's handled by the optimizer function
    )
    print(f"[{md_flavor}] Optimizer functions obtained.")

    state = init_fn(current_sim_state)
    energy = float(getattr(state, "energy", "nan"))
    print(f"[{md_flavor}] Initial state created by init_fn. {energy=:.4f}")

    # Run optimization for a few steps
    energies = [1000.0, state.energy.item()]  # Ensure float for comparison
    max_steps = (
        1000  # MODIFIED: Drastically reduced for initial debugging of ase_fire hanging
    )
    steps_taken = 0

    while abs(energies[-2] - energies[-1]) > 1e-6 and steps_taken < max_steps:
        state = update_fn(state)
        energies.append(state.energy.item())
        steps_taken += 1

    print(f"[{md_flavor}] Loop finished after {steps_taken} steps.")

    if (
        steps_taken == max_steps and abs(energies[-2] - energies[-1]) > 1e-6
    ):  # MODIFIED: Check if max_steps was hit AND not converged
        print(
            f"WARNING: Unit Cell FIRE {md_flavor=} optimization did not converge "
            f"in {max_steps} steps. Final energy: {energies[-1]:.4f}"
        )
    else:
        print(
            f"Unit Cell FIRE {md_flavor=} optimization converged in {steps_taken} "
            f"steps. Final energy: {energies[-1]:.4f}"
        )

    energies = energies[1:]

    # Check that energy decreased
    assert energies[-1] < energies[0], (
        f"Unit Cell FIRE optimization for {md_flavor=} should reduce energy "
        f"(initial: {energies[0]}, final: {energies[-1]})"
    )

    # Check force convergence
    max_force = torch.max(torch.norm(state.forces, dim=1))
    pressure = torch.trace(state.stress.squeeze(0)) / 3.0
    assert pressure < 0.01, (
        f"Pressure should be small after optimization, got {pressure=}"
    )
    assert max_force < 0.3, (
        f"{md_flavor=} forces should be small after optimization, got {max_force=}"
    )

    assert not torch.allclose(state.positions, initial_state_positions), (
        f"{md_flavor=} positions should have changed after optimization."
    )
    assert not torch.allclose(state.cell, initial_state_cell), (
        f"{md_flavor=} cell should have changed after optimization."
    )


@pytest.mark.parametrize("md_flavor", get_args(MdFlavor))
def test_frechet_cell_fire_optimization(
    ar_supercell_sim_state: SimState, lj_model: torch.nn.Module, md_flavor: MdFlavor
) -> None:
    """Test that the Frechet Cell FIRE optimizer actually minimizes energy for different
    md_flavors."""
    print(f"\n--- Starting test_frechet_cell_fire_optimization for {md_flavor=} ---")

    # Add random displacement to positions and cell
    # Create a fresh copy for each test run to avoid interference
    current_positions = (
        ar_supercell_sim_state.positions.clone()
        + torch.randn_like(ar_supercell_sim_state.positions) * 0.1
    )
    current_cell = (
        ar_supercell_sim_state.cell.clone()
        + torch.randn_like(ar_supercell_sim_state.cell) * 0.01
    )

    current_sim_state = SimState(
        positions=current_positions,
        masses=ar_supercell_sim_state.masses.clone(),
        cell=current_cell,
        pbc=ar_supercell_sim_state.pbc,
        atomic_numbers=ar_supercell_sim_state.atomic_numbers.clone(),
        batch=ar_supercell_sim_state.batch.clone(),
    )
    print(f"[{md_flavor}] Initial SimState created for Frechet test.")

    initial_state_positions = current_sim_state.positions.clone()
    initial_state_cell = current_sim_state.cell.clone()

    # Initialize FIRE optimizer
    print(f"[{md_flavor}] Initializing Frechet {md_flavor} optimizer...")
    init_fn, update_fn = frechet_cell_fire(
        model=lj_model,
        dt_max=0.3,
        dt_start=0.1,
        md_flavor=md_flavor,
    )
    print(f"[{md_flavor}] Frechet optimizer functions obtained.")

    state = init_fn(current_sim_state)
    energy = float(getattr(state, "energy", "nan"))
    print(f"[{md_flavor}] Initial state created by Frechet init_fn. {energy=:.4f}")

    # Run optimization for a few steps
    energies = [1000.0, state.energy.item()]  # Ensure float for comparison
    max_steps = 1000
    steps_taken = 0
    print(f"[{md_flavor}] Entering Frechet optimization loop (max_steps: {max_steps})...")

    while abs(energies[-2] - energies[-1]) > 1e-6 and steps_taken < max_steps:
        state = update_fn(state)
        energies.append(state.energy.item())
        steps_taken += 1

    print(f"[{md_flavor}] Frechet loop finished after {steps_taken} steps.")

    if steps_taken == max_steps and abs(energies[-2] - energies[-1]) > 1e-6:
        print(
            f"WARNING: Frechet Cell FIRE {md_flavor=} optimization did not converge "
            f"in {max_steps} steps. Final energy: {energies[-1]:.4f}"
        )
    else:
        print(
            f"Frechet Cell FIRE {md_flavor=} optimization converged in {steps_taken} "
            f"steps. Final energy: {energies[-1]:.4f}"
        )

    energies = energies[1:]

    # Check that energy decreased
    assert energies[-1] < energies[0], (
        f"Frechet FIRE {md_flavor=} optimization should reduce energy "
        f"(initial: {energies[0]}, final: {energies[-1]})"
    )

    # Check force convergence
    max_force = torch.max(torch.norm(state.forces, dim=1))
    pressure = (
        torch.trace(state.stress.squeeze(0)) / 3.0
    )  # Assumes single batch for this state stress access

    # Adjust tolerances if needed, Frechet might behave slightly differently
    pressure_tolerance = 0.01
    force_tolerance = 0.2

    assert torch.abs(pressure) < pressure_tolerance, (
        f"{md_flavor=} pressure should be small after Frechet optimization, "
        f"got {pressure.item()}"
    )
    assert max_force < force_tolerance, (
        f"{md_flavor=} forces should be small after Frechet optimization, got {max_force}"
    )

    assert not torch.allclose(state.positions, initial_state_positions, atol=1e-5), (
        f"{md_flavor=} positions should have changed after Frechet optimization."
    )
    assert not torch.allclose(state.cell, initial_state_cell, atol=1e-5), (
        f"{md_flavor=} cell should have changed after Frechet optimization."
    )


def test_fire_multi_batch(
    ar_supercell_sim_state: SimState, lj_model: torch.nn.Module
) -> None:
    """Test FIRE optimization with multiple batches."""
    # Create a multi-batch system by duplicating ar_fcc_state

    generator = torch.Generator(device=ar_supercell_sim_state.device)

    ar_supercell_sim_state_1 = copy.deepcopy(ar_supercell_sim_state)
    ar_supercell_sim_state_2 = copy.deepcopy(ar_supercell_sim_state)

    for state in [ar_supercell_sim_state_1, ar_supercell_sim_state_2]:
        generator.manual_seed(43)
        state.positions += (
            torch.randn(
                state.positions.shape,
                device=state.device,
                generator=generator,
            )
            * 0.1
        )

    multi_state = concatenate_states(
        [ar_supercell_sim_state_1, ar_supercell_sim_state_2],
        device=ar_supercell_sim_state.device,
    )

    # Initialize FIRE optimizer
    init_fn, update_fn = fire(
        model=lj_model,
        dt_max=0.3,
        dt_start=0.1,
    )

    state = init_fn(multi_state)
    initial_state = copy.deepcopy(state)

    # Run optimization for a few steps
    prev_energy = torch.ones(2, device=state.device, dtype=state.energy.dtype) * 1000
    current_energy = initial_state.energy
    step = 0
    while not torch.allclose(current_energy, prev_energy, atol=1e-9):
        prev_energy = current_energy
        state = update_fn(state)
        current_energy = state.energy

        step += 1
        if step > 500:
            raise ValueError("Optimization did not converge")

    # check that we actually optimized
    assert step > 10

    # Check that energy decreased for both batches
    assert torch.all(state.energy < initial_state.energy), (
        "FIRE optimization should reduce energy for all batches"
    )

    # transfer the energy and force checks to the batched optimizer
    max_force = torch.max(torch.norm(state.forces, dim=1))
    assert torch.all(max_force < 0.1), (
        f"Forces should be small after optimization, got {max_force=}"
    )

    n_ar_atoms = ar_supercell_sim_state.n_atoms
    assert not torch.allclose(
        state.positions[:n_ar_atoms], multi_state.positions[:n_ar_atoms]
    )
    assert not torch.allclose(
        state.positions[n_ar_atoms:], multi_state.positions[n_ar_atoms:]
    )

    # we are evolving identical systems
    assert current_energy[0] == current_energy[1]


def test_fire_batch_consistency(
    ar_supercell_sim_state: SimState, lj_model: torch.nn.Module
) -> None:
    """Test batched FIRE optimization is consistent with individual optimizations."""
    generator = torch.Generator(device=ar_supercell_sim_state.device)

    ar_supercell_sim_state_1 = copy.deepcopy(ar_supercell_sim_state)
    ar_supercell_sim_state_2 = copy.deepcopy(ar_supercell_sim_state)

    # Add same random perturbation to both states
    for state in [ar_supercell_sim_state_1, ar_supercell_sim_state_2]:
        generator.manual_seed(43)
        state.positions += (
            torch.randn(
                state.positions.shape,
                device=state.device,
                generator=generator,
            )
            * 0.1
        )

    # Optimize each state individually
    final_individual_states = []
    total_steps = []

    def energy_converged(current_energy: float, prev_energy: float) -> bool:
        """Check if optimization should continue based on energy convergence."""
        return not torch.allclose(current_energy, prev_energy, atol=1e-6)

    for state in [ar_supercell_sim_state_1, ar_supercell_sim_state_2]:
        init_fn, update_fn = fire(
            model=lj_model,
            dt_max=0.3,
            dt_start=0.1,
        )

        state_opt = init_fn(state)

        # Run optimization until convergence
        current_energy = state_opt.energy
        prev_energy = current_energy + 1

        step = 0
        while energy_converged(current_energy, prev_energy):
            prev_energy = current_energy
            state_opt = update_fn(state_opt)
            current_energy = state_opt.energy
            step += 1
            if step > 1000:
                raise ValueError("Optimization did not converge")

        final_individual_states.append(state_opt)
        total_steps.append(step)

    # Now optimize both states together in a batch
    multi_state = concatenate_states(
        [
            copy.deepcopy(ar_supercell_sim_state_1),
            copy.deepcopy(ar_supercell_sim_state_2),
        ],
        device=ar_supercell_sim_state.device,
    )

    init_fn, batch_update_fn = fire(
        model=lj_model,
        dt_max=0.3,
        dt_start=0.1,
    )

    batch_state = init_fn(multi_state)

    # Run optimization until convergence for both batches
    current_energies = batch_state.energy.clone()
    prev_energies = current_energies + 1

    step = 0
    while energy_converged(current_energies[0], prev_energies[0]) and energy_converged(
        current_energies[1], prev_energies[1]
    ):
        prev_energies = current_energies.clone()
        batch_state = batch_update_fn(batch_state)
        current_energies = batch_state.energy.clone()
        step += 1
        if step > 1000:
            raise ValueError("Optimization did not converge")

    individual_energies = [state.energy.item() for state in final_individual_states]
    # Check that final energies from batched optimization match individual optimizations
    for step, individual_energy in enumerate(individual_energies):
        assert abs(batch_state.energy[step].item() - individual_energy) < 1e-4, (
            f"Energy for batch {step} doesn't match individual optimization: "
            f"batch={batch_state.energy[step].item()}, individual={individual_energy}"
        )


def test_unit_cell_fire_multi_batch(
    ar_supercell_sim_state: SimState, lj_model: torch.nn.Module
) -> None:
    """Test FIRE optimization with multiple batches."""
    # Create a multi-batch system by duplicating ar_fcc_state

    generator = torch.Generator(device=ar_supercell_sim_state.device)

    ar_supercell_sim_state_1 = copy.deepcopy(ar_supercell_sim_state)
    ar_supercell_sim_state_2 = copy.deepcopy(ar_supercell_sim_state)

    for state in [ar_supercell_sim_state_1, ar_supercell_sim_state_2]:
        generator.manual_seed(43)
        state.positions += (
            torch.randn(
                state.positions.shape,
                device=state.device,
                generator=generator,
            )
            * 0.1
        )

    multi_state = concatenate_states(
        [ar_supercell_sim_state_1, ar_supercell_sim_state_2],
        device=ar_supercell_sim_state.device,
    )

    # Initialize FIRE optimizer
    init_fn, update_fn = unit_cell_fire(
        model=lj_model,
        dt_max=0.3,
        dt_start=0.1,
    )

    state = init_fn(multi_state)
    initial_state = copy.deepcopy(state)

    # Run optimization for a few steps
    prev_energy = torch.ones(2, device=state.device, dtype=state.energy.dtype) * 1000
    current_energy = initial_state.energy
    step = 0
    while not torch.allclose(current_energy, prev_energy, atol=1e-9):
        prev_energy = current_energy
        state = update_fn(state)
        current_energy = state.energy

        step += 1
        if step > 500:
            raise ValueError("Optimization did not converge")

    # check that we actually optimized
    assert step > 10

    # Check that energy decreased for both batches
    assert torch.all(state.energy < initial_state.energy), (
        "FIRE optimization should reduce energy for all batches"
    )

    # transfer the energy and force checks to the batched optimizer
    max_force = torch.max(torch.norm(state.forces, dim=1))
    assert torch.all(max_force < 0.1), (
        f"Forces should be small after optimization, got {max_force=}"
    )

    pressure_0 = torch.trace(state.stress[0]) / 3.0
    pressure_1 = torch.trace(state.stress[1]) / 3.0
    assert torch.allclose(pressure_0, pressure_1), (
        f"Pressure should be the same for all batches, got {pressure_0=}, {pressure_1=}"
    )
    assert pressure_0 < 0.01, (
        f"Pressure should be small after optimization, got {pressure_0=}"
    )
    assert pressure_1 < 0.01, (
        f"Pressure should be small after optimization, got {pressure_1=}"
    )

    n_ar_atoms = ar_supercell_sim_state.n_atoms
    assert not torch.allclose(
        state.positions[:n_ar_atoms], multi_state.positions[:n_ar_atoms]
    )
    assert not torch.allclose(
        state.positions[n_ar_atoms:], multi_state.positions[n_ar_atoms:]
    )
    assert not torch.allclose(state.cell, multi_state.cell)

    # we are evolving identical systems
    assert current_energy[0] == current_energy[1]


def test_unit_cell_fire_batch_consistency(
    ar_supercell_sim_state: SimState, lj_model: torch.nn.Module
) -> None:
    """Test batched FIRE optimization is consistent with individual optimizations."""
    generator = torch.Generator(device=ar_supercell_sim_state.device)

    ar_supercell_sim_state_1 = copy.deepcopy(ar_supercell_sim_state)
    ar_supercell_sim_state_2 = copy.deepcopy(ar_supercell_sim_state)

    # Add same random perturbation to both states
    for state in [ar_supercell_sim_state_1, ar_supercell_sim_state_2]:
        generator.manual_seed(43)
        state.positions += (
            torch.randn(
                state.positions.shape,
                device=state.device,
                generator=generator,
            )
            * 0.1
        )

    # Optimize each state individually
    final_individual_states = []
    total_steps = []

    def energy_converged(current_energy: float, prev_energy: float) -> bool:
        """Check if optimization should continue based on energy convergence."""
        return not torch.allclose(current_energy, prev_energy, atol=1e-6)

    for state in [ar_supercell_sim_state_1, ar_supercell_sim_state_2]:
        init_fn, update_fn = unit_cell_fire(
            model=lj_model,
            dt_max=0.3,
            dt_start=0.1,
        )

        state_opt = init_fn(state)

        # Run optimization until convergence
        current_energy = state_opt.energy
        prev_energy = current_energy + 1

        step = 0
        while energy_converged(current_energy, prev_energy):
            prev_energy = current_energy
            state_opt = update_fn(state_opt)
            current_energy = state_opt.energy
            step += 1
            if step > 1000:
                raise ValueError("Optimization did not converge")

        final_individual_states.append(state_opt)
        total_steps.append(step)

    # Now optimize both states together in a batch
    multi_state = concatenate_states(
        [
            copy.deepcopy(ar_supercell_sim_state_1),
            copy.deepcopy(ar_supercell_sim_state_2),
        ],
        device=ar_supercell_sim_state.device,
    )

    init_fn, batch_update_fn = unit_cell_fire(
        model=lj_model,
        dt_max=0.3,
        dt_start=0.1,
    )

    batch_state = init_fn(multi_state)

    # Run optimization until convergence for both batches
    current_energies = batch_state.energy.clone()
    prev_energies = current_energies + 1

    step = 0
    while energy_converged(current_energies[0], prev_energies[0]) and energy_converged(
        current_energies[1], prev_energies[1]
    ):
        prev_energies = current_energies.clone()
        batch_state = batch_update_fn(batch_state)
        current_energies = batch_state.energy.clone()
        step += 1
        if step > 1000:
            raise ValueError("Optimization did not converge")

    individual_energies = [state.energy.item() for state in final_individual_states]
    # Check that final energies from batched optimization match individual optimizations
    for step, individual_energy in enumerate(individual_energies):
        assert abs(batch_state.energy[step].item() - individual_energy) < 1e-4, (
            f"Energy for batch {step} doesn't match individual optimization: "
            f"batch={batch_state.energy[step].item()}, individual={individual_energy}"
        )


def test_unit_cell_frechet_fire_multi_batch(
    ar_supercell_sim_state: SimState, lj_model: torch.nn.Module
) -> None:
    """Test FIRE optimization with multiple batches."""
    # Create a multi-batch system by duplicating ar_fcc_state

    generator = torch.Generator(device=ar_supercell_sim_state.device)

    ar_supercell_sim_state_1 = copy.deepcopy(ar_supercell_sim_state)
    ar_supercell_sim_state_2 = copy.deepcopy(ar_supercell_sim_state)

    for state in [ar_supercell_sim_state_1, ar_supercell_sim_state_2]:
        generator.manual_seed(43)
        state.positions += (
            torch.randn(
                state.positions.shape,
                device=state.device,
                generator=generator,
            )
            * 0.1
        )

    multi_state = concatenate_states(
        [ar_supercell_sim_state_1, ar_supercell_sim_state_2],
        device=ar_supercell_sim_state.device,
    )

    # Initialize FIRE optimizer
    init_fn, update_fn = frechet_cell_fire(
        model=lj_model,
        dt_max=0.3,
        dt_start=0.1,
    )

    state = init_fn(multi_state)
    initial_state = copy.deepcopy(state)

    # Run optimization for a few steps
    prev_energy = torch.ones(2, device=state.device, dtype=state.energy.dtype) * 1000
    current_energy = initial_state.energy
    step = 0
    while not torch.allclose(current_energy, prev_energy, atol=1e-9):
        prev_energy = current_energy
        state = update_fn(state)
        current_energy = state.energy

        step += 1
        if step > 500:
            raise ValueError("Optimization did not converge")

    # check that we actually optimized
    assert step > 10

    # Check that energy decreased for both batches
    assert torch.all(state.energy < initial_state.energy), (
        "FIRE optimization should reduce energy for all batches"
    )

    # transfer the energy and force checks to the batched optimizer
    max_force = torch.max(torch.norm(state.forces, dim=1))
    assert torch.all(max_force < 0.1), (
        f"Forces should be small after optimization, got {max_force=}"
    )

    pressure_0 = torch.trace(state.stress[0]) / 3.0
    pressure_1 = torch.trace(state.stress[1]) / 3.0
    assert torch.allclose(pressure_0, pressure_1), (
        f"Pressure should be the same for all batches, got {pressure_0=}, {pressure_1=}"
    )
    assert pressure_0 < 0.01, (
        f"Pressure should be small after optimization, got {pressure_0=}"
    )
    assert pressure_1 < 0.01, (
        f"Pressure should be small after optimization, got {pressure_1=}"
    )

    n_ar_atoms = ar_supercell_sim_state.n_atoms
    assert not torch.allclose(
        state.positions[:n_ar_atoms], multi_state.positions[:n_ar_atoms]
    )
    assert not torch.allclose(
        state.positions[n_ar_atoms:], multi_state.positions[n_ar_atoms:]
    )
    assert not torch.allclose(state.cell, multi_state.cell)

    # we are evolving identical systems
    assert current_energy[0] == current_energy[1]


def test_unit_cell_frechet_fire_batch_consistency(
    ar_supercell_sim_state: SimState, lj_model: torch.nn.Module
) -> None:
    """Test batched FIRE optimization is consistent with individual optimizations."""
    generator = torch.Generator(device=ar_supercell_sim_state.device)

    ar_supercell_sim_state_1 = copy.deepcopy(ar_supercell_sim_state)
    ar_supercell_sim_state_2 = copy.deepcopy(ar_supercell_sim_state)

    # Add same random perturbation to both states
    for state in [ar_supercell_sim_state_1, ar_supercell_sim_state_2]:
        generator.manual_seed(43)
        state.positions += (
            torch.randn(
                state.positions.shape,
                device=state.device,
                generator=generator,
            )
            * 0.1
        )

    # Optimize each state individually
    final_individual_states = []
    total_steps = []

    def energy_converged(current_energy: float, prev_energy: float) -> bool:
        """Check if optimization should continue based on energy convergence."""
        return not torch.allclose(current_energy, prev_energy, atol=1e-6)

    for state in [ar_supercell_sim_state_1, ar_supercell_sim_state_2]:
        init_fn, update_fn = frechet_cell_fire(
            model=lj_model,
            dt_max=0.3,
            dt_start=0.1,
        )

        state_opt = init_fn(state)

        # Run optimization until convergence
        current_energy = state_opt.energy
        prev_energy = current_energy + 1

        step = 0
        while energy_converged(current_energy, prev_energy):
            prev_energy = current_energy
            state_opt = update_fn(state_opt)
            current_energy = state_opt.energy
            step += 1
            if step > 1000:
                raise ValueError("Optimization did not converge")

        final_individual_states.append(state_opt)
        total_steps.append(step)

    # Now optimize both states together in a batch
    multi_state = concatenate_states(
        [
            copy.deepcopy(ar_supercell_sim_state_1),
            copy.deepcopy(ar_supercell_sim_state_2),
        ],
        device=ar_supercell_sim_state.device,
    )

    init_fn, batch_update_fn = frechet_cell_fire(
        model=lj_model,
        dt_max=0.3,
        dt_start=0.1,
    )

    batch_state = init_fn(multi_state)

    # Run optimization until convergence for both batches
    current_energies = batch_state.energy.clone()
    prev_energies = current_energies + 1

    step = 0
    while energy_converged(current_energies[0], prev_energies[0]) and energy_converged(
        current_energies[1], prev_energies[1]
    ):
        prev_energies = current_energies.clone()
        batch_state = batch_update_fn(batch_state)
        current_energies = batch_state.energy.clone()
        step += 1
        if step > 1000:
            raise ValueError("Optimization did not converge")

    individual_energies = [state.energy.item() for state in final_individual_states]
    # Check that final energies from batched optimization match individual optimizations
    for step, individual_energy in enumerate(individual_energies):
        assert abs(batch_state.energy[step].item() - individual_energy) < 1e-4, (
            f"Energy for batch {step} doesn't match individual optimization: "
            f"batch={batch_state.energy[step].item()}, individual={individual_energy}"
        )


def test_fire_fixed_cell_frechet_consistency(  # noqa: C901
    ar_supercell_sim_state: SimState, lj_model: torch.nn.Module
) -> None:
    """Test batched Frechet Fixed cell FIRE optimization is
    consistent with FIRE (position only) optimizations."""
    generator = torch.Generator(device=ar_supercell_sim_state.device)

    ar_supercell_sim_state_1 = copy.deepcopy(ar_supercell_sim_state)
    ar_supercell_sim_state_2 = copy.deepcopy(ar_supercell_sim_state)

    # Add same random perturbation to both states
    for state in [ar_supercell_sim_state_1, ar_supercell_sim_state_2]:
        generator.manual_seed(43)
        state.positions += (
            torch.randn(
                state.positions.shape,
                device=state.device,
                generator=generator,
            )
            * 0.1
        )

    # Optimize each state individually
    final_individual_states_frechet = []
    total_steps_frechet = []

    def energy_converged(current_energy: float, prev_energy: float) -> bool:
        """Check if optimization should continue based on energy convergence."""
        return not torch.allclose(current_energy, prev_energy, atol=1e-6)

    for state in [ar_supercell_sim_state_1, ar_supercell_sim_state_2]:
        init_fn, update_fn = unit_cell_fire(
            model=lj_model,
            dt_max=0.3,
            dt_start=0.1,
            hydrostatic_strain=True,
            constant_volume=True,
        )

        state_opt = init_fn(state)

        # Run optimization until convergence
        current_energy = state_opt.energy
        prev_energy = current_energy + 1

        step = 0
        while energy_converged(current_energy, prev_energy):
            prev_energy = current_energy
            state_opt = update_fn(state_opt)
            current_energy = state_opt.energy
            step += 1
            if step > 1000:
                raise ValueError("Optimization did not converge")

        final_individual_states_frechet.append(state_opt)
        total_steps_frechet.append(step)

    # Optimize each state individually
    final_individual_states_fire = []
    total_steps_fire = []

    def energy_converged(current_energy: float, prev_energy: float) -> bool:
        """Check if optimization should continue based on energy convergence."""
        return not torch.allclose(current_energy, prev_energy, atol=1e-6)

    for state in [ar_supercell_sim_state_1, ar_supercell_sim_state_2]:
        init_fn, update_fn = fire(
            model=lj_model,
            dt_max=0.3,
            dt_start=0.1,
        )

        state_opt = init_fn(state)

        # Run optimization until convergence
        current_energy = state_opt.energy
        prev_energy = current_energy + 1

        step = 0
        while energy_converged(current_energy, prev_energy):
            prev_energy = current_energy
            state_opt = update_fn(state_opt)
            current_energy = state_opt.energy
            step += 1
            if step > 1000:
                raise ValueError("Optimization did not converge")

        final_individual_states_fire.append(state_opt)
        total_steps_fire.append(step)

    individual_energies_frechet = [
        state.energy.item() for state in final_individual_states_frechet
    ]
    individual_energies_fire = [
        state.energy.item() for state in final_individual_states_fire
    ]
    # Check that final energies from fixed cell optimization match
    # position only optimizations
    for step, energy_frechet in enumerate(individual_energies_frechet):
        assert abs(energy_frechet - individual_energies_fire[step]) < 1e-4, (
            f"Energy for batch {step} doesn't match position only optimization: "
            f"batch={energy_frechet}, individual={individual_energies_fire[step]}"
        )


def test_fire_fixed_cell_unit_cell_consistency(  # noqa: C901
    ar_supercell_sim_state: SimState, lj_model: torch.nn.Module
) -> None:
    """Test batched Frechet Fixed cell FIRE optimization is
    consistent with FIRE (position only) optimizations."""
    generator = torch.Generator(device=ar_supercell_sim_state.device)

    ar_supercell_sim_state_1 = copy.deepcopy(ar_supercell_sim_state)
    ar_supercell_sim_state_2 = copy.deepcopy(ar_supercell_sim_state)

    # Add same random perturbation to both states
    for state in [ar_supercell_sim_state_1, ar_supercell_sim_state_2]:
        generator.manual_seed(43)
        state.positions += (
            torch.randn(
                state.positions.shape,
                device=state.device,
                generator=generator,
            )
            * 0.1
        )

    # Optimize each state individually
    final_individual_states_unit_cell = []
    total_steps_unit_cell = []

    def energy_converged(current_energy: float, prev_energy: float) -> bool:
        """Check if optimization should continue based on energy convergence."""
        return not torch.allclose(current_energy, prev_energy, atol=1e-6)

    for state in [ar_supercell_sim_state_1, ar_supercell_sim_state_2]:
        init_fn, update_fn = unit_cell_fire(
            model=lj_model,
            dt_max=0.3,
            dt_start=0.1,
            hydrostatic_strain=True,
            constant_volume=True,
        )

        state_opt = init_fn(state)

        # Run optimization until convergence
        current_energy = state_opt.energy
        prev_energy = current_energy + 1

        step = 0
        while energy_converged(current_energy, prev_energy):
            prev_energy = current_energy
            state_opt = update_fn(state_opt)
            current_energy = state_opt.energy
            step += 1
            if step > 1000:
                raise ValueError("Optimization did not converge")

        final_individual_states_unit_cell.append(state_opt)
        total_steps_unit_cell.append(step)

    # Optimize each state individually
    final_individual_states_fire = []
    total_steps_fire = []

    def energy_converged(current_energy: float, prev_energy: float) -> bool:
        """Check if optimization should continue based on energy convergence."""
        return not torch.allclose(current_energy, prev_energy, atol=1e-6)

    for state in [ar_supercell_sim_state_1, ar_supercell_sim_state_2]:
        init_fn, update_fn = fire(
            model=lj_model,
            dt_max=0.3,
            dt_start=0.1,
        )

        state_opt = init_fn(state)

        # Run optimization until convergence
        current_energy = state_opt.energy
        prev_energy = current_energy + 1

        step = 0
        while energy_converged(current_energy, prev_energy):
            prev_energy = current_energy
            state_opt = update_fn(state_opt)
            current_energy = state_opt.energy
            step += 1
            if step > 1000:
                raise ValueError("Optimization did not converge")

        final_individual_states_fire.append(state_opt)
        total_steps_fire.append(step)

    individual_energies_unit_cell = [
        state.energy.item() for state in final_individual_states_unit_cell
    ]
    individual_energies_fire = [
        state.energy.item() for state in final_individual_states_fire
    ]
    # Check that final energies from fixed cell optimization match
    # position only optimizations
    for step, energy_unit_cell in enumerate(individual_energies_unit_cell):
        assert abs(energy_unit_cell - individual_energies_fire[step]) < 1e-4, (
            f"Energy for batch {step} doesn't match position only optimization: "
            f"batch={energy_unit_cell}, individual={individual_energies_fire[step]}"
        )
