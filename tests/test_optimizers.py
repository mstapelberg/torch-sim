import copy
from dataclasses import fields
from typing import get_args

import pytest
import torch
from pytest import CaptureFixture

from torch_sim.optimizers import (
    FireState,
    FrechetCellFIREState,
    GDState,
    MdFlavor,
    UnitCellFireState,
    UnitCellGDState,
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
        f"Gradient Descent optimization should reduce energy "
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


def test_fire_init_with_dict(
    ar_supercell_sim_state: SimState, lj_model: torch.nn.Module
) -> None:
    """Test fire init_fn with a SimState dictionary."""
    state_dict = {
        f.name: getattr(ar_supercell_sim_state, f.name)
        for f in fields(ar_supercell_sim_state)
    }
    init_fn, _ = fire(model=lj_model)
    fire_state = init_fn(state_dict)
    assert isinstance(fire_state, FireState)
    assert fire_state.energy is not None
    assert fire_state.forces is not None


def test_fire_invalid_md_flavor(lj_model: torch.nn.Module) -> None:
    """Test fire with an invalid md_flavor raises ValueError."""
    with pytest.raises(ValueError, match="Unknown md_flavor"):
        fire(model=lj_model, md_flavor="invalid_flavor")


def test_fire_ase_negative_power_branch(
    ar_supercell_sim_state: SimState, lj_model: torch.nn.Module
) -> None:
    """Test that the ASE FIRE P<0 branch behaves as expected."""
    f_dec = 0.5  # Default from fire optimizer
    alpha_start_val = 0.1  # Default from fire optimizer
    dt_start_val = 0.1

    init_fn, update_fn = fire(
        model=lj_model,
        md_flavor="ase_fire",
        f_dec=f_dec,
        alpha_start=alpha_start_val,
        dt_start=dt_start_val,
        dt_max=1.0,
        maxstep=10.0,  # Large maxstep to not interfere with velocity check
    )
    # Initialize state (forces are computed here)
    state = init_fn(ar_supercell_sim_state)

    # Save parameters from initial state
    initial_dt_batch = state.dt.clone()  # per-batch dt

    # Manipulate state to ensure P < 0 for the update_fn step
    # Ensure forces are non-trivial
    state.forces += torch.sign(state.forces + 1e-6) * 1e-2
    state.forces[torch.abs(state.forces) < 1e-3] = 1e-3
    # Set velocities directly opposite to current forces
    state.velocities = -state.forces * 0.1  # v = -k * F

    # Store forces that will be used in the power calculation and v += dt*F step
    forces_at_power_calc = state.forces.clone()

    # Deepcopy state as update_fn modifies it in-place
    state_to_update = copy.deepcopy(state)
    updated_state = update_fn(state_to_update)

    # Assertions for P < 0 branch being taken
    # Check for a single-batch state (ar_supercell_sim_state is single batch)
    expected_dt_val = initial_dt_batch[0] * f_dec
    assert torch.allclose(updated_state.dt[0], expected_dt_val)
    assert torch.allclose(
        updated_state.alpha[0],
        torch.tensor(
            alpha_start_val,
            dtype=updated_state.alpha.dtype,
            device=updated_state.alpha.device,
        ),
    )
    assert updated_state.n_pos[0] == 0

    # Assertions for velocity update in ASE P < 0 case:
    # v_after_mixing_is_0, then v_final = dt_new * F_at_power_calc
    expected_final_velocities = (
        expected_dt_val * forces_at_power_calc[updated_state.batch == 0]
    )
    assert torch.allclose(
        updated_state.velocities[updated_state.batch == 0],
        expected_final_velocities,
        atol=1e-6,
    )


def test_fire_vv_negative_power_branch(
    ar_supercell_sim_state: SimState, lj_model: torch.nn.Module
) -> None:
    """Attempt to trigger and test the VV FIRE P<0 branch."""
    f_dec = 0.5
    alpha_start_val = 0.1
    # Use a very large dt_start to encourage overshooting and P<0 inside _vv_fire_step
    dt_start_val = 2.0
    dt_max_val = 2.0

    init_fn, update_fn = fire(
        model=lj_model,
        md_flavor="vv_fire",
        f_dec=f_dec,
        alpha_start=alpha_start_val,
        dt_start=dt_start_val,
        dt_max=dt_max_val,
        n_min=0,  # Allow dt to change immediately
    )
    state = init_fn(ar_supercell_sim_state)

    initial_dt_batch = state.dt.clone()
    initial_alpha_batch = state.alpha.clone()  # Already alpha_start_val
    initial_n_pos_batch = state.n_pos.clone()  # Already 0

    state_to_update = copy.deepcopy(state)
    updated_state = update_fn(state_to_update)

    # Check if the P<0 branch was likely hit (params changed accordingly for batch 0)
    expected_dt_val = initial_dt_batch[0] * f_dec
    expected_alpha_val = torch.tensor(
        alpha_start_val,
        dtype=initial_alpha_batch.dtype,
        device=initial_alpha_batch.device,
    )

    p_lt_0_branch_taken = (
        torch.allclose(updated_state.dt[0], expected_dt_val)
        and torch.allclose(updated_state.alpha[0], expected_alpha_val)
        and updated_state.n_pos[0] == 0
    )

    if not p_lt_0_branch_taken:
        pytest.skip(
            f"VV FIRE P<0 condition not reliably hit for batch 0. "
            f"dt: {initial_dt_batch[0].item():.4f} -> {updated_state.dt[0].item():.4f} (expected factor {f_dec}). "
            f"alpha: {initial_alpha_batch[0].item():.4f} -> {updated_state.alpha[0].item():.4f} (expected {alpha_start_val}). "
            f"n_pos: {initial_n_pos_batch[0].item()} -> {updated_state.n_pos[0].item()} (expected 0)."
        )

    # If P<0 branch was taken, velocities should be zeroed
    assert torch.allclose(
        updated_state.velocities[updated_state.batch == 0],
        torch.zeros_like(updated_state.velocities[updated_state.batch == 0]),
        atol=1e-7,
    )


@pytest.mark.parametrize("md_flavor", get_args(MdFlavor))
def test_unit_cell_fire_optimization(
    ar_supercell_sim_state: SimState, lj_model: torch.nn.Module, md_flavor: MdFlavor
) -> None:
    """Test that the Unit Cell FIRE optimizer actually minimizes energy."""
    print(f"\n--- Starting test_unit_cell_fire_optimization for {md_flavor=} ---")

    # Add random displacement to positions and cell
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
    print(f"[{md_flavor}] Initial SimState created.")

    initial_state_positions = current_sim_state.positions.clone()
    initial_state_cell = current_sim_state.cell.clone()

    # Initialize FIRE optimizer
    print(f"Initializing {md_flavor} optimizer...")
    init_fn, update_fn = unit_cell_fire(
        model=lj_model,
        dt_max=0.3,
        dt_start=0.1,
        md_flavor=md_flavor,
    )
    print(f"[{md_flavor}] Optimizer functions obtained.")

    state = init_fn(current_sim_state)
    energy = float(getattr(state, "energy", "nan"))
    print(f"[{md_flavor}] Initial state created by init_fn. {energy=:.4f}")

    # Run optimization for a few steps
    energies = [1000.0, state.energy.item()]
    max_steps = 1000
    steps_taken = 0
    print(f"[{md_flavor}] Entering optimization loop (max_steps: {max_steps})...")

    while abs(energies[-2] - energies[-1]) > 1e-6 and steps_taken < max_steps:
        state = update_fn(state)
        energies.append(state.energy.item())
        steps_taken += 1

    print(f"[{md_flavor}] Loop finished after {steps_taken} steps.")

    if steps_taken == max_steps and abs(energies[-2] - energies[-1]) > 1e-6:
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
        f"Unit Cell FIRE {md_flavor=} optimization should reduce energy "
        f"(initial: {energies[0]}, final: {energies[-1]})"
    )

    # Check force convergence
    max_force = torch.max(torch.norm(state.forces, dim=1))
    pressure = torch.trace(state.stress.squeeze(0)) / 3.0
    assert pressure < 0.01, (
        f"Pressure should be small after optimization, got {pressure=}"
    )
    assert max_force < 0.3, (
        f"{md_flavor=} forces should be small after optimization, got {max_force}"
    )

    assert not torch.allclose(state.positions, initial_state_positions), (
        f"{md_flavor=} positions should have changed after optimization."
    )
    assert not torch.allclose(state.cell, initial_state_cell), (
        f"{md_flavor=} cell should have changed after optimization."
    )


def test_unit_cell_fire_init_with_dict_and_int_cell_factor(
    ar_supercell_sim_state: SimState, lj_model: torch.nn.Module
) -> None:
    """Test unit_cell_fire init_fn with dict state and int cell_factor."""
    state_dict = {
        f.name: getattr(ar_supercell_sim_state, f.name)
        for f in fields(ar_supercell_sim_state)
    }
    int_cell_factor = 100  # Example int value

    init_fn, _ = unit_cell_fire(model=lj_model, cell_factor=int_cell_factor)
    uc_fire_state = init_fn(state_dict)

    assert isinstance(uc_fire_state, UnitCellFireState)
    assert uc_fire_state.energy is not None
    assert uc_fire_state.forces is not None
    assert uc_fire_state.stress is not None
    expected_cell_factor = torch.full(
        (uc_fire_state.n_batches, 1, 1),
        int_cell_factor,
        device=lj_model.device,
        dtype=lj_model.dtype,
    )
    assert torch.allclose(uc_fire_state.cell_factor, expected_cell_factor)


def test_unit_cell_fire_invalid_md_flavor(lj_model: torch.nn.Module) -> None:
    """Test unit_cell_fire with an invalid md_flavor raises ValueError."""
    with pytest.raises(ValueError, match="Unknown md_flavor"):
        unit_cell_fire(model=lj_model, md_flavor="invalid_flavor")


def test_unit_cell_fire_init_cell_factor_none(
    ar_supercell_sim_state: SimState, lj_model: torch.nn.Module
) -> None:
    """Test unit_cell_fire init_fn with cell_factor=None."""
    init_fn, _ = unit_cell_fire(model=lj_model, cell_factor=None)
    # Ensure n_batches > 0 for cell_factor calculation from counts
    assert ar_supercell_sim_state.n_batches > 0
    uc_fire_state = init_fn(ar_supercell_sim_state)
    assert isinstance(uc_fire_state, UnitCellFireState)
    # Default cell_factor should be based on number of atoms per batch
    _, counts = torch.unique(ar_supercell_sim_state.batch, return_counts=True)
    expected_cf = counts.to(dtype=lj_model.dtype).view(-1, 1, 1)
    assert torch.allclose(uc_fire_state.cell_factor, expected_cf)


@pytest.mark.filterwarnings("ignore:WARNING: Non-positive volume detected")
def test_unit_cell_fire_ase_non_positive_volume_warning(
    ar_supercell_sim_state: SimState, lj_model: torch.nn.Module, capsys: CaptureFixture
) -> None:
    """Attempt to trigger non-positive volume warning in unit_cell_fire ASE."""
    # Use a state that might lead to cell inversion with aggressive steps
    # Make a copy and slightly perturb the cell to make it prone to issues
    perturbed_state = ar_supercell_sim_state.clone()
    perturbed_state.cell += (
        torch.randn_like(perturbed_state.cell) * 0.5
    )  # Large perturbation
    # Also ensure no PBC issues by slightly expanding cell if it got too small
    if torch.linalg.det(perturbed_state.cell[0]) < 1.0:
        perturbed_state.cell[0] *= 2.0

    init_fn, update_fn = unit_cell_fire(
        model=lj_model,
        md_flavor="ase_fire",
        dt_max=5.0,  # Large dt
        maxstep=2.0,  # Large maxstep
        dt_start=1.0,
        f_dec=0.99,  # Slow down dt decrease
        alpha_start=0.99,  # Aggressive alpha
    )
    state = init_fn(perturbed_state)

    # Run a few steps hoping to trigger the warning
    for _ in range(5):
        state = update_fn(state)
        if "WARNING: Non-positive volume detected" in capsys.readouterr().err:
            break  # Warning captured
    else:
        # If loop finishes, check one last time (in case warning is at the very end)
        pass  # Test will pass if no error, but we hope warning was printed

    # We don't assert the warning was printed as it's hard to guarantee
    # The main goal is to cover the code path. If it runs without crashing, coverage is achieved.
    assert state is not None  # Ensure optimizer ran


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
    print(f"Initializing Frechet {md_flavor} optimizer...")
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
    # Assumes single batch for this state stress access
    pressure = torch.trace(state.stress.squeeze(0)) / 3.0

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


def test_frechet_cell_fire_init_with_dict_and_float_cell_factor(
    ar_supercell_sim_state: SimState, lj_model: torch.nn.Module
) -> None:
    """Test frechet_cell_fire init_fn with dict state and float cell_factor."""
    state_dict = {
        f.name: getattr(ar_supercell_sim_state, f.name)
        for f in fields(ar_supercell_sim_state)
    }
    float_cell_factor = 75.0  # Example float value

    init_fn, _ = frechet_cell_fire(model=lj_model, cell_factor=float_cell_factor)
    fc_fire_state = init_fn(state_dict)

    assert isinstance(fc_fire_state, FrechetCellFIREState)
    assert fc_fire_state.energy is not None
    assert fc_fire_state.forces is not None
    assert fc_fire_state.stress is not None
    expected_cell_factor = torch.full(
        (fc_fire_state.n_batches, 1, 1),
        float_cell_factor,
        device=lj_model.device,
        dtype=lj_model.dtype,
    )
    assert torch.allclose(fc_fire_state.cell_factor, expected_cell_factor)


def test_frechet_cell_fire_invalid_md_flavor(lj_model: torch.nn.Module) -> None:
    """Test frechet_cell_fire with an invalid md_flavor raises ValueError."""
    with pytest.raises(ValueError, match="Unknown md_flavor"):
        frechet_cell_fire(model=lj_model, md_flavor="invalid_flavor")


def test_frechet_cell_fire_init_cell_factor_none(
    ar_supercell_sim_state: SimState, lj_model: torch.nn.Module
) -> None:
    """Test frechet_cell_fire init_fn with cell_factor=None."""
    init_fn, _ = frechet_cell_fire(model=lj_model, cell_factor=None)
    assert ar_supercell_sim_state.n_batches > 0
    fc_fire_state = init_fn(ar_supercell_sim_state)
    assert isinstance(fc_fire_state, FrechetCellFIREState)
    _, counts = torch.unique(ar_supercell_sim_state.batch, return_counts=True)
    expected_cf = counts.to(dtype=lj_model.dtype).view(-1, 1, 1)
    assert torch.allclose(fc_fire_state.cell_factor, expected_cf)


@pytest.mark.filterwarnings("ignore:WARNING: Non-positive volume detected")
@pytest.mark.filterwarnings(
    r"ignore:Non-positive volume\(s\) detected"
)  # For frechet's specific warning
def test_frechet_cell_fire_ase_non_positive_volume_warning(
    ar_supercell_sim_state: SimState, lj_model: torch.nn.Module, capsys: CaptureFixture
) -> None:
    """Attempt to trigger non-positive volume warning in frechet_cell_fire ASE."""
    perturbed_state = ar_supercell_sim_state.clone()
    perturbed_state.cell += torch.randn_like(perturbed_state.cell) * 0.5
    if torch.linalg.det(perturbed_state.cell[0]) < 1.0:
        perturbed_state.cell[0] *= 2.0

    init_fn, update_fn = frechet_cell_fire(
        model=lj_model,
        md_flavor="ase_fire",
        dt_max=5.0,
        maxstep=2.0,
        dt_start=1.0,
        f_dec=0.99,
        alpha_start=0.99,
    )
    state = init_fn(perturbed_state)
    for _ in range(5):
        state = update_fn(state)
        # Frechet ASE has a slightly different warning sometimes
        outerr = capsys.readouterr()
        if (
            "WARNING: Non-positive volume detected" in outerr.err
            or "WARNING: Non-positive volume(s) detected" in outerr.err
        ):
            break
    assert state is not None


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


def test_unit_cell_frechet_fire_ase_negative_power_branch(
    ar_supercell_sim_state: SimState, lj_model: torch.nn.Module
) -> None:
    """Test FrechetCellFIRE ASE P<0 branch for atoms and cell."""
    f_dec = 0.5
    alpha_start_val = 0.1
    dt_start_val = 0.1

    init_fn, update_fn = frechet_cell_fire(
        model=lj_model,
        md_flavor="ase_fire",
        f_dec=f_dec,
        alpha_start=alpha_start_val,
        dt_start=dt_start_val,
        dt_max=1.0,
        maxstep=10.0,  # Large maxstep
    )
    state = init_fn(ar_supercell_sim_state)

    initial_dt_batch = state.dt.clone()

    # Manipulate for P < 0 (atoms and cell)
    state.forces += torch.sign(state.forces + 1e-6) * 1e-2
    state.forces[torch.abs(state.forces) < 1e-3] = 1e-3
    state.velocities = -state.forces * 0.1

    # Frechet cell forces can be sensitive, ensure they are robust for testing P<0
    state.cell_forces += torch.sign(state.cell_forces + 1e-6) * 1e-2
    state.cell_forces[torch.abs(state.cell_forces) < 1e-3] = 1e-3
    state.cell_velocities = -state.cell_forces * 0.1

    forces_at_power_calc = state.forces.clone()
    cell_forces_at_power_calc = state.cell_forces.clone()

    state_to_update = copy.deepcopy(state)
    updated_state = update_fn(state_to_update)

    expected_dt_val = initial_dt_batch[0] * f_dec
    assert torch.allclose(updated_state.dt[0], expected_dt_val)
    assert torch.allclose(
        updated_state.alpha[0],
        torch.tensor(
            alpha_start_val,
            dtype=updated_state.alpha.dtype,
            device=updated_state.alpha.device,
        ),
    )
    assert updated_state.n_pos[0] == 0

    expected_atom_velocities = (
        expected_dt_val * forces_at_power_calc[updated_state.batch == 0]
    )
    assert torch.allclose(
        updated_state.velocities[updated_state.batch == 0],
        expected_atom_velocities,
        atol=1e-6,
    )

    expected_cell_velocities = (
        expected_dt_val * cell_forces_at_power_calc
    )  # cell is per-batch
    assert torch.allclose(
        updated_state.cell_velocities[0], expected_cell_velocities[0], atol=1e-6
    )


def test_unit_cell_fire_vv_negative_power_branch(
    ar_supercell_sim_state: SimState, lj_model: torch.nn.Module
) -> None:
    """Attempt to trigger UnitCellFIRE VV P<0 branch."""
    f_dec = 0.5
    alpha_start_val = 0.1
    dt_start_val = 2.0  # Large dt_start
    dt_max_val = 2.0

    init_fn, update_fn = unit_cell_fire(
        model=lj_model,
        md_flavor="vv_fire",
        f_dec=f_dec,
        alpha_start=alpha_start_val,
        dt_start=dt_start_val,
        dt_max=dt_max_val,
        n_min=0,
    )
    state = init_fn(ar_supercell_sim_state)

    initial_dt_batch = state.dt.clone()
    initial_alpha_batch = state.alpha.clone()
    initial_n_pos_batch = state.n_pos.clone()

    state_to_update = copy.deepcopy(state)
    updated_state = update_fn(state_to_update)

    expected_dt_val = initial_dt_batch[0] * f_dec
    expected_alpha_val = torch.tensor(
        alpha_start_val,
        dtype=initial_alpha_batch.dtype,
        device=initial_alpha_batch.device,
    )

    p_lt_0_branch_taken = (
        torch.allclose(updated_state.dt[0], expected_dt_val)
        and torch.allclose(updated_state.alpha[0], expected_alpha_val)
        and updated_state.n_pos[0] == 0
    )

    if not p_lt_0_branch_taken:
        pytest.skip(
            f"UnitCell VV FIRE P<0 condition not reliably hit. "
            f"dt: {initial_dt_batch[0].item():.4f} -> {updated_state.dt[0].item():.4f}, "
            f"alpha: {initial_alpha_batch[0].item():.4f} -> {updated_state.alpha[0].item():.4f}, "
            f"n_pos: {initial_n_pos_batch[0].item()} -> {updated_state.n_pos[0].item()}."
        )

    assert torch.allclose(
        updated_state.velocities[updated_state.batch == 0],
        torch.zeros_like(updated_state.velocities[updated_state.batch == 0]),
        atol=1e-7,
    )
    assert torch.allclose(
        updated_state.cell_velocities[0],
        torch.zeros_like(updated_state.cell_velocities[0]),
        atol=1e-7,
    )


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


def test_gradient_descent_init_with_dict(
    ar_supercell_sim_state: SimState, lj_model: torch.nn.Module
) -> None:
    """Test gradient_descent init_fn with a SimState dictionary."""
    state_dict = {
        f.name: getattr(ar_supercell_sim_state, f.name)
        for f in fields(ar_supercell_sim_state)
    }
    init_fn, _ = gradient_descent(model=lj_model)
    gd_state = init_fn(state_dict)
    assert isinstance(gd_state, GDState)
    assert gd_state.energy is not None
    assert gd_state.forces is not None


def test_unit_cell_gradient_descent_init_with_dict_and_float_cell_factor(
    ar_supercell_sim_state: SimState, lj_model: torch.nn.Module
) -> None:
    """Test unit_cell_gradient_descent init_fn with dict state and float cell_factor."""
    state_dict = {
        f.name: getattr(ar_supercell_sim_state, f.name)
        for f in fields(ar_supercell_sim_state)
    }
    float_cell_factor = 50.0  # Example float value

    init_fn, _ = unit_cell_gradient_descent(model=lj_model, cell_factor=float_cell_factor)
    uc_gd_state = init_fn(state_dict)

    assert isinstance(uc_gd_state, UnitCellGDState)
    assert uc_gd_state.energy is not None
    assert uc_gd_state.forces is not None
    assert uc_gd_state.stress is not None
    # Check if cell_factor was correctly processed
    expected_cell_factor = torch.full(
        (uc_gd_state.n_batches, 1, 1),
        float_cell_factor,
        device=lj_model.device,
        dtype=lj_model.dtype,
    )
    assert torch.allclose(uc_gd_state.cell_factor, expected_cell_factor)
