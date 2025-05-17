"""Structural optimization with MACE using FIRE optimizer.
Comparing the ASE and VV FIRE optimizers.
"""

# /// script
# dependencies = [
#     "mace-torch>=0.3.12",
# ]
# ///

import os
import time
from typing import Literal

import numpy as np
import torch
from ase.build import bulk
from ase.optimize import FIRE as ASEFIRE
from ase.filters import FrechetCellFilter
from mace.calculators.foundations_models import mace_mp
from mace.calculators.foundations_models import mace_mp as mace_mp_calculator_for_ase
import matplotlib.pyplot as plt

import torch_sim as ts
from torch_sim.models.mace import MaceModel, MaceUrls
from torch_sim.optimizers import fire, frechet_cell_fire, GDState
from torch_sim.state import SimState


# Set device, data type and unit conversion
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32
unit_conv = ts.units.UnitConversion

# Option 1: Load the raw model from the downloaded model
loaded_model = mace_mp(
    model=MaceUrls.mace_mpa_medium,
    return_raw_model=True,
    default_dtype=dtype,
    device=device,
)

# Number of steps to run
max_iterations = 10 if os.getenv("CI") else 500
supercell_scale = (1, 1, 1) if os.getenv("CI") else (3, 2, 2)
ase_max_optimizer_steps = max_iterations * 10 # Max steps for each individual ASE optimization run

# Set random seed for reproducibility
rng = np.random.default_rng(seed=0)

# Create diamond cubic Silicon
si_dc = bulk("Si", "diamond", a=5.21, cubic=True).repeat(supercell_scale)
si_dc.positions += 0.1 * rng.standard_normal(si_dc.positions.shape).clip(-1, 1)

# Create FCC Copper
cu_dc = bulk("Cu", "fcc", a=3.85).repeat([r + 1 for r in supercell_scale])
cu_dc.positions += 0.1 * rng.standard_normal(cu_dc.positions.shape).clip(-1, 1)

# Create BCC Iron
fe_dc = bulk("Fe", "bcc", a=2.95).repeat([r + 1 for r in supercell_scale])
fe_dc.positions += 0.1 * rng.standard_normal(fe_dc.positions.shape).clip(-1, 1)

si_dc_vac = si_dc.copy()
si_dc_vac.positions += 0.1 * rng.standard_normal(si_dc_vac.positions.shape).clip(-1, 1)
# select 2 numbers in range 0 to len(si_dc_vac)
indices = rng.choice(len(si_dc_vac), size=2, replace=False)
for idx in indices:
    si_dc_vac.pop(idx)


cu_dc_vac = cu_dc.copy()
cu_dc_vac.positions += 0.1 * rng.standard_normal(cu_dc_vac.positions.shape).clip(-1, 1)
# remove 2 atoms from cu_dc_vac at random
indices = rng.choice(len(cu_dc_vac), size=2, replace=False)
for idx in indices:
    index = idx + 3
    if index < len(cu_dc_vac):
        cu_dc_vac.pop(index)
    else:
        print(f"Index {index} is out of bounds for cu_dc_vac")
        cu_dc_vac.pop(0)

fe_dc_vac = fe_dc.copy()
fe_dc_vac.positions += 0.1 * rng.standard_normal(fe_dc_vac.positions.shape).clip(-1, 1)
# remove 2 atoms from fe_dc_vac at random
indices = rng.choice(len(fe_dc_vac), size=2, replace=False)
for idx in indices:
    index = idx + 2
    if index < len(fe_dc_vac):
        fe_dc_vac.pop(index)
    else:
        print(f"Index {index} is out of bounds for fe_dc_vac")
        fe_dc_vac.pop(0)


# Create a list of our atomic systems
atoms_list = [si_dc, cu_dc, fe_dc, si_dc_vac, cu_dc_vac]

# Print structure information
print(f"Silicon atoms: {len(si_dc)}")
print(f"Copper atoms: {len(cu_dc)}")
print(f"Iron atoms: {len(fe_dc)}")
print(f"Total number of structures: {len(atoms_list)}")

# Create batched model
model = MaceModel(
    model=loaded_model,
    device=device,
    compute_forces=True,
    compute_stress=True,
    dtype=dtype,
    enable_cueq=False,
)

# Convert atoms to state
state = ts.io.atoms_to_state(atoms_list, device=device, dtype=dtype)
# Run initial inference
initial_energies = model(state)["energy"]


def run_optimization(
    initial_state: SimState,
    optimizer_type: Literal["torch_sim", "ase"],
    # For torch_sim:
    ts_md_flavor: Literal["vv_fire", "ase_fire"] | None = None,
    ts_use_frechet: bool = False, # To decide between fire() and frechet_cell_fire()
    # For ASE:
    ase_use_frechet_filter: bool = False,
    # Common:
    force_tol: float = 0.05,
) -> tuple[torch.Tensor, SimState]:
    """Runs optimization and returns convergence steps and final state."""
    if optimizer_type == "torch_sim":
        assert ts_md_flavor is not None, "ts_md_flavor must be provided for torch_sim"
        print(
            f"\n--- Running Torch-Sim optimization: flavor={ts_md_flavor}, "
            f"frechet_cell_opt={ts_use_frechet}, force_tol={force_tol} ---"
        )
        start_time = time.perf_counter()

        if ts_use_frechet:
            # Uses frechet_cell_fire for combined cell and position optimization
            init_fn_opt, update_fn_opt = frechet_cell_fire(
                model=model, md_flavor=ts_md_flavor
            )
        else:
            # Uses fire for position-only optimization
            init_fn_opt, update_fn_opt = fire(model=model, md_flavor=ts_md_flavor)

        opt_state = init_fn_opt(initial_state.clone())

        batcher = ts.InFlightAutoBatcher(
            model=model, # The MaceModel wrapper
            memory_scales_with="n_atoms",
            max_memory_scaler=1000,
            max_iterations=max_iterations,
            return_indices=True,
        )
        batcher.load_states(opt_state)

        total_structures = opt_state.n_batches
        convergence_steps = torch.full(
            (total_structures,), -1, dtype=torch.long, device=device
        )
        convergence_fn = ts.generate_force_convergence_fn(force_tol=force_tol)
        converged_tensor_global = torch.zeros(
            total_structures, dtype=torch.bool, device=device
        )
        global_step = 0
        all_converged_states = []
        convergence_tensor_for_batcher = None
        last_active_state = opt_state

        while True:
            result = batcher.next_batch(
                last_active_state, convergence_tensor_for_batcher
            )
            opt_state, converged_states_from_batcher, current_indices_list = result
            all_converged_states.extend(converged_states_from_batcher)

            if opt_state is None:
                print("All structures converged or batcher reached max iterations.")
                break
            
            last_active_state = opt_state
            current_indices = torch.tensor(
                current_indices_list, dtype=torch.long, device=device
            )

            steps_this_round = 10
            for _ in range(steps_this_round):
                opt_state = update_fn_opt(opt_state)
            global_step += steps_this_round

            convergence_tensor_for_batcher = convergence_fn(opt_state, None)
            newly_converged_mask_local = convergence_tensor_for_batcher & (
                convergence_steps[current_indices] == -1
            )
            converged_indices_global = current_indices[newly_converged_mask_local]

            if converged_indices_global.numel() > 0:
                convergence_steps[converged_indices_global] = global_step
                converged_tensor_global[converged_indices_global] = True
                total_converged_frac = converged_tensor_global.sum().item() / total_structures
                print(
                    f"{global_step=}: Converged indices {converged_indices_global.tolist()}, "
                    f"Total converged: {total_converged_frac:.2%}"
                )

            if global_step % 50 == 0:
                total_converged_frac = converged_tensor_global.sum().item() / total_structures
                active_structures = opt_state.n_batches if opt_state else 0
                print(
                    f"{global_step=}: Active structures: {active_structures}, "
                    f"Total converged: {total_converged_frac:.2%}"
                )
        
        final_states_list = batcher.restore_original_order(all_converged_states)
        final_state_concatenated = ts.concatenate_states(final_states_list)
        end_time = time.perf_counter()
        print(
            f"Finished Torch-Sim ({ts_md_flavor}, frechet={ts_use_frechet}) in "
            f"{end_time - start_time:.2f} seconds."
        )
        return convergence_steps, final_state_concatenated

    elif optimizer_type == "ase":
        print(
            f"\n--- Running ASE optimization: frechet_filter={ase_use_frechet_filter}, "
            f"force_tol={force_tol} ---"
        )
        start_time = time.perf_counter()

        individual_initial_states = initial_state.split()
        num_structures = len(individual_initial_states)
        final_ase_atoms_list = []
        convergence_steps_list = []

        for i, single_sim_state in enumerate(individual_initial_states):
            print(f"Optimizing structure {i+1}/{num_structures} with ASE...")
            ase_atoms_orig = ts.io.state_to_atoms(single_sim_state)[0]

            ase_calc_instance = mace_mp_calculator_for_ase(
                model=MaceUrls.mace_mpa_medium,
                device=device,
                default_dtype=str(dtype).split('.')[-1],
            )
            ase_atoms_orig.calc = ase_calc_instance

            optim_target_atoms = ase_atoms_orig
            if ase_use_frechet_filter:
                print(f"Applying FrechetCellFilter to structure {i+1}")
                optim_target_atoms = FrechetCellFilter(ase_atoms_orig)
            
            dyn = ASEFIRE(optim_target_atoms, trajectory=None, logfile=None)

            try:
                dyn.run(fmax=force_tol, steps=ase_max_optimizer_steps)
                if dyn.converged():
                    convergence_steps_list.append(dyn.nsteps)
                    print(f"ASE structure {i+1} converged in {dyn.nsteps} steps.")
                else:
                    print(
                        f"ASE optimization for structure {i+1} did not converge within "
                        f"{ase_max_optimizer_steps} steps. Steps taken: {dyn.nsteps}."
                    )
                    convergence_steps_list.append(-1)
            except Exception as e:
                print(f"ASE optimization failed for structure {i+1}: {e}")
                convergence_steps_list.append(-1)

            final_ase_atoms_list.append(optim_target_atoms.atoms if ase_use_frechet_filter else ase_atoms_orig)

        # Convert list of final ASE atoms objects back to a base SimState first
        # to easily get positions, cell, etc.
        # However, ts.io.atoms_to_state might not preserve all attributes needed for GDState directly.
        # It's better to extract all required components directly from final_ase_atoms_list.

        all_positions = []
        all_masses = []
        all_atomic_numbers = []
        all_cells = []
        all_batches_for_gd = []
        final_energies_ase = []
        final_forces_ase_tensors = [] # List to store force tensors

        current_atom_offset = 0
        for batch_idx, ats_final in enumerate(final_ase_atoms_list):
            all_positions.append(torch.tensor(ats_final.get_positions(), device=device, dtype=dtype))
            all_masses.append(torch.tensor(ats_final.get_masses(), device=device, dtype=dtype))
            all_atomic_numbers.append(torch.tensor(ats_final.get_atomic_numbers(), device=device, dtype=torch.long))
            # ASE cell is row-vector, SimState expects column-vector
            all_cells.append(torch.tensor(ats_final.get_cell().array.T, device=device, dtype=dtype))
            
            num_atoms_in_current = len(ats_final)
            all_batches_for_gd.append(torch.full((num_atoms_in_current,), batch_idx, device=device, dtype=torch.long))
            current_atom_offset += num_atoms_in_current

            try:
                if ats_final.calc is None:
                    print(f"Re-attaching ASE calculator for final energy/forces for structure {batch_idx}.")
                    temp_calc = mace_mp_calculator_for_ase(
                        model=MaceUrls.mace_mpa_medium, device=device, default_dtype=str(dtype).split('.')[-1]
                    )
                    ats_final.calc = temp_calc
                final_energies_ase.append(ats_final.get_potential_energy())
                final_forces_ase_tensors.append(torch.tensor(ats_final.get_forces(), device=device, dtype=dtype))
            except Exception as e:
                print(f"Could not get final energy/forces for an ASE structure {batch_idx}: {e}")
                final_energies_ase.append(float('nan'))
                # Append a zero tensor of appropriate shape if forces fail, or handle error
                # For GDState, forces are required. If any structure fails, GDState creation might fail.
                # We need to ensure all_positions, etc. are also correctly populated even on failure.
                # For now, let's assume if energy fails, forces might also, and GDState might be problematic.
                # A robust solution would be to skip failed structures or return None.
                # For now, let's make forces a zero tensor of expected shape if it fails.
                if all_positions and len(all_positions[-1]) > 0:
                     final_forces_ase_tensors.append(torch.zeros_like(all_positions[-1]))
                else: # Cannot determine shape, this path is problematic
                     final_forces_ase_tensors.append(torch.empty((0,3), device=device, dtype=dtype))


        if not all_positions: # If all optimizations failed early
            print("Warning: No successful ASE structures to form GDState.")
            return torch.tensor(convergence_steps_list, dtype=torch.long, device=device), None


        # Concatenate all parts
        concatenated_positions = torch.cat(all_positions, dim=0)
        concatenated_masses = torch.cat(all_masses, dim=0)
        concatenated_atomic_numbers = torch.cat(all_atomic_numbers, dim=0)
        concatenated_cells = torch.stack(all_cells, dim=0) # Cells are (N_batch, 3, 3)
        concatenated_batch_indices = torch.cat(all_batches_for_gd, dim=0)
        
        concatenated_energies = torch.tensor(final_energies_ase, device=device, dtype=dtype)
        concatenated_forces = torch.cat(final_forces_ase_tensors, dim=0)

        # Check for NaN energies which might cause issues
        if torch.isnan(concatenated_energies).any():
            print("Warning: NaN values found in final ASE energies. GDState energy tensor will contain NaNs.")
            # Consider replacing NaNs if GDState or subsequent ops can't handle them:
            # concatenated_energies = torch.nan_to_num(concatenated_energies, nan=0.0) # Example replacement

        # Create GDState instance
        # pbc is global, taken from initial_state
        final_state_as_gd = GDState(
            positions=concatenated_positions,
            masses=concatenated_masses,
            cell=concatenated_cells,
            pbc=initial_state.pbc, # Assuming pbc is constant and global
            atomic_numbers=concatenated_atomic_numbers,
            batch=concatenated_batch_indices,
            energy=concatenated_energies,
            forces=concatenated_forces,
        )
        
        convergence_steps = torch.tensor(convergence_steps_list, dtype=torch.long, device=device)

        end_time = time.perf_counter()
        print(
            f"Finished ASE optimization (frechet_filter={ase_use_frechet_filter}) "
            f"in {end_time - start_time:.2f} seconds."
        )
        return convergence_steps, final_state_as_gd
    else:
        raise ValueError(f"Unknown optimizer_type: {optimizer_type}")


# --- Main Script ---
force_tol = 0.05

# Configurations to test
configs_to_run = [
    {
        "name": "torch-sim VV-FIRE (PosOnly)",
        "type": "torch_sim", "ts_md_flavor": "vv_fire", "ts_use_frechet": False,
    },
    {
        "name": "torch-sim ASE-FIRE (PosOnly)",
        "type": "torch_sim", "ts_md_flavor": "ase_fire", "ts_use_frechet": False,
    },
    {
        "name": "torch-sim VV-FIRE (Frechet Cell)",
        "type": "torch_sim", "ts_md_flavor": "vv_fire", "ts_use_frechet": True,
    },
    {
        "name": "torch-sim ASE-FIRE (Frechet Cell)",
        "type": "torch_sim", "ts_md_flavor": "ase_fire", "ts_use_frechet": True,
    },
    {
        "name": "ASE FIRE (Native, CellOpt)", # Will optimize cell if stress is available
        "type": "ase", "ase_use_frechet_filter": False,
    },
    {
        "name": "ASE FIRE (Native Frechet Filter, CellOpt)",
        "type": "ase", "ase_use_frechet_filter": True,
    },
]

results_all = {}

for config_run in configs_to_run:
    print(f"\n\nStarting configuration: {config_run['name']}")
    # Get relevant params, providing defaults where necessary for the run_optimization call
    optimizer_type_val = config_run["type"]
    ts_md_flavor_val = config_run.get("ts_md_flavor") # Will be None for ASE type, handled by assert
    ts_use_frechet_val = config_run.get("ts_use_frechet", False)
    ase_use_frechet_filter_val = config_run.get("ase_use_frechet_filter", False)

    steps, final_state_opt = run_optimization(
        initial_state=state.clone(), # Use a fresh clone for each run
        optimizer_type=optimizer_type_val,
        ts_md_flavor=ts_md_flavor_val,
        ts_use_frechet=ts_use_frechet_val,
        ase_use_frechet_filter=ase_use_frechet_filter_val,
        force_tol=force_tol,
    )
    results_all[config_run["name"]] = {"steps": steps, "final_state": final_state_opt}


print("\n\n--- Overall Comparison ---")
print(f"{force_tol=:.2f} eV/Å")
print(f"Initial energies: {[f'{e.item():.3f}' for e in initial_energies]} eV")

for name, result_data in results_all.items():
    final_state_res = result_data["final_state"]
    steps_res = result_data["steps"]
    print(f"\nResults for: {name}")
    if final_state_res is not None and hasattr(final_state_res, 'energy') and final_state_res.energy is not None:
        energy_str = [f'{e.item():.3f}' for e in final_state_res.energy]
        print(f"  Final energies: {energy_str} eV")
    else:
        print(f"  Final energies: Not available or state is None")
    print(f"  Convergence steps: {steps_res.tolist()}")
    
    not_converged_indices = torch.where(steps_res == -1)[0].tolist()
    if not_converged_indices:
        print(f"  Did not converge for structure indices: {not_converged_indices}")

# Mean Displacement Comparisons
comparison_pairs = [
    ("torch-sim ASE-FIRE (PosOnly)", "ASE FIRE (Native, CellOpt)"), # Note: one is pos-only, other cell-opt
    ("torch-sim ASE-FIRE (Frechet Cell)", "ASE FIRE (Native Frechet Filter, CellOpt)"),
    ("torch-sim VV-FIRE (Frechet Cell)", "ASE FIRE (Native Frechet Filter, CellOpt)"),
    ("torch-sim VV-FIRE (PosOnly)", "torch-sim ASE-FIRE (PosOnly)"), # Original comparison
]

for name1, name2 in comparison_pairs:
    if name1 in results_all and name2 in results_all:
        state1 = results_all[name1]["final_state"]
        state2 = results_all[name2]["final_state"]

        if state1 is None or state2 is None:
            print(f"\nCannot compare {name1} and {name2}, one or both states are None.")
            continue
        
        state1_list = state1.split()
        
        state2_list = state2.split()
        
        if len(state1_list) != len(state2_list):
            print(f"\nCannot compare {name1} and {name2}, different number of structures.")
            continue

        mean_displacements = []
        for s1, s2 in zip(state1_list, state2_list, strict=True):
            if s1.n_atoms == 0 or s2.n_atoms == 0 : # Handle empty states if they occur
                mean_displacements.append(float('nan'))
                continue
            pos1_centered = s1.positions - s1.positions.mean(dim=0, keepdim=True)
            pos2_centered = s2.positions - s2.positions.mean(dim=0, keepdim=True)
            if pos1_centered.shape != pos2_centered.shape:
                 print(f"Warning: Shape mismatch for {name1} vs {name2} in structure. Skipping displacement calc.")
                 mean_displacements.append(float('nan'))
                 continue
            displacement = torch.norm(pos1_centered - pos2_centered, dim=1)
            mean_disp = torch.mean(displacement).item()
            mean_displacements.append(mean_disp)
        
        print(f"\nMean Disp ({name1} vs {name2}): {[f'{d:.4f}' for d in mean_displacements]} Å")
    else:
        print(f"\nSkipping displacement comparison for ({name1} vs {name2}), one or both results missing.")


# --- Plotting Results ---

# Names for the 5 structures for plotting labels
structure_names = [ats.get_chemical_formula() for ats in atoms_list]
# Make them more concise if needed:
structure_names = ["Si_bulk", "Cu_bulk", "Fe_bulk", "Si_vac", "Cu_vac"] # Example concise names
num_structures_plot = len(structure_names)


# --- Plot 1: Convergence Steps (Multi-bar per structure) ---
plot_methods_fig1 = list(results_all.keys())
num_methods_fig1 = len(plot_methods_fig1)
steps_data_fig1 = np.zeros((num_structures_plot, num_methods_fig1)) # rows: structures, cols: methods

for method_idx, method_name in enumerate(plot_methods_fig1):
    result_data = results_all[method_name]
    if result_data["final_state"] is None:
        steps_data_fig1[:, method_idx] = np.nan # Mark all as NaN for this method
        print(f"Plot1: Skipping steps for {method_name} as final_state is None.")
        continue
    
    steps_tensor = result_data["steps"].cpu().numpy()
    # Replace -1 (not converged) with a high value for plotting, or handle differently
    # For now, let's use a penalty (e.g., max_iterations_overall + buffer)
    # or keep as -1 and let user interpret. For bar plot, positive values are better.
    # Let's use np.nan for non-converged for now, and plot what did converge.
    # Or, use ase_max_optimizer_steps as a cap if not converged.
    # Let's use the actual steps, and if -1, plot it as a very high bar or special marker.
    # For a bar chart, a common approach is to cap it or show it differently.
    # We will cap at ase_max_optimizer_steps + a bit if -1
    penalty_steps = ase_max_optimizer_steps + 100 
    steps_plot_values = np.where(steps_tensor == -1, penalty_steps, steps_tensor)

    if len(steps_plot_values) == num_structures_plot:
        steps_data_fig1[:, method_idx] = steps_plot_values
    else:
        print(f"Warning: Mismatch in number of structures for steps in {method_name}. Expected {num_structures_plot}, got {len(steps_plot_values)}")
        steps_data_fig1[:, method_idx] = np.nan


fig1, ax1 = plt.subplots(figsize=(15, 8))
x_fig1 = np.arange(num_structures_plot) # x locations for the groups
width_fig1 = 0.8 / num_methods_fig1 # width of the bars

rects_all_fig1 = []
for i in range(num_methods_fig1):
    # Offset each bar in the group
    rects = ax1.bar(x_fig1 - 0.4 + (i + 0.5) * width_fig1, steps_data_fig1[:, i], width_fig1, label=plot_methods_fig1[i])
    rects_all_fig1.append(rects)
    # Add text for -1 (non-converged) if we plot them as penalty
    for bar_idx, bar_val in enumerate(steps_data_fig1[:, i]):
        original_step_val = results_all[plot_methods_fig1[i]]["steps"].cpu().numpy()[bar_idx]
        if original_step_val == -1:
             ax1.text(rects[bar_idx].get_x() + rects[bar_idx].get_width() / 2., 
                      rects[bar_idx].get_height() - 10, # Position slightly below top of penalty bar
                      'NC', ha='center', va='top', color='white', fontsize=7, weight='bold')


ax1.set_ylabel('Convergence Steps (NC = Not Converged, shown at penalty)')
ax1.set_xlabel('Structure')
ax1.set_title('Convergence Steps per Structure and Method')
ax1.set_xticks(x_fig1)
ax1.set_xticklabels(structure_names, rotation=45, ha="right")
ax1.legend(title="Optimization Method", bbox_to_anchor=(1.05, 1), loc='upper left')
ax1.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust rect to make space for legend


# --- Plot 2: Average Final Energy Difference from Baselines ---
baseline_ase_pos_only = "ASE FIRE (Native, CellOpt)"
baseline_ase_frechet = "ASE FIRE (Native Frechet Filter, CellOpt)"
avg_energy_diffs_fig2 = []
plot_names_fig2 = []

# Ensure baselines exist and have data
baseline_pos_only_data = results_all.get(baseline_ase_pos_only)
baseline_frechet_data = results_all.get(baseline_ase_frechet)

for name, result_data in results_all.items():
    if result_data["final_state"] is None:
        print(f"Plot2: Skipping energy diff for {name} as final_state is None.")
        continue

    plot_names_fig2.append(name)
    current_energies = result_data["final_state"].energy.cpu().numpy()
    
    chosen_baseline_energies = None
    is_baseline_self = False
    if "torch-sim" in name:
        if "PosOnly" in name:
            if baseline_pos_only_data and baseline_pos_only_data["final_state"] is not None:
                chosen_baseline_energies = baseline_pos_only_data["final_state"].energy.cpu().numpy()
        elif "Frechet Cell" in name:
            if baseline_frechet_data and baseline_frechet_data["final_state"] is not None:
                chosen_baseline_energies = baseline_frechet_data["final_state"].energy.cpu().numpy()
    elif name == baseline_ase_pos_only or name == baseline_ase_frechet:
        avg_energy_diffs_fig2.append(0.0) # Difference to self is 0
        is_baseline_self = True # Flag to handle text for baseline bars
        # continue # Continue was here, but we need to plot the baseline bar itself
    
    if not is_baseline_self: # Only calculate diff if not a baseline comparing to itself
        if chosen_baseline_energies is not None:
            if current_energies.shape == chosen_baseline_energies.shape:
                energy_diff = np.mean(current_energies - chosen_baseline_energies)
                avg_energy_diffs_fig2.append(energy_diff)
            else:
                avg_energy_diffs_fig2.append(np.nan)
                print(f"Shape mismatch for energy comparison: {name} vs its baseline")
        else:
            # If no appropriate baseline, or baseline data is missing
            print(f"Plot2: No appropriate baseline for {name} or baseline data missing. Setting energy diff to NaN.")
            avg_energy_diffs_fig2.append(np.nan)

fig2, ax2 = plt.subplots(figsize=(12, 7))
bars_fig2 = ax2.bar(plot_names_fig2, avg_energy_diffs_fig2, color='lightcoral') # Store the bars
ax2.set_ylabel('Avg. Final Energy Diff. from Corresponding ASE Baseline (eV)')
ax2.set_xlabel('Optimization Method')
ax2.set_title('Average Final Energy Difference from ASE Baselines')
ax2.axhline(0, color='black', linewidth=0.8, linestyle='--')

# Add text labels on top of bars for Figure 2
for bar in bars_fig2:
    yval = bar.get_height()
    if not np.isnan(yval): # Only add text if not NaN
        # Adjust text position based on whether the bar is positive or negative
        text_y_offset = 0.001 if yval >= 0 else -0.005 # Small offset for visibility
        va_align = 'bottom' if yval >=0 else 'top'
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + text_y_offset, 
                 f"{yval:.3f}", ha='center', va=va_align, fontsize=8, color='black')

plt.xticks(rotation=45, ha="right")
plt.tight_layout()


# --- Plot 3: Average Mean Displacement from ASE Counterparts ---
disp_plot_data_fig3 = {} 
comparison_pairs_plot3 = [
    ("torch-sim ASE-FIRE (PosOnly)", baseline_ase_pos_only, "TS ASE PosOnly vs ASE Native"),
    ("torch-sim VV-FIRE (PosOnly)", baseline_ase_pos_only, "TS VV PosOnly vs ASE Native"),
    ("torch-sim ASE-FIRE (Frechet Cell)", baseline_ase_frechet, "TS ASE Frechet vs ASE Frechet"),
    ("torch-sim VV-FIRE (Frechet Cell)", baseline_ase_frechet, "TS VV Frechet vs ASE Frechet"),
]

for ts_method_name, ase_method_name, plot_label in comparison_pairs_plot3:
    if ts_method_name in results_all and ase_method_name in results_all:
        state1_data = results_all[ts_method_name]
        state2_data = results_all[ase_method_name]

        if state1_data["final_state"] is None or state2_data["final_state"] is None:
            print(f"Plot3: Skipping displacement for {plot_label} due to missing state data.")
            disp_plot_data_fig3[plot_label] = np.nan
            continue
        
        state1_list = state1_data["final_state"].split()
        state2_list = state2_data["final_state"].split()
        
        if len(state1_list) != len(state2_list):
            print(f"Plot3: Structure count mismatch for {plot_label}.")
            disp_plot_data_fig3[plot_label] = np.nan
            continue

        mean_displacements_current_pair = []
        for s1, s2 in zip(state1_list, state2_list, strict=True):
            if s1.n_atoms == 0 or s2.n_atoms == 0 or s1.n_atoms != s2.n_atoms:
                mean_displacements_current_pair.append(np.nan)
                continue
            pos1_centered = s1.positions - s1.positions.mean(dim=0, keepdim=True)
            pos2_centered = s2.positions - s2.positions.mean(dim=0, keepdim=True)
            displacement = torch.norm(pos1_centered - pos2_centered, dim=1)
            mean_disp = torch.mean(displacement).item()
            mean_displacements_current_pair.append(mean_disp)
        
        if mean_displacements_current_pair:
            avg_disp = np.nanmean(mean_displacements_current_pair)
            disp_plot_data_fig3[plot_label] = avg_disp
        else:
            disp_plot_data_fig3[plot_label] = np.nan
    else:
        print(f"Plot3: Missing data for {ts_method_name} or {ase_method_name}.")
        disp_plot_data_fig3[plot_label] = np.nan


if disp_plot_data_fig3:
    disp_methods_fig3 = list(disp_plot_data_fig3.keys())
    disp_values_fig3 = list(disp_plot_data_fig3.values())

    fig3, ax3 = plt.subplots(figsize=(12, 8))
    ax3.bar(disp_methods_fig3, disp_values_fig3, color='mediumseagreen')
    ax3.set_ylabel('Avg. Mean Atomic Displacement (Å) to ASE Counterpart')
    ax3.set_xlabel('Comparison Pair')
    ax3.set_title('Mean Displacement of Torch-Sim Methods to ASE Counterparts')
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
else:
    print("No displacement data to plot for Figure 3.")

plt.show()
