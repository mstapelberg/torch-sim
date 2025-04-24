import torch 
import torch_sim as ts
import matplotlib.pyplot as plt
import numpy as np
from ase.io import read
from torch_sim.models.mace import MaceModel
from torch_sim.workflows.neb import NEB 

# Configure logging to DEBUG level first
import logging
import sys
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(message)s', stream=sys.stdout)
logging.getLogger('torch_sim.workflows.neb').setLevel(logging.DEBUG)


torch_sim_device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch_sim_dtype = torch.float32 # because I wanna go fast

# Load the actual MACE model
mace_potential = torch.load("../../../../forge/scratch/potentials/mace_gen_7_ensemble/job_gen_7-2025-04-14_model_0_pr_stagetwo.model", map_location=torch_sim_device)

# Create the torch_sim wrapper
ts_mace_model = MaceModel(
    model=mace_potential,
    device=torch_sim_device,
    dtype=torch_sim_dtype,
    compute_forces=True, # Default, but good to be explicit
    compute_stress=True, # Needed by interface if we want stress later
)

initial_trajectory = read('../../../../forge/scratch/data/neb_workflow_data/Cr7Ti8V104W8Zr_Cr_to_V_site102_to_69_initial.xyz', index=':')

print(len(initial_trajectory))

initial_system = ts.io.atoms_to_state(initial_trajectory[0], device=torch_sim_device, dtype=torch_sim_dtype)
final_system = ts.io.atoms_to_state(initial_trajectory[-1], device=torch_sim_device, dtype=torch_sim_dtype)

neb_workflow = NEB(
    model=ts_mace_model,
    device=torch_sim_device,
    dtype=torch_sim_dtype,
    spring_constant=0.1,
    n_images=5,
    use_climbing_image=True, # Turn climbing off for initial GD test
    optimizer_type="gd", # Select Gradient Descent
    optimizer_params={"lr": 0.01},
    trajectory_filename="neb_path_gd_5im.hdf5"
)

final_path_gd = neb_workflow.run(
    initial_system=initial_system,
    final_system=final_system,
    max_steps=600, # Allow enough steps for potentially slow GD
    fmax=0.05
)

# Check if it converged and plot results
results = ts_mace_model(
    dict(
        positions=final_path_gd.positions,
        cell=final_path_gd.cell,
        atomic_numbers=final_path_gd.atomic_numbers,
        batch=final_path_gd.batch,
        pbc=True,
    )
)

energies = results['energy'].tolist()

# Including the energies from the ASE NEB calculation for comparison
ase_energies = [0.0, 0.154541015625, 0.6151123046875, 0.8592529296875, 0.8148193359375, 0.5965576171875, 0.47705078125]

scaled_energies = [e - energies[0] for e in energies]

print(scaled_energies)
torch_sim_barrier = max(scaled_energies) - scaled_energies[0]
ase_barrier = max(ase_energies) - ase_energies[0]

# Create normalized reaction coordinates (0 to 1) for both datasets
torch_sim_coords = np.linspace(0, 1, len(scaled_energies))
ase_coords = np.linspace(0, 1, len(ase_energies))

# Create a common x-axis with 100 points for smoother plotting
common_coords = np.linspace(0, 1, 100)

# Interpolate both energy profiles to the common coordinate system
torch_sim_interp = np.interp(common_coords, torch_sim_coords, scaled_energies)
ase_interp = np.interp(common_coords, ase_coords, ase_energies)

plt.plot(common_coords, torch_sim_interp, label='torch-sim')
plt.plot(common_coords, ase_interp, label='ASE')
plt.xlabel('Reaction Coordinate')
plt.ylabel('Energy (eV)')
plt.title(f'ASE Barrier = {ase_barrier:.4f} eV, torch-sim Barrier = {torch_sim_barrier:.4f} eV, Difference = {torch_sim_barrier - ase_barrier:.4f} eV')
plt.legend()
plt.show()