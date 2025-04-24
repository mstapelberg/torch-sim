import torch 
import torch_sim as ts
import matplotlib.pyplot as plt
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
mace_potential = torch.load("../forge/scratch/potentials/mace_gen_7_ensemble/job_gen_7-2025-04-14_model_0_pr_stagetwo.model", map_location=torch_sim_device)

# Create the torch_sim wrapper
ts_mace_model = MaceModel(
    model=mace_potential,
    device=torch_sim_device,
    dtype=torch_sim_dtype,
    compute_forces=True, # Default, but good to be explicit
    compute_stress=True, # Needed by interface if we want stress later
)

initial_trajectory = read('../forge/scratch/data/neb_workflow_data/Cr7Ti8V104W8Zr_Cr_to_V_site102_to_69_initial.xyz', index=':')

print(len(initial_trajectory))

initial_system = ts.io.atoms_to_state(initial_trajectory[0], device=torch_sim_device, dtype=torch_sim_dtype)
final_system = ts.io.atoms_to_state(initial_trajectory[-1], device=torch_sim_device, dtype=torch_sim_dtype)

neb_workflow = NEB(
    model=ts_mace_model,
    device=torch_sim_device,
    dtype=torch_sim_dtype,
    spring_constant=5.0,
    n_images=5,
    use_climbing_image=True, # Turn climbing off for initial GD test
    optimizer_type="fire", # Select Gradient Descent
    #optimizer_params={"constant_volume": True},
    #gd_lr=0.01, # Start with a small learning rate
    trajectory_filename="neb_path_gd.hdf5"
)

final_path_gd = neb_workflow.run(
    initial_system=initial_system,
    final_system=final_system,
    max_steps=300, # Allow enough steps for potentially slow GD
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

energies = results['energy']

print(energies.tolist())
barrier = max(energies) - energies[0]
plt.plot(results['energy'])
plt.xlabel('Image Coordinate')
plt.ylabel('Energy (eV)')
plt.title(f'Barrier = {barrier:.4f} eV')
plt.show()