'''
dispersion_simulations.py

script for running dispersion simulations on porous media samples.
'''

from src.porousflow.dispersion import run_dispersion_sim_physical
import os
from mpi4py import MPI
import numpy as np


def load_media_samples(image_path):
    data = np.load(image_path)
    images = data['images']
    filled_images = data['filled_images']
    return images, filled_images

def load_lbm_results(lbm_path, index):
    data = np.load(os.path.join(lbm_path, f'simulation_result_{index}.npz'))
    ux = data['u_physical']
    uy = data['uy_physical']
    return ux, uy

def save_simulation_results(save_path, sample_index, results):
    os.makedirs(save_path, exist_ok=True)
    np.savez_compressed(os.path.join(save_path, f'simulation_result_{sample_index}.npz'), **results)

def check_if_file_exists(save_path, sample_index):
    return os.path.exists(os.path.join(save_path, f'simulation_result_{sample_index}.npz'))

def worker(index, save_path, solid, ux, uy, rank):
    start = MPI.Wtime()

    # dispersion in x direction
    Dx = run_dispersion_sim_physical(
        solid=solid,
        velocity=ux,
        steps=50_000,
        num_particles=10_000,
        velocity_strength=1.0,
        dt=0.01,
        D=1.0,
        dx=1.0
    )

    # dispersion in y direction
    Dy = run_dispersion_sim_physical(
        solid=solid,
        velocity=uy,
        steps=50_000,
        num_particles=10_000,
        velocity_strength=1.0,
        dt=0.01,
        D=1.0,
        dx=1.0
    )

    end = MPI.Wtime()
    results = {
        'Dx': Dx,
        'Dy': Dy
    }
    save_simulation_results(save_path, index, results)
    print(
        f"[Rank {rank}] Finished sample {index} in {end - start:.2f}s\n"
        f"  Output path: {save_path}/simulation_result_{index}.npz\n"
        f"  Dispersion coefficients:\n"
        f"     Dx = {Dx[0,0]:.3e}, {Dx[0,1]:.3e}; {Dx[1,0]:.3e}, {Dx[1,1]:.3e}\n"
        f"     Dy = {Dy[0,0]:.3e}, {Dy[0,1]:.3e}; {Dy[1,0]:.3e}, {Dy[1,1]:.3e}\n"
    )

def main(data_type):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    root_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(root_path, "data", "media_samples_"+data_type, "media_samples.npz")
    lbm_path = os.path.join(root_path, "data", "lbm_simulation_results_"+data_type)
    save_path = os.path.join(root_path, "data", "dispersion_simulation_results_"+data_type)

    # Step 1: Rank 0 loads data and determines remaining work
    if rank == 0:
        print(f'Starting dispersion simulations for {data_type} data...')
        start_time = MPI.Wtime()
        images, filled_images = load_media_samples(data_path)
        num_samples = images.shape[0]

        # Determine which samples need to be processed
        indices_to_simulate = [
            i for i in range(num_samples)
            if not check_if_file_exists(save_path, i)
        ]
        print(f"Total samples: {num_samples}")
        print(f"Samples to simulate: {len(indices_to_simulate)}")
    else:
        images = None
        filled_images = None
        indices_to_simulate = None

    # Broadcast remaining indices to all ranks
    images = comm.bcast(images, root=0)
    filled_images = comm.bcast(filled_images, root=0)
    indices_to_simulate = comm.bcast(indices_to_simulate, root=0)

    # Step 2: Distribute work among ranks
    for local_i in range(rank, len(indices_to_simulate), size):
        index = indices_to_simulate[local_i]
        solid = filled_images[index]
        ux, uy = load_lbm_results(lbm_path, index)
        worker(index, save_path, solid, ux, uy, rank)

    # Finalize MPI
    comm.Barrier()
    if rank == 0:
        end_time = MPI.Wtime()
        print("All simulations completed.")
        print(f"Total elapsed time for all simulations: {end_time - start_time:.2f}s")

if __name__ == "__main__":

    main('train')
    main('validation')
    main('test')