from src.porousflow.lbm.lbm import LBM_solver
import numpy as np
import os
from mpi4py import MPI

def load_media_samples(file_path):
    data = np.load(file_path)
    images = data['images']
    filled_images = data['filled_images']
    return images, filled_images

def save_simulation_results(save_path, sample_index, results):
    os.makedirs(save_path, exist_ok=True)
    np.savez_compressed(os.path.join(save_path, f'simulation_result_{sample_index}.npz'), **results)

def check_if_file_exists(save_path, sample_index):
    return os.path.exists(os.path.join(save_path, f'simulation_result_{sample_index}.npz'))

def worker(index, save_path, solid, rank):
    start = MPI.Wtime()

    ux_physical, kxx_phys, kxy_phys, iter_x, Ma_x, Re_x, dt_x, tau_x, Re2_x, dx_x, Fx = LBM_solver(
        solid, force_dir=0, L_physical=1e-3)

    uy_physical, kyx_phys, kyy_phys, iter_y, Ma_y, Re_y, dt_y, tau_y, Re2_y, dx_y, Fy = LBM_solver(
        solid, force_dir=1, L_physical=1e-3)

    end = MPI.Wtime()

    K = np.array([[kxx_phys, kxy_phys],
                  [kyx_phys, kyy_phys]])

    results = {
        'u_physical': ux_physical,
        'uy_physical': uy_physical,
        'K': K,
        'iteration_x': iter_x,
        'iteration_y': iter_y,
        'Ma_x': Ma_x,
        'Ma_y': Ma_y,
        'Re_lattice': Re_y,
        'dt': dt_y,
        'tau': tau_y,
        'Re_lattice_2': Re2_y,
        'dx': dx_y,
        'F_lattice': Fy,
    }

    save_simulation_results(save_path, index, results)

    print(
        f"[Rank {rank}] Finished sample {index} in {end - start:.2f}s\n"
        f"  Output path: {save_path}/simulation_result_{index}.npz\n"
        f"  Iterations:  x={iter_x},  y={iter_y}\n"
        f"  Mach:        x={Ma_x:.3f}, y={Ma_y:.3f}\n"
        f"  Permeability tensor K: {kxx_phys:.3e}, {kxy_phys:.3e}; {kyx_phys:.3e}, {kyy_phys:.3e}\n"
    )

def main(data_type):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    root_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(root_path, "data", f"media_samples_{data_type}", "media_samples.npz")
    save_path = os.path.join(root_path, "data", "lbm_simulation_results_"+data_type)

    # Step 1: Rank 0 loads data and determines remaining work
    if rank == 0:
        print(f"Starting LBM simulations for {data_type} data...")
        start_time = MPI.Wtime()
        images, filled_images = load_media_samples(data_path)
        num_samples = images.shape[0]

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

    # Step 2: Broadcast data and remaining index list
    images = comm.bcast(images, root=0)
    filled_images = comm.bcast(filled_images, root=0)
    indices_to_simulate = comm.bcast(indices_to_simulate, root=0)

    # Step 3: Even static division of remaining tasks
    for local_i in range(rank, len(indices_to_simulate), size):
        index = indices_to_simulate[local_i]
        solid = filled_images[index]
        worker(index, save_path, solid, rank)

    comm.Barrier()
    if rank == 0:
        end_time = MPI.Wtime()
        print("All simulations completed.")
        print(f"Total elapsed time for all simulations: {end_time - start_time:.2f}s")

if __name__ == "__main__":

    main('train')
    main('validation')
    main('test')
