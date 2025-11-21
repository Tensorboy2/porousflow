from src.porousflow.lbm.lbm import LBM_solver
import numpy as np
import os
import sys

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

def worker(index, save_path, solid):
    import time
    start = time.time()
    L_physical = 1e-3
    tau=0.6
    force_scaling=1e-1
    # Run LBM in x and y directions
    ux_physical, kxx_phys, kxy_phys, iter_x, Ma_x, Re_phys_x, dt_x, tau_x, Re_lattice_x, dx_x, Fx = LBM_solver(
        solid, force_dir=0, L_physical=L_physical, tau=tau, force_strength=force_scaling)

    uy_physical, kyx_phys, kyy_phys, iter_y, Ma_y, Re_phys_y, dt_y, tau_y, Re_lattice_y, dx_y, Fy = LBM_solver(
        solid, force_dir=1, L_physical=L_physical, tau=tau, force_strength=force_scaling)

    K = np.array([[kxx_phys, kxy_phys],
                  [kyx_phys, kyy_phys]])

    results = {
        'ux_physical': ux_physical,
        'uy_physical': uy_physical,
        'K': K,
        'iteration_x': iter_x,
        'iteration_y': iter_y,
        'Ma_x': Ma_x,
        'Ma_y': Ma_y,
        'Re_lattice_x': Re_lattice_x,
        'Re_lattice_y': Re_lattice_y,
        'dt': dt_y,
        'tau': tau_y,
        'Re_phys_x': Re_phys_x,
        'Re_phys_y': Re_phys_y,
        'dx': dx_y,
        'F_lattice': Fy,
    }

    save_simulation_results(save_path, index, results)

    end = time.time()
    print(
        f"Finished sample [{index}] in {end - start:.2f}s\n"
        f"  Output path: {save_path}/simulation_result_{index}.npz\n"
        f"  Iterations:  x={iter_x},  y={iter_y}\n"
        f"  Mach:        x={Ma_x:.5e}, y={Ma_y:.5e}\n"
        f"  Re_phys:        x={Re_phys_x:.5e}, y={Re_phys_y:.5e}\n"
        f"  Re_lattice:        x={Re_lattice_x:.5e}, y={Re_lattice_y:.5e}\n"
        f"  Permeability tensor K: {kxx_phys:.3e}, {kxy_phys:.3e}; {kyx_phys:.3e}, {kyy_phys:.3e}\n"
    )

def main():
    if len(sys.argv) != 3:
        print("Usage: python run_lbm_sample.py <index> <data_type>")
        sys.exit(1)

    index = int(sys.argv[1])
    data_type = sys.argv[2]

    root_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(root_path, "data", f"media_samples_{data_type}", "media_samples.npz")
    save_path = os.path.join(root_path, "data", f"lbm_simulation_results_{data_type}")

    images, filled_images = load_media_samples(data_path)

    if check_if_file_exists(save_path, index):
        print(f"[Sample {index}] Already exists. Skipping.")
        return

    solid = filled_images[index]
    worker(index, save_path, solid)

if __name__ == "__main__":
    main()
