from src.porousflow.dispersion import run_dispersion_sim_physical
import numpy as np
import os
import sys

def load_media_samples(image_path):
    data = np.load(image_path)
    images = data['images']
    filled_images = data['filled_images']
    return images, filled_images

def load_lbm_results(lbm_path, index):
    data = np.load(os.path.join(lbm_path, f'simulation_result_{index}.npz'))
    ux = data['ux_physical']
    uy = data['uy_physical']
    return ux, uy

def save_simulation_results(save_path, sample_index, results):
    os.makedirs(save_path, exist_ok=True)
    np.savez_compressed(os.path.join(save_path, f'simulation_result_{sample_index}.npz'), **results)

def check_if_file_exists(save_path, sample_index):
    return os.path.exists(os.path.join(save_path, f'simulation_result_{sample_index}.npz'))

def worker(index, save_path, solid, ux, uy):
    import time
    start = time.time()

    dx = 1
    Pe = 200
    L = solid.shape[0]

    ux_mean = np.mean(ux[~solid])
    uy_mean = np.mean(uy[~solid])
    D_m_x = ux_mean * L / Pe
    D_m_y = uy_mean * L / Pe
    steps = int(1e5)
    dt = 2e-2

    # dispersion x
    Dx = run_dispersion_sim_physical(
        solid=solid,
        velocity=ux,
        steps=steps,
        num_particles=1_000,
        velocity_strength=1.0,
        dt=dt,
        D=D_m_x,
        dx=dx
    )

    # dispersion y
    Dy = run_dispersion_sim_physical(
        solid=solid,
        velocity=uy,
        steps=steps,
        num_particles=1_000,
        velocity_strength=1.0,
        dt=dt,
        D=D_m_y,
        dx=dx
    )

    results = {'Dx': Dx, 'Dy': Dy}
    save_simulation_results(save_path, index, results)

    end = time.time()
    print(
        f"[Sample {index}] Finished in {end-start:.2f}s\n"
        f"  Output path: {save_path}/simulation_result_{index}.npz\n"
        f"     Dx: Dx_xx{Dx[0,0]:.3e}, Dx_xy{Dx[0,1]:.3e}, Dx_yx{Dx[1,0]:.3e}, Dx_yy{Dx[1,1]:.3e}\n"
        f"     Dy: Dy_xx{Dy[0,0]:.3e}, Dy_xy{Dy[0,1]:.3e}, Dy_yx{Dy[1,0]:.3e}, Dy_yy{Dy[1,1]:.3e}\n"
    )

def main():
    if len(sys.argv) != 3:
        print("Usage: python run_dispersion_sample.py <index> <data_type>")
        sys.exit(1)

    index = int(sys.argv[1])
    data_type = sys.argv[2]

    root_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(root_path, f"data/media_samples_{data_type}/media_samples.npz")
    lbm_path = os.path.join(root_path, f"data/lbm_simulation_results_{data_type}")
    save_path = os.path.join(root_path, f"data/dispersion_simulation_results_{data_type}")

    images, filled_images = load_media_samples(data_path)

    if check_if_file_exists(save_path, index):
        print(f"[Sample {index}] Already exists. Skipping.")
        return

    solid = filled_images[index]
    ux, uy = load_lbm_results(lbm_path, index)
    worker(index, save_path, solid, ux, uy)

if __name__ == "__main__":
    main()
