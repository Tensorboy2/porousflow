'''
effective_self_diffusivity.py

Module for measuring the effective self diffusion in arbitrary geometries, 
'''
from src.porousflow.lbm.lbm import LBM_solver
from src.porousflow.dispersion import run_dispersion_sim_self_diffusivity
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
    Pe = 0
    L = solid.shape[0]

    ux_mean = np.mean(ux[~solid])
    uy_mean = np.mean(uy[~solid])
    D_m_x = 1#ux_mean * L / Pe
    D_m_y = 1#uy_mean * L / Pe
    steps = int(100*L**2)
    # dt = 1e-3
    bx=np.max(ux)
    by=np.max(uy)
    a = 0.5
    dtx = ((-1+np.sqrt(1+2*a*bx))**2)/(2*bx**2)
    dty = ((-1+np.sqrt(1+2*a*bx))**2)/(2*by**2)

    # dispersion x
    Mx = run_dispersion_sim_self_diffusivity(
        solid=solid,
        velocity=ux,
        steps=steps,
        num_particles=1_000,
        velocity_strength=0.0,
        dt=dtx,
        D=D_m_x,
        dx=dx
    )

    # dispersion y
    My = run_dispersion_sim_self_diffusivity(
        solid=solid,
        velocity=uy,
        steps=steps,
        num_particles=1_000,
        velocity_strength=0.0,
        dt=dty,
        D=D_m_y,
        dx=dx
    )

    results = {'Mx': Mx, 'My': My}
    save_simulation_results(save_path, index, results)

    end = time.time()
    print(
        f"[Sample {index}] Finished in {end-start:.2f}s\n"
        f"  Output path: {save_path}/simulation_result_{index}.npz\n"
        # f"     Mx: Mx_xx{Mx[0,0]:.3e}, Mx_xy{Mx[0,1]:.3e}, Mx_yx{Mx[1,0]:.3e}, Mx_yy{Mx[1,1]:.3e}\n"
        # f"     My: My_xx{My[0,0]:.3e}, My_xy{My[0,1]:.3e}, My_yx{My[1,0]:.3e}, My_yy{My[1,1]:.3e}\n"
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
    save_path = os.path.join(root_path, f"data/self_diffusivity_results_{data_type}")

    images, filled_images = load_media_samples(data_path)

    if check_if_file_exists(save_path, index):
        print(f"[Sample {index}] Already exists. Skipping.")
        return

    solid = filled_images[index]
    ux, uy = load_lbm_results(lbm_path, index)
    worker(index, save_path, solid, ux, uy)


if __name__ == '__main__':
    main()