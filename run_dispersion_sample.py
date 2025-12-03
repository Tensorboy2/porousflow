import time
from src.porousflow.dispersion import run_dispersion_sim_physical
import numpy as np
import os
import sys
import zarr

def load_media_samples(file_path,index):
    root = zarr.open(file_path,mode='r')
    filled_images_ds = root['filled_images']['filled_images']
    solid = filled_images_ds[index]
    return solid

def load_lbm_results(lbm_path, index):
    # data = np.load(os.path.join(lbm_path, f'simulation_result_{index}.npz'))
    root = zarr.open(lbm_path,mode='r')
    ux = root['lbm_results']['ux_physical'][index]
    uy = root['lbm_results']['uy_physical'][index]
    return ux, uy

def save_simulation_results(save_path, sample_index, results, Pe_index):
    root = zarr.open(save_path,mode='a')
    # for key in results.keys():
    root['dispersion_results']['Dx'][sample_index][Pe_index] = results['Dx']
    root['dispersion_results']['Dy'][sample_index][Pe_index] = results['Dy']


def worker(index, save_path, solid, ux, uy, Pe, Pe_index):

    dx = 1
    L = solid.shape[0]

    ux_mean = np.mean(ux[~solid])
    uy_mean = np.mean(uy[~solid])

    Pes = np.array([0,10,50,100,500])
    alpha = Pe / L
    bx=np.max(ux)*alpha
    by=np.max(uy)*alpha
    a = 0.2
    if Pe_index == 0:
        dtx = a**2/2
        dty = a**2/2
    else:
        dtx = ((-1+np.sqrt(1+2*a*bx))**2)/(2*bx**2)
        dty = ((-1+np.sqrt(1+2*a*by))**2)/(2*by**2)

    D_m_x = 1
    D_m_y = 1
    steps = int(100 * L**2)
    # dt = 1e-3


    # for alpha, Pe in zip(alphas, Pes):
    start = time.time()

    Dx = run_dispersion_sim_physical(
        solid=solid,
        velocity=ux / ux_mean,
        steps=steps,
        num_particles=1_000,
        velocity_strength=alpha,
        dt=dtx,
        D=D_m_x,
        dx=dx
    )

    Dy = run_dispersion_sim_physical(
        solid=solid,
        velocity=uy / uy_mean,
        steps=steps,
        num_particles=1_000,
        velocity_strength=alpha,
        dt=dty,
        D=D_m_y,
        dx=dx
    )

    results = {'Dx': Dx, 'Dy': Dy}

    end = time.time()
    print(
        f"[Sample {index}, Pe {Pe}], Finished in {end-start:.2f}s\n"
        f"     Dx: Dx_xx{Dx[0,0]:.3e}, Dx_xy{Dx[0,1]:.3e}, Dx_yx{Dx[1,0]:.3e}, Dx_yy{Dx[1,1]:.3e}\n"
        f"     Dy: Dy_xx{Dy[0,0]:.3e}, Dy_xy{Dy[0,1]:.3e}, Dy_yx{Dy[1,0]:.3e}, Dy_yy{Dy[1,1]:.3e}\n"
    )

    save_simulation_results(save_path, index, results, Pe_index)
    # print(f"[Sample {index}] saved to: {save_path}/simulation_result_{index}.npz")


def main():
    if len(sys.argv) != 4:
        print("Usage: python run_dispersion_sample.py <index> <data_type>")
        sys.exit(1)

    index = int(sys.argv[1])
    data_type = sys.argv[2]
    Pe_index = int(sys.argv[3])

    Pes = [0,10,50,100,500]
    Pe = Pes[Pe_index]

    root_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(root_path, f"data/{data_type}.zarr")
    lbm_path = os.path.join(root_path, f"data/{data_type}.zarr")
    save_path = os.path.join(root_path, f"data/{data_type}.zarr")

    solid = load_media_samples(data_path,index)

    ux, uy = load_lbm_results(lbm_path, index)
    worker(index, save_path, solid, ux, uy, Pe, Pe_index)

if __name__ == "__main__":
    main()
