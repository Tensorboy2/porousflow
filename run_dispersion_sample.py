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
    K = root['lbm_results']['K'][index]
    nu = 1e-6
    return ux, uy, K, nu

def save_simulation_results(save_path, sample_index, results, Pe_index):
    root = zarr.open(save_path,mode='a')
    # for key in results.keys():
    root['dispersion_results']['Dx'][sample_index,Pe_index] = results['Dx']
    root['dispersion_results']['Dy'][sample_index,Pe_index] = results['Dy']


def worker_old(index, save_path, solid, ux, uy, Pe, Pe_index):

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
        dtx = ((-1-np.sqrt(1+2*a*bx))**2)/(2*bx**2)
        dty = ((-1-np.sqrt(1+2*a*by))**2)/(2*by**2)

    D_m_x = 1
    D_m_y = 1
    steps = int(100 * L**2)
    # dt = 1e-3
    tx= np.arange(steps)*dtx
    ty= np.arange(steps)*dty


    # for alpha, Pe in zip(alphas, Pes):
    start = time.time()

    Dx = np.zeros((2,2))
    Dy = np.zeros((2,2))
    Mx = run_dispersion_sim_physical(
        solid=solid,
        velocity=ux / ux_mean,
        steps=steps,
        num_particles=1_000,
        velocity_strength=alpha,
        dt=dtx,
        D=D_m_x,
        dx=dx
    )

    Dx[0,0] = np.mean(Mx[-1000:,0,0]/(2*tx[-1000:]))
    Dx[0,1] = np.mean(Mx[-1000:,0,1]/(2*tx[-1000:]))
    Dx[1,0] = np.mean(Mx[-1000:,1,0]/(2*tx[-1000:]))
    Dx[1,1] = np.mean(Mx[-1000:,1,1]/(2*tx[-1000:]))

    My = run_dispersion_sim_physical(
        solid=solid,
        velocity=uy / uy_mean,
        steps=steps,
        num_particles=1_000,
        velocity_strength=alpha,
        dt=dty,
        D=D_m_y,
        dx=dx
    )
    Dy[0,0] = np.mean(My[-1000:,0,0]/(2*ty[-1000:]))
    Dy[0,1] = np.mean(My[-1000:,0,1]/(2*ty[-1000:]))
    Dy[1,0] = np.mean(My[-1000:,1,0]/(2*ty[-1000:]))
    Dy[1,1] = np.mean(My[-1000:,1,1]/(2*ty[-1000:]))

    results = {'Dx': Dx, 'Dy': Dy}

    end = time.time()

    save_simulation_results(save_path, index, results, Pe_index)
    print(
        f"[Sample {index}, Pe {Pe}], Finished in {end-start:.2f}s, saved to: {save_path}\n"
        f"     Dx: Dx_xx{Dx[0,0]:.3e}, Dx_xy{Dx[0,1]:.3e}, Dx_yx{Dx[1,0]:.3e}, Dx_yy{Dx[1,1]:.3e}\n"
        f"     Dy: Dy_xx{Dy[0,0]:.3e}, Dy_xy{Dy[0,1]:.3e}, Dy_yx{Dy[1,0]:.3e}, Dy_yy{Dy[1,1]:.3e}\n"
    )
    # print(f"[Sample {index}] saved to: {save_path}/simulation_result_{index}.npz")

def worker(index, save_path, solid, ux, uy, Pe, Pe_index, K, nu):
    '''
    This function preforms the dispersion simulation for a given Pe.
    It stores the resulting tensor in the Zarr object.
    
    :param index: Sample index
    :param save_path: Path to Zarr object
    :param solid: Porous medium geometry
    :param ux: Fluid velocity field pointing in the x directi9on
    :param uy: Fluid velocity field pointing in the y direction
    :param Pe: Pecle number 
    :param Pe_index: Index of current Pe
    '''

    dx = 1.0
    L = solid.shape[0]
    alpha = 0.1
    steps = int(50 * L**2)

    # align velicity fields along principal directions
    A = np.linalg.inv(K)*nu
    alpha_x, beta_x = A[0,0], A[0,1]
    alpha_y, beta_y = A[1,0], A[1,1]

    u_x_aligned = alpha_x * ux + beta_x * uy
    u_y_aligned = alpha_y * ux + beta_y * uy


    # --- 1. Normalize velocity fields ---
    fluid_mask = ~solid
    ux_mean = np.mean(np.abs(ux[fluid_mask]))
    uy_mean = np.mean(np.abs(uy[fluid_mask]))

    ux_norm = u_x_aligned#ux / ux_mean
    uy_norm = u_y_aligned#uy / uy_mean

    ux_max = np.max(np.abs(ux_norm))
    uy_max = np.max(np.abs(uy_norm))

    # --- 2. Molecular diffusivity ---
    # Pe = (mean_u * L) / D_m, and mean_u=1 after normalization
    D_m = L / Pe

    # --- 3. Time step calculation ---
    def compute_dt(umax):
        dt_diff = alpha * (dx**2) / (4 * D_m)
        if Pe <= 0.1:
            return dt_diff
        dt_adv = alpha * dx / umax
        return min(dt_diff, dt_adv)

    dt_x = compute_dt(ux_max)
    dt_y = compute_dt(uy_max)

    tx = np.arange(steps) * dt_x
    ty = np.arange(steps) * dt_y

    start = time.time()

    # --- 4. Simulations ---
    def run_and_extract(u_norm, dt, t_arr):
        M = run_dispersion_sim_physical(
            solid=solid,
            velocity=u_norm,
            steps=steps,
            num_particles=1000,
            velocity_strength=1.0,
            dt=dt,
            D=D_m,
            dx=dx,
        )
        # Extract tensor from last 1000 steps
        M_tail = M[-1000:]
        t_tail = t_arr[-1000:, None, None]
        return np.mean(M_tail / (2 * t_tail), axis=0)

    Dx = run_and_extract(ux_norm, dt_x, tx)
    Dy = run_and_extract(uy_norm, dt_y, ty)

    Dx_norm = Dx / D_m
    Dy_norm = Dy / D_m

    results = {
        "Dx": Dx_norm,
        "Dy": Dy_norm,
        "D_m_x": D_m,
        "D_m_y": D_m,
    }

    end = time.time()
    save_simulation_results(save_path, index, results, Pe_index)

    print(
        f"[Sample {index}, Pe {Pe}] Finished in {end-start:.2f}s\n"
        f"    D_m: {D_m:.3e}, dt_x={dt_x:.3e}, dt_y={dt_y:.3e}\n"
        f"    Dx/D_m: {Dx_norm[0,0]:.3f}, {Dx_norm[0,1]:.3f}, {Dx_norm[1,0]:.3f}, {Dx_norm[1,1]:.3f}\n"
        f"    Dy/D_m: {Dy_norm[0,0]:.3f}, {Dy_norm[0,1]:.3f}, {Dy_norm[1,0]:.3f}, {Dy_norm[1,1]:.3f}"
    )


def main():
    if len(sys.argv) != 4:
        print("Usage: python run_dispersion_sample.py <index> <data_type>")
        sys.exit(1)

    index = int(sys.argv[1])
    data_type = sys.argv[2]
    Pe_index = int(sys.argv[3])

    Pes = [0.1,10,50,100,500]
    Pe = Pes[Pe_index]

    root_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(root_path, f"data/{data_type}.zarr")
    lbm_path = os.path.join(root_path, f"data/{data_type}.zarr")
    save_path = os.path.join(root_path, f"data/{data_type}.zarr")

    solid = load_media_samples(data_path,index)

    ux, uy, K, nu = load_lbm_results(lbm_path, index)
    worker(index, save_path, solid, ux, uy, Pe, Pe_index, K, nu)

if __name__ == "__main__":
    main()
