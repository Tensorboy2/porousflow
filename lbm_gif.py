"""
lbm.py

LBM simulation for 2D Poiseuille flow with velocity field recording for animation.
"""
import numpy as np
from numba import njit

@njit(fastmath=True)
def LBM_solver(solid,
            force_dir=0,
            L_physical=1e-3,
            nu=1e-6,
            rho_phys=1e3,
            g=10.0,
            tau=0.9,
            max_iterations=100_000,
            convergence_threshold=1e-8,
            force_strength=1e-5,
            record_interval=100):  # New parameter: record every N iterations
    """
    LBM solver that returns velocity field snapshots for animation.
    
    Returns:
        u_phys: final velocity field
        kx_phys, ky_phys: permeabilities
        iteration: number of iterations
        Ma, Re_phys, dt, tau, Re_lattice, dx, F_lattice: simulation parameters
        u_history: list of velocity field snapshots (lattice units)
        record_iterations: list of iteration numbers for each snapshot
    """
    Nx, Ny = solid.shape

    # fluid mask
    fluid = np.empty((Nx, Ny), dtype=np.bool_)
    for x in range(Nx):
        for y in range(Ny):
            fluid[x, y] = not solid[x, y]

    # D2Q9
    c = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
                  [1, 1], [-1, 1], [-1, -1], [1, -1]], dtype=np.int32)
    w = np.array([4.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0,
                  1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0], dtype=np.float64)
    bounce_back = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int32)

    # parameter mapping
    cs = 1.0 / np.sqrt(3.0)
    dx = L_physical / Nx
    L_lattice = Ny
    nu_lattice = (tau - 0.5) / 3.0
    dt = nu_lattice * dx**2 / nu

    # Force: physical -> lattice
    F_lattice = force_strength * g * dt * dt / dx

    # initialize fields
    rho = np.ones((Nx, Ny), dtype=np.float64)
    u = np.zeros((Nx, Ny, 2), dtype=np.float64)

    # force field (lattice units) per cell
    F = np.zeros((Nx, Ny, 2), dtype=np.float64)
    for x in range(Nx):
        for y in range(Ny):
            if fluid[x, y]:
                F[x, y, force_dir] = F_lattice

    # distributions
    f = np.zeros((Nx, Ny, 9), dtype=np.float64)
    for x in range(Nx):
        for y in range(Ny):
            if fluid[x, y]:
                u_sq = u[x, y, 0]**2 + u[x, y, 1]**2
                for i in range(9):
                    eu = u[x, y, 0]*c[i, 0] + u[x, y, 1]*c[i, 1]
                    feq = w[i] * rho[x, y] * (1.0 + 3.0*eu + 4.5*eu*eu - 1.5*u_sq)
                    f[x, y, i] = feq

    f_new = np.zeros((Nx, Ny, 9), dtype=np.float64)

    # Prepare storage for velocity history
    max_records = (max_iterations // record_interval) + 1
    u_history = np.zeros((max_records, Nx, Ny, 2), dtype=np.float64)
    record_iterations = np.zeros(max_records, dtype=np.int32)
    record_count = 0

    max_u = 0.0
    check_interval = 100
    warmup_iterations = 5000

    for iteration in range(max_iterations):
        # COLLISION with Guo forcing
        for x in range(Nx):
            for y in range(Ny):
                if fluid[x, y]:
                    ux = u[x, y, 0]
                    uy = u[x, y, 1]
                    u_sq = ux*ux + uy*uy
                    Fx = F[x, y, 0]
                    Fy = F[x, y, 1]
                    u_dot_F = ux*Fx + uy*Fy

                    for i in range(9):
                        ci_x = c[i, 0]
                        ci_y = c[i, 1]
                        eu = ux*ci_x + uy*ci_y
                        c_dot_F = ci_x*Fx + ci_y*Fy

                        Fi_i = w[i] * (3.0 * c_dot_F + 9.0 * eu * c_dot_F - 3.0 * u_dot_F)
                        feq = w[i] * rho[x, y] * (1.0 + 3.0*eu + 4.5*eu*eu - 1.5*u_sq)
                        f[x, y, i] = f[x, y, i] - (f[x, y, i] - feq) / tau + (1.0 - 0.5/tau) * Fi_i

        # STREAMING
        f_new.fill(0.0)
        for x in range(Nx):
            for y in range(Ny):
                if fluid[x, y]:
                    for i in range(9):
                        new_x = (x + c[i, 0]) % Nx
                        new_y = (y + c[i, 1]) % Ny
                        if fluid[new_x, new_y]:
                            f_new[new_x, new_y, i] = f[x, y, i]
                        else:
                            f_new[x, y, bounce_back[i]] = f[x, y, i]
        f, f_new = f_new, f

        # MACROSCOPIC quantities
        for x in range(Nx):
            for y in range(Ny):
                if fluid[x, y]:
                    rho_sum = 0.0
                    ux_m = 0.0
                    uy_m = 0.0
                    for i in range(9):
                        fi = f[x, y, i]
                        rho_sum += fi
                        ux_m += fi * c[i, 0]
                        uy_m += fi * c[i, 1]
                    if rho_sum != 0.0:
                        rho[x, y] = rho_sum
                        u[x, y, 0] = (ux_m + 0.5 * F[x, y, 0]) / rho_sum
                        u[x, y, 1] = (uy_m + 0.5 * F[x, y, 1]) / rho_sum
                    else:
                        u[x, y, 0] = 0.0
                        u[x, y, 1] = 0.0

        # RECORD velocity field at specified intervals
        if iteration % record_interval == 0:
            for x in range(Nx):
                for y in range(Ny):
                    u_history[record_count, x, y, 0] = u[x, y, 0]
                    u_history[record_count, x, y, 1] = u[x, y, 1]
            record_iterations[record_count] = iteration
            record_count += 1

        # CONVERGENCE CHECK
        if iteration > warmup_iterations and iteration % check_interval == 0:
            new_max_u = 0.0
            for x in range(Nx):
                for y in range(Ny):
                    if fluid[x, y]:
                        u_mag = np.sqrt(u[x, y, 0]**2 + u[x, y, 1]**2)
                        if u_mag > new_max_u:
                            new_max_u = u_mag
            if max_u > 0.0 and abs(new_max_u - max_u) / max_u < convergence_threshold:
                # Record final state
                for x in range(Nx):
                    for y in range(Ny):
                        u_history[record_count, x, y, 0] = u[x, y, 0]
                        u_history[record_count, x, y, 1] = u[x, y, 1]
                record_iterations[record_count] = iteration
                record_count += 1
                break
            max_u = new_max_u

    # Trim arrays to actual number of records
    u_history = u_history[:record_count]
    record_iterations = record_iterations[:record_count]

    # permeability calculation
    sum_ux = 0.0
    sum_uy = 0.0
    fluid_count = 0
    F_final = 0.0
    total_rho = 0.0
    for x in range(Nx):
        for y in range(Ny):
            if fluid[x, y]:
                sum_ux += u[x, y, 0]
                sum_uy += u[x, y, 1]
                total_rho += rho[x, y]
                F_final += F[x, y, force_dir]
                fluid_count += 1

    avg_ux_lattice = sum_ux / fluid_count
    avg_uy_lattice = sum_uy / fluid_count
    avg_rho_lattice = total_rho / fluid_count
    F_final_avg = F_final / fluid_count

    k_x_lattice = (avg_ux_lattice * nu_lattice * avg_rho_lattice) / (F_final_avg)
    k_y_lattice = (avg_uy_lattice * nu_lattice * avg_rho_lattice) / (F_final_avg)

    kx_phys = k_x_lattice * dx**2
    ky_phys = k_y_lattice * dx**2

    u_lattice = np.mean(u)
    Ma = u_lattice / cs
    u_phys = u * dx / dt

    Re_phys = np.mean(u_phys) * L_physical / nu
    Re_lattice = u_lattice * L_lattice / nu_lattice
    
    return u_phys, kx_phys, ky_phys, iteration, Ma, Re_phys, dt, tau, Re_lattice, dx, F_lattice, u_history, record_iterations


def create_velocity_gif(u_history, record_iterations, solid, dx, dt, filename='velocity_evolution.gif', fps=10):
    """
    Create an animated GIF of velocity field evolution.
    
    Parameters:
        u_history: array of velocity snapshots (lattice units)
        record_iterations: array of iteration numbers
        solid: boolean array marking solid cells
        dx: lattice spacing (m)
        dt: time step (s)
        filename: output filename
        fps: frames per second
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    
    # Convert to physical units
    u_history_phys = u_history * dx / dt
    
    # Compute velocity magnitude
    u_mag = np.sqrt(u_history_phys[:, :, :, 0]**2 + u_history_phys[:, :, :, 1]**2)
    
    # Mask solid regions
    u_mag_masked = np.where(solid, np.nan, u_mag[0])
    
    fig, ax = plt.subplots(figsize=(10, 8))
    vmax = np.nanmax(u_mag)
    
    im = ax.imshow(u_mag[0].T, origin='lower', cmap='viridis', vmin=0, vmax=vmax)
    # cbar = plt.colorbar(im, ax=ax, label='Velocity magnitude (m/s)')
    title = ax.set_title(f'Iteration: {record_iterations[0]}')
    ax.axis('off')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    def update(frame):
        im.set_array(u_mag[frame].T)
        title.set_text(f'Iteration: {record_iterations[frame]}')
        return [im,title]
    
    anim = FuncAnimation(fig, update, frames=len(u_history), interval=1000/fps, blit=True)
    anim.save(filename, writer=PillowWriter(fps=fps))
    plt.close()
    print(f"Animation saved as {filename}")


if __name__ == "__main__":
    # shape = (128, 128)
    # solid = np.zeros(shape, dtype=bool)
    # solid[:, 0] = True
    # solid[:, -1] = True
    import zarr
    data = zarr.open('data/train.zarr', mode='r')
    solid = data['filled_images']['filled_images'][0]
    L = 1e-3
    shape = solid.shape
    import time
    start_time = time.time()
    
    # Run simulation with recording
    result = LBM_solver(solid, 
                       L_physical=L,
                       max_iterations=1000,
                       tau=0.6,
                       record_interval=10)  # Record every 100 iterations
    
    u_phys, kxx_phys, kxy_phys, iteration, Ma, Re_phys, dt, tau, Re_lattice_2, dx, F_lattice, u_history, record_iterations = result
    
    result_y = LBM_solver(solid, 
                       L_physical=L,
                       max_iterations=1000,
                       tau=0.6,
                       record_interval=10,
                       force_dir=1)  # Record every 100 iterations
    
    u_phys, kyx_phys, kyy_phys, iteration, Ma, Re_phys, dt, tau, Re_lattice_2, dx, F_lattice, u_history_y, record_iterations_y = result_y
    
    print("Elapsed time: {:.2f} seconds".format(time.time() - start_time))
    print("Simulation completed in {} iterations".format(iteration))
    print("Recorded {} velocity snapshots".format(len(u_history)))
    print("Permeability kxx (m^2): {:.6e}, kxy (m^2): {:.6e}".format(kxx_phys, kxy_phys))
    print("Permeability kyx (m^2): {:.6e}, kyy (m^2): {:.6e}".format(kyx_phys, kyy_phys))
    print("Mach number: {:.4e}, tau: {:.4f} Lattice Reynolds number: {:.3e}".format(Ma, tau, Re_lattice_2))
    print("Re_phys {:.3e}".format(Re_phys))
    print(f"U_physical max: {np.max(u_phys)} m/s")
    print(f"dx: {dx} m, dt: {dt} s, F_lattice: {F_lattice}")

    # H_phys = (shape[1] - 2) * (L / shape[0])
    # k_theory = H_phys**2 / 12.0
    # rel_err_kx = abs(kx_phys - k_theory) / k_theory
    # rel_err_ky = abs(ky_phys - k_theory) / k_theory

    # print(f"Theoretical permeability (m^2): {k_theory:.6e}")
    # print(f"  kx (sim) = {kx_phys:.6e} m^2   rel error = {rel_err_kx:.6f}")
    # print(f"  ky (sim) = {ky_phys:.6e} m^2   rel error = {rel_err_ky:.6f}")
    
    # Create the GIF
    print("\nCreating animation...")
    create_velocity_gif(u_history, record_iterations, solid, dx, dt, 
                       filename='porelab_junior_plots/velocity_evolution_x.gif', fps=10)
    create_velocity_gif(u_history_y, record_iterations_y, solid, dx, dt, 
                       filename='porelab_junior_plots/velocity_evolution_y.gif', fps=10)