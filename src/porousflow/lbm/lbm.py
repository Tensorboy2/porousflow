"""
lbm.py

LBM simulation for 2D Poiseuille flow using analytical parameter mapping.
"""
import numpy as np
from numba import njit

@njit(fastmath=True)
def LBM_solver(solid, force_dir=0,
            L_physical=1e-3,  # channel width w [m]
            nu=1e-6,          # kinematic viscosity [m^2/s]
            rho_phys=1e3,     # density [kg/m^3]
            g=10.0,      # gravity [m/s^2]
            tau=1.5,         # relaxation parameter 
            max_iterations=200_000,
            convergence_threshold=1e-8):
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

    # --------------------------------------------------------------------
    # parameter mapping
    # --------------------------------------------------------------------
    cs = 1.0 / np.sqrt(3.0)
    Ma_target = 0.1
    Re_target = 19.7

    U_lattice = Ma_target * cs
    Ma = Ma_target

    dx = L_physical / Nx
    U_physical = g*(L_physical**2)/(8.0*nu) # Poiseuille analytic u-max.  Re_target * nu / L_physical
    tau = 1.9
    nu_lattice = (tau - 0.5) / 3.0
    dt = nu_lattice * dx**2 / nu#U_lattice * dx / U_physical

    # nu_lattice = nu * dt / (dx * dx)
    # tau = (3.0 * nu_lattice + 0.5)
    L_lattice = float(Nx)
    Re_lattice = U_lattice * L_lattice / nu_lattice

    # --------------------------------------------------------------------
    # Force: physical -> lattice
    # --------------------------------------------------------------------
    # physical force density f_phys = rho_phys * g  (N/m^3)
    f_phys = rho_phys * g
    # convert to lattice units (you used dt^2/dx; kept same)
    F_lattice = 1e-1*g* dt * dt / dx
    # F_lattice = g * dt / cs**2



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
    # init to equilibrium
    for x in range(Nx):
        for y in range(Ny):
            if fluid[x, y]:
                u_sq = u[x, y, 0]**2 + u[x, y, 1]**2
                for i in range(9):
                    eu = u[x, y, 0]*c[i, 0] + u[x, y, 1]*c[i, 1]
                    feq = w[i] * rho[x, y] * (1.0 + 3.0*eu + 4.5*eu*eu - 1.5*u_sq)
                    f[x, y, i] = feq

    # --------------------------------------------------------------------
    # main loop: collision (with Guo) -> streaming -> macroscopic update
    # --------------------------------------------------------------------
    f_new = np.zeros((Nx, Ny, 9), dtype=np.float64)

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

                        # Guo forcing Fi (for cs^2 = 1/3)
                        # Fi = w_i * [ (e_i·F)/c_s^2 + ( (e_i·u)(e_i·F)/c_s^4 - (u·F)/c_s^2 ) ]
                        # with c_s^2 = 1/3 so 1/c_s^2 = 3, 1/c_s^4 = 9
                        Fi_i = w[i] * ( 3.0 * c_dot_F + 9.0 * eu * c_dot_F - 3.0 * u_dot_F )

                        # equilibrium
                        feq = w[i] * rho[x, y] * (1.0 + 3.0*eu + 4.5*eu*eu - 1.5*u_sq)

                        # collision: relaxation + forcing correction factor (1 - 1/(2*tau))
                        f[x, y, i] = f[x, y, i] - (f[x, y, i] - feq) / tau + (1.0 - 0.5/tau) * Fi_i

        # STREAMING (with bounce-back into solids)
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
                            # bounce-back into current cell (simple BB)
                            f_new[x, y, bounce_back[i]] = f[x, y, i]
        # swap
        f, f_new = f_new, f

        # MACROSCOPIC quantities with forcing correction for momentum:
        # rho = sum_i f_i
        # u = (sum_i f_i e_i + 0.5 * F) / rho
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
                        # add 0.5*F to momentum (Guo correction)
                        u[x, y, 0] = (ux_m + 0.5 * F[x, y, 0]) / rho_sum
                        u[x, y, 1] = (uy_m + 0.5 * F[x, y, 1]) / rho_sum
                    else:
                        u[x, y, 0] = 0.0
                        u[x, y, 1] = 0.0

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
                break
            max_u = new_max_u

    # --------------------------------------------------------------------
    # permeability, convert back to physical units (kept your original code)
    # --------------------------------------------------------------------
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

    u_lattice = np.max(u)
    Ma = u_lattice / cs
    u_phys = u * dx / dt

    return u_phys, kx_phys, ky_phys, iteration, Ma, Re_lattice, dt, tau, u_lattice*L_lattice/nu_lattice, dx , F_lattice


if __name__ == "__main__":
    shape = (128, 128)
    solid = np.zeros(shape, dtype=bool)
    solid[:, 0] = True
    solid[:, -1] = True
    L = 1e-3  # channel width [m]
    import time
    start_time = time.time()
    u_phys, kx_phys, ky_phys, iteration, Ma, Re_lattice, dt, tau, Re_lattice_2, dx , F_lattice = LBM_solver(solid, 
                                                                         L_physical=L,
                                                                         )
    print("Elapsed time: {:.2f} seconds".format(time.time() - start_time))
    print("Simulation completed in {} iterations".format(iteration))
    print("Permeability kx (m^2): {:.6e}, ky (m^2): {:.6e}".format(kx_phys, ky_phys))
    print("Mach number: {:.4f},, tau: {:.4f} Lattice Reynolds number: {:.2f}".format(Ma,tau, Re_lattice_2))
    print("Re_phys {}".format(u_phys.max() * L / (1e-6)))
    print(f"U_physical max: {np.max(u_phys)} m/s")
    print(f"dx: {dx} m, dt: {dt} s, F_lattice: {F_lattice} ")

    H_phys = (shape[1] - 2) * (L / shape[0])

    # Theoretical permeability (m^2) for 2D parallel-plate Poiseuille (per unit depth)
    k_theory = H_phys**2 / 12.0

    # Compare
    rel_err_kx = abs(kx_phys - k_theory) / k_theory
    rel_err_ky = abs(ky_phys - k_theory) / k_theory

    print(f"Theoretical permeability (m^2): {k_theory:.6e}")
    print(f"  kx (sim) = {kx_phys:.6e} m^2   rel error = {rel_err_kx:.6f}")
    print(f"  ky (sim) = {ky_phys:.6e} m^2   rel error = {rel_err_ky:.6f}")