"""
lbm_optimized.py

Highly optimized LBM simulation for 2D Poiseuille flow.
Key optimizations:
- Array layout (9, Nx, Ny) for cache efficiency
- Parallel execution with prange
- Pre-computed constants
- Minimized branching in hot loops
- Efficient memory access patterns
"""
import numpy as np
from numba import njit, prange

@njit(fastmath=True, parallel=True)
def LBM_solver(solid, force_dir=0,
            L_physical=1e-3,
            nu=1e-6,
            rho_phys=1e3,
            g=10.0,
            tau=1.5,
            max_iterations=40_000,
            convergence_threshold=1e-6):
    
    Nx, Ny = solid.shape
    
    # Fluid mask
    fluid = ~solid
    
    # D2Q9 - note: keeping as (9, 2) for clarity, accessed frequently
    c = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
                  [1, 1], [-1, 1], [-1, -1], [1, -1]], dtype=np.int32)
    w = np.array([4.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0,
                  1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0], dtype=np.float64)
    bounce_back = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int32)
    
    # Pre-compute constants
    cs = 1.0 / np.sqrt(3.0)
    cs2_inv = 3.0
    cs4_inv = 9.0
    Ma_target = 0.1
    U_lattice = Ma_target * cs
    
    # Parameter mapping
    dx = L_physical / Nx
    U_physical = g * (L_physical**2) / (8.0 * nu)
    tau = 0.55
    nu_lattice = (tau - 0.5) / 3.0
    dt = nu_lattice * dx**2 / nu
    
    L_lattice = float(Nx)
    Re_lattice = U_lattice * L_lattice / nu_lattice
    
    # Force
    f_phys = rho_phys * g
    F_lattice = g * dt * dt / dx
    
    # Pre-compute force-related constants
    inv_tau = 1.0 / tau
    force_factor = 1.0 - 0.5 * inv_tau
    
    # Initialize fields - CHANGED LAYOUT: (9, Nx, Ny)
    f = np.zeros((9, Nx, Ny), dtype=np.float64)
    f_new = np.zeros((9, Nx, Ny), dtype=np.float64)
    rho = np.ones((Nx, Ny), dtype=np.float64)
    u = np.zeros((Nx, Ny, 2), dtype=np.float64)
    
    # Force field
    F = np.zeros((Nx, Ny, 2), dtype=np.float64)
    F[:, :, force_dir] = F_lattice
    F[solid, :] = 0.0  # Zero force in solid
    
    # Initialize to equilibrium
    for x in prange(Nx):
        for y in range(Ny):
            if fluid[x, y]:
                u_sq = 0.0  # Initial velocity is zero
                for i in range(9):
                    f[i, x, y] = w[i] * rho[x, y]
    
    # Main loop variables
    max_u = 0.0
    check_interval = 100
    warmup_iterations = 10000
    
    # Pre-compute weighted constants (must be done manually for each index)
    w_3half_0 = w[0] * 1.5
    w_3half_1 = w[1] * 1.5
    w_3half_2 = w[2] * 1.5
    w_3half_3 = w[3] * 1.5
    w_3half_4 = w[4] * 1.5
    w_3half_5 = w[5] * 1.5
    w_3half_6 = w[6] * 1.5
    w_3half_7 = w[7] * 1.5
    w_3half_8 = w[8] * 1.5
    
    for iteration in range(max_iterations):
        
        # ============================================================
        # COLLISION with Guo forcing
        # ============================================================
        for x in prange(Nx):
            for y in range(Ny):
                if fluid[x, y]:
                    ux = u[x, y, 0]
                    uy = u[x, y, 1]
                    u_sq = ux*ux + uy*uy
                    Fx = F[x, y, 0]
                    Fy = F[x, y, 1]
                    u_dot_F = ux*Fx + uy*Fy
                    rho_xy = rho[x, y]
                    
                    # Unroll for better performance
                    # Direction 0: [0, 0]
                    eu = 0.0
                    c_dot_F = 0.0
                    Fi_i = w[0] * (-cs2_inv * u_dot_F)
                    feq = w[0] * rho_xy * (1.0 - w_3half_0 * u_sq)
                    f[0, x, y] -= (f[0, x, y] - feq) * inv_tau - force_factor * Fi_i
                    
                    # Direction 1: [1, 0]
                    eu = ux
                    c_dot_F = Fx
                    Fi_i = w[1] * (cs2_inv * c_dot_F + cs4_inv * eu * c_dot_F - cs2_inv * u_dot_F)
                    feq = w[1] * rho_xy * (1.0 + cs2_inv*eu + 4.5*eu*eu - w_3half_1*u_sq)
                    f[1, x, y] -= (f[1, x, y] - feq) * inv_tau - force_factor * Fi_i
                    
                    # Direction 2: [0, 1]
                    eu = uy
                    c_dot_F = Fy
                    Fi_i = w[2] * (cs2_inv * c_dot_F + cs4_inv * eu * c_dot_F - cs2_inv * u_dot_F)
                    feq = w[2] * rho_xy * (1.0 + cs2_inv*eu + 4.5*eu*eu - w_3half_2*u_sq)
                    f[2, x, y] -= (f[2, x, y] - feq) * inv_tau - force_factor * Fi_i
                    
                    # Direction 3: [-1, 0]
                    eu = -ux
                    c_dot_F = -Fx
                    Fi_i = w[3] * (cs2_inv * c_dot_F + cs4_inv * eu * c_dot_F - cs2_inv * u_dot_F)
                    feq = w[3] * rho_xy * (1.0 + cs2_inv*eu + 4.5*eu*eu - w_3half_3*u_sq)
                    f[3, x, y] -= (f[3, x, y] - feq) * inv_tau - force_factor * Fi_i
                    
                    # Direction 4: [0, -1]
                    eu = -uy
                    c_dot_F = -Fy
                    Fi_i = w[4] * (cs2_inv * c_dot_F + cs4_inv * eu * c_dot_F - cs2_inv * u_dot_F)
                    feq = w[4] * rho_xy * (1.0 + cs2_inv*eu + 4.5*eu*eu - w_3half_4*u_sq)
                    f[4, x, y] -= (f[4, x, y] - feq) * inv_tau - force_factor * Fi_i
                    
                    # Direction 5: [1, 1]
                    eu = ux + uy
                    c_dot_F = Fx + Fy
                    Fi_i = w[5] * (cs2_inv * c_dot_F + cs4_inv * eu * c_dot_F - cs2_inv * u_dot_F)
                    feq = w[5] * rho_xy * (1.0 + cs2_inv*eu + 4.5*eu*eu - w_3half_5*u_sq)
                    f[5, x, y] -= (f[5, x, y] - feq) * inv_tau - force_factor * Fi_i
                    
                    # Direction 6: [-1, 1]
                    eu = -ux + uy
                    c_dot_F = -Fx + Fy
                    Fi_i = w[6] * (cs2_inv * c_dot_F + cs4_inv * eu * c_dot_F - cs2_inv * u_dot_F)
                    feq = w[6] * rho_xy * (1.0 + cs2_inv*eu + 4.5*eu*eu - w_3half_6*u_sq)
                    f[6, x, y] -= (f[6, x, y] - feq) * inv_tau - force_factor * Fi_i
                    
                    # Direction 7: [-1, -1]
                    eu = -ux - uy
                    c_dot_F = -Fx - Fy
                    Fi_i = w[7] * (cs2_inv * c_dot_F + cs4_inv * eu * c_dot_F - cs2_inv * u_dot_F)
                    feq = w[7] * rho_xy * (1.0 + cs2_inv*eu + 4.5*eu*eu - w_3half_7*u_sq)
                    f[7, x, y] -= (f[7, x, y] - feq) * inv_tau - force_factor * Fi_i
                    
                    # Direction 8: [1, -1]
                    eu = ux - uy
                    c_dot_F = Fx - Fy
                    Fi_i = w[8] * (cs2_inv * c_dot_F + cs4_inv * eu * c_dot_F - cs2_inv * u_dot_F)
                    feq = w[8] * rho_xy * (1.0 + cs2_inv*eu + 4.5*eu*eu - w_3half_8*u_sq)
                    f[8, x, y] -= (f[8, x, y] - feq) * inv_tau - force_factor * Fi_i
        
        # ============================================================
        # STREAMING with bounce-back
        # ============================================================
        f_new.fill(0.0)
        
        for i in range(9):
            ci_x = c[i, 0]
            ci_y = c[i, 1]
            bb_i = bounce_back[i]
            
            for x in prange(Nx):
                for y in range(Ny):
                    if fluid[x, y]:
                        new_x = (x + ci_x) % Nx
                        new_y = (y + ci_y) % Ny
                        
                        if fluid[new_x, new_y]:
                            f_new[i, new_x, new_y] = f[i, x, y]
                        else:
                            # Bounce-back
                            f_new[bb_i, x, y] = f[i, x, y]
        
        # Swap arrays
        f, f_new = f_new, f
        
        # ============================================================
        # MACROSCOPIC quantities with accumulation for convergence
        # ============================================================
        new_max_u_sq = 0.0  # Track u² to avoid sqrt until needed
        
        for x in prange(Nx):
            for y in range(Ny):
                if fluid[x, y]:
                    rho_sum = 0.0
                    ux_m = 0.0
                    uy_m = 0.0
                    
                    # Unrolled summation
                    for i in range(9):
                        fi = f[i, x, y]
                        rho_sum += fi
                        ux_m += fi * c[i, 0]
                        uy_m += fi * c[i, 1]
                    
                    rho[x, y] = rho_sum
                    
                    # Guo correction
                    ux = (ux_m + 0.5 * F[x, y, 0]) / rho_sum
                    uy = (uy_m + 0.5 * F[x, y, 1]) / rho_sum
                    
                    u[x, y, 0] = ux
                    u[x, y, 1] = uy
                    
                    # Track max velocity squared (for convergence)
                    u_sq_local = ux*ux + uy*uy
                    if u_sq_local > new_max_u_sq:
                        new_max_u_sq = u_sq_local
        
        # ============================================================
        # CONVERGENCE CHECK
        # ============================================================
        if iteration > warmup_iterations and iteration % check_interval == 0:
            new_max_u = np.sqrt(new_max_u_sq)
            if max_u > 0.0 and abs(new_max_u - max_u) / max_u < convergence_threshold:
                break
            max_u = new_max_u
    
    # ============================================================
    # Post-processing
    # ============================================================
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
    
    k_x_lattice = (avg_ux_lattice * nu_lattice * avg_rho_lattice) / F_final_avg
    k_y_lattice = (avg_uy_lattice * nu_lattice * avg_rho_lattice) / F_final_avg
    
    kx_phys = k_x_lattice * dx**2
    ky_phys = k_y_lattice * dx**2
    
    u_lattice = np.max(u)
    Ma = u_lattice / cs
    u_phys = u * dx / dt
    
    return u_phys, kx_phys, ky_phys, iteration, Ma, Re_lattice, dt, tau, u_lattice*L_lattice/nu_lattice, dx, F_lattice


if __name__ == "__main__":
    import time
    
    shape = (128, 128)
    solid = np.zeros(shape, dtype=bool)
    solid[:, 0] = True
    solid[:, -1] = True
    L = 1e-3
    
    print("Starting optimized LBM simulation...")
    start = time.time()
    
    u_phys, kx_phys, ky_phys, iteration, Ma, Re_lattice, dt, tau, Re_lattice_2, dx, F_lattice = LBM_solver(
        solid, 
        L_physical=L,
        max_iterations=200_000,
    )
    
    elapsed = time.time() - start
    
    print(f"\n{'='*60}")
    print(f"Simulation completed in {iteration} iterations ({elapsed:.2f} seconds)")
    print(f"Performance: {iteration/elapsed:.1f} iterations/second")
    print(f"{'='*60}")
    print(f"Permeability kx (m^2): {kx_phys:.6e}, ky (m^2): {ky_phys:.6e}")
    print(f"Mach number: {Ma:.4f}, tau: {tau:.4f}, Lattice Reynolds: {Re_lattice_2:.2f}")
    print(f"Re_phys: {u_phys.max() * L / (1e-6):.2f}")
    print(f"U_physical max: {np.max(u_phys):.6e} m/s")
    print(f"dx: {dx:.6e} m, dt: {dt:.6e} s, F_lattice: {F_lattice:.6e}")
    
    H_phys = (shape[1] - 2) * (L / shape[0])
    k_theory = H_phys**2 / 12.0
    
    rel_err_kx = abs(kx_phys - k_theory) / k_theory
    rel_err_ky = abs(ky_phys - k_theory) / k_theory
    
    print(f"\nTheoretical permeability (m^2): {k_theory:.6e}")
    print(f"  kx (sim) = {kx_phys:.6e} m^2   rel error = {rel_err_kx:.6f}")
    print(f"  ky (sim) = {ky_phys:.6e} m^2   rel error = {rel_err_ky:.6f}")