'''
dispersion.py

Module for simulating dispersion in porous flow.
'''
import numpy as np
from numba import njit



@njit(fastmath=True)
def find_most_central_fluid_point(solid):
    """
    Finds the most central fluid point in the solid structure.
    """
    Nx, Ny = solid.shape
    center_x, center_y = Nx // 2, Ny // 2
    min_distance = float('inf')
    closest_point = None
    
    for x in range(Nx):
        for y in range(Ny):
            if not solid[x, y]:  # Check if it's fluid
                distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                if distance < min_distance:
                    min_distance = distance
                    closest_point = np.array([x, y])
    
    return closest_point

@njit(fastmath=True)
def run_dispersion_sim_physical(solid, velocity, steps, num_particles, 
                       velocity_strength=1.0, dt=0.01, D=1.0, dx=1.0):
    Nx, Ny = solid.shape
    Lx, Ly = Nx * dx, Ny * dx  # Physical domain size
    
    # Initialize particles in PHYSICAL coordinates
    center = find_most_central_fluid_point(solid)
    center_phys_x = center[0] * dx
    center_phys_y = center[1] * dx
    
    # Use zeros + loop instead of np.full (numba compatible)
    particles_positions_unwrapped = np.zeros((num_particles, 2), dtype=np.float64)
    particles_positions_wrapped = np.zeros((num_particles, 2), dtype=np.float64)
    
    for i in range(num_particles):
        particles_positions_unwrapped[i, 0] = center_phys_x
        particles_positions_unwrapped[i, 1] = center_phys_y
        particles_positions_wrapped[i, 0] = center_phys_x
        particles_positions_wrapped[i, 1] = center_phys_y
    
    initial_positions = particles_positions_unwrapped.copy()
    
    M_t_all = np.zeros((steps, 2, 2))
    
    for step in range(steps):
        for i in range(num_particles):
            # Positions are in PHYSICAL units (meters)
            x_phys, y_phys = particles_positions_wrapped[i]
            
            # Convert to grid indices for velocity lookup
            ix = int(x_phys / dx) % Nx
            iy = int(y_phys / dx) % Ny
            
            # Diffusion (physical units)
            disp_x = np.sqrt(2 * D * dt) * np.random.normal(0, 1)
            disp_y = np.sqrt(2 * D * dt) * np.random.normal(0, 1)
            
            # Advection (velocity already in physical units)
            vx = velocity[ix, iy, 0] * velocity_strength
            vy = velocity[ix, iy, 1] * velocity_strength
            disp_x += vx * dt
            disp_y += vy * dt
            
            # New position in physical coordinates
            new_x_phys = (x_phys + disp_x) % Lx
            new_y_phys = (y_phys + disp_y) % Ly
            
            # Check collision
            new_ix = int(new_x_phys / dx) % Nx
            new_iy = int(new_y_phys / dx) % Ny
            
            if solid[new_ix, new_iy] == 0:
                particles_positions_wrapped[i, 0] = new_x_phys
                particles_positions_wrapped[i, 1] = new_y_phys
                particles_positions_unwrapped[i, 0] += disp_x
                particles_positions_unwrapped[i, 1] += disp_y
        
        # Dispersion tensor (now in physical units²)
        disp = particles_positions_unwrapped - initial_positions
        mean_x = np.sum(disp[:, 0]) / num_particles
        mean_y = np.sum(disp[:, 1]) / num_particles
        
        fluc_x = disp[:, 0] - mean_x
        fluc_y = disp[:, 1] - mean_y
        
        M_t_all[step, 0, 0] = np.sum(fluc_x * fluc_x) / num_particles
        M_t_all[step, 0, 1] = np.sum(fluc_x * fluc_y) / num_particles
        M_t_all[step, 1, 0] = np.sum(fluc_y * fluc_x) / num_particles
        M_t_all[step, 1, 1] = np.sum(fluc_y * fluc_y) / num_particles

    D = M_t_all[-1] / (2.0 * dt * steps)
    
    return D

if __name__ == "__main__":
    from lbm.lbm import LBM_solver

    # Example usage
    solid = np.zeros((128, 128), dtype=bool)
    solid[:,0] = True
    solid[:,-1] = True

    u_physical, kx_phys, ky_phys, iteration, Ma, Re_lattice, dt, tau, Re_lattice_2 = LBM_solver(
        solid, L_physical=1e-3, max_iterations=1_000)

    u_max = np.max(u_physical[:,:,0])
    dx = 1e-3 / 128  # from L_physical and grid size
    D = 2.023e-9  # physical diffusion coeff [m^2/s]
    target_Pe = 1e-3*u_max/D #1000  # advection-dominated but not purely ballistic
    L = solid.shape[0] * dx
    # D = u_max * L / target_Pe
    num_particles = 10_000
    number_of_steps = 30_000
    dt_diff = dx**2 / (2.0 * D)
    dt_adv  = dx / u_max
    dt = min(dt_diff, dt_adv)
    D = run_dispersion_sim_physical(
        solid, u_physical, steps=40000, num_particles=1000, 
        velocity_strength=1.0, dt=dt, D=D, dx=dx)
    
    D_sim = D[0, 0]  # Longitudinal dispersion coefficient
    
    # print("Dispersion tensor at final step:\n", M_t_all[-1])

    # Theoretical calculation
    L = solid.shape[0] * dx
    g =10.0  # body force in m/s^2
    nu=1e-6  # physical kinematic viscosity in m^2/s
    u_max = g*(L**2)/(8.0*nu) # Poiseuille analytic u-max.
    print("u_max =", u_max)
    D_phys = u_max * L / target_Pe

    Pe = L * u_max / D_phys
    print("target_Pe =", target_Pe)
    print("Calculated Pe =", Pe)
    # Pe = target_Pe / 3.0  # = (2/3 u_max)*(L/2)/D_phys -> simplifies to target_Pe/3

    kappa = 1.0/210.0
    D_eff_theory = D_phys * (1.0 + Pe**2 * kappa)

    # compact formula (same result)
    # D_eff_compact = u_max * L * (1.0/target_Pe + target_Pe / 1890.0)
    # D_laminar = (L/2)**2 * u_max**2 / (48 * D_phys)
    # D_eff = D_phys + 1/210 * (u_max)**2* (128)**2 / D_phys
    D_theory = D_phys + (2/105)* (u_max*2/3)**2 * (L)**2 / D_phys

    print("L =", L)
    print("D_phys =", D_phys)
    print("Pe (used) =", Pe)
    print(f"D_theory ={D_theory:.5e}")
    print(f"D_sim ={D_sim:.5e}")
    # print("D_laminar =", D_laminar)

    # print("D_eff =", D_eff)
    print("relative error theory-sim =", abs(D_theory - D_sim) / D_eff_theory)
    print("D_eff/D_phys =", D_theory / D_phys)
    t_trans = (L/2)**2 / D_phys
    t_sim = number_of_steps * dt
    print('t_sim / t_trans =', t_sim / t_trans)