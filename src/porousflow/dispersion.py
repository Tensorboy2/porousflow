'''
dispersion.py

Module for simulating dispersion in porous flow.
'''
import numpy as np
import os 
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
def run_dispersion_sim_physical(solid, velocity, steps=10_000, num_particles=1_000, 
                       velocity_strength=1.0, dt=1e-3, D=1.0, dx=1.0):
    Nx, Ny = solid.shape
    Lx, Ly = Nx * dx, Ny * dx  # Physical domain size
    
    # Initialize particles in PHYSICAL coordinates
    # center = find_most_central_fluid_point(solid)
    center_x, center_y = initial_point(velocity)
    center_phys_x = center_x
    center_phys_y = center_y
    
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
    # inc = 1000
    # positions_for_plot = np.zeros((num_particles, 2, steps//inc), dtype=np.float64)
    # positions_for_plot[:, 0, 0] = particles_positions_wrapped[:, 0]
    # positions_for_plot[:, 1, 0] = particles_positions_wrapped[:, 1]
    
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

            # if step % inc == 0 and step > 0:
            #     positions_for_plot[i, 0, step//inc] = particles_positions_unwrapped[i, 0]
            #     positions_for_plot[i, 1, step//inc] = particles_positions_unwrapped[i, 1]
        
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

        # --- Build time array t safely ---
    # t = np.empty(steps, dtype=np.float64)
    # for i in range(steps):
    #     t[i] = i * dt

    # # --- Compute M_t_all[1:] / (2 * t[1:]) safely ---
    # n_rows = steps - 1  # number of usable rows (skip step 0)
    # M = np.empty((n_rows, 2, 2), dtype=np.float64)

    # for i in range(n_rows):
    #     denom = 2.0 * t[i + 1]
    #     for a in range(2):
    #         for b in range(2):
    #             M[i, a, b] = M_t_all[i + 1, a, b] / denom

    # # --- Mean over last 1000 rows safely ---
    # start = 0
    # if n_rows > 1000:
    #     start = n_rows - 1000
    # count = n_rows - start

    # D = np.zeros((2, 2), dtype=np.float64)
    # for a in range(2):
    #     for b in range(2):
    #         s = 0.0
    #         for i in range(start, n_rows):
    #             s += M[i, a, b]
    #         D[a, b] = s / count

    return M_t_all


@njit(fastmath=True)
def initial_point(u):
    u_max = 0
    x_max, y_max = 0,0
    for x in range(u.shape[0]):
        for y in range(u.shape[1]):
            if u[x,y,0]**2 + u[x,y,1]**2 > u_max:
                u_max = u[x,y,0]**2 + u[x,y,1]**2
                x_max, y_max = x,y
    return x_max,y_max      


@njit(fastmath=True)
def run_dispersion_sim_self_diffusivity(solid, velocity, steps=10_000, num_particles=1_000, 
                       velocity_strength=1.0, dt=1e-3, D=1.0, dx=1.0):
    Nx, Ny = solid.shape
    Lx, Ly = Nx * dx, Ny * dx  # Physical domain size

    center_x, center_y = initial_point(velocity)#np.argmax(velocity)#fluid_points[idx]
    center_phys_x = center_x#[0]
    center_phys_y = center_y#[1]
    
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

    # D = M_t_all[-1] / (2.0 * dt * steps)
    
    return M_t_all

@njit(fastmath=True)
def run_dispersion_sim_physical_test(solid, velocity, steps=10_000, num_particles=1_000, 
                       velocity_strength=1.0, dt=1e-3, D=1.0, dx=1.0):
    Nx, Ny = solid.shape
    Lx, Ly = Nx * dx, Ny * dx  # Physical domain size
    
    # Initialize particles in PHYSICAL coordinates
    # center = find_most_central_fluid_point(solid)
    # fluid_points = np.argwhere(~solid)  # indices of all fluid points

    # index of the max value
    # idx = np.argmax(fluid_values)

    center_x, center_y = initial_point(velocity)#np.argmax(velocity)#fluid_points[idx]
    center_phys_x = center_x#[0]
    center_phys_y = center_y#[1]
    
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
    inc = 100
    positions_for_plot = np.zeros((steps//inc,num_particles, 2), dtype=np.float64)
    positions_for_plot[0,:, 0] = particles_positions_wrapped[:, 0]
    positions_for_plot[0,:, 1] = particles_positions_wrapped[:, 1]
    
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

            if step % inc == 0 and step > 0:
                positions_for_plot[step//inc,i, 0] = particles_positions_wrapped[i, 0]
                positions_for_plot[step//inc,i, 1] = particles_positions_wrapped[i, 1]
        
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
    
    return D, M_t_all, positions_for_plot

if __name__ == "__main__":
    from lbm.lbm import LBM_solver
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation



    # Example usage
    solid = np.zeros((128, 128), dtype=bool)
    solid[0,:] = True
    solid[-1,:] = True

    # u_physical, kx_phys, ky_phys, iteration, Ma, Re_lattice, dt, tau, Re_lattice_2, dx, F = LBM_solver(solid,force_dir=1)


    # u_max = np.max(u_physical[:,:,0])
    u_physical = np.zeros((128,128,2), dtype=np.float64)
    # analytic Poiseuille profile
    for i in range(1, solid.shape[0]-1):
        for j in range(0, solid.shape[1]):
            y = i - 1
            L = solid.shape[0] - 2
            u_physical[i, j, 1] = 1 * 4.0 * y * (L - y) / (L**2)
    # plt.imshow(u_physical[:,:,1], cmap='viridis')
    # plt.colorbar()
    # plt.show()

    u_mean = np.mean(u_physical[:,:,][~solid])
    u = u_physical/u_mean
    u_mean = 1.0
    # L = 1e-3* 126/128
    # dx = L / 128  # from L_physical and grid size

    D_phys = 1e-5#2.023e-9  # physical diffusion coeff [m^2/s]
    target_Pe = 200#1e-3*u_mean/D_phys #1000  # advection-dominated but not purely ballistic
    L = (solid.shape[0]-2)
    D_m = 1#u_mean * L / target_Pe
    print("D_m =", D_m)
    # D = u_max * L / target_Pe
    dx = 1.0 # from L_physical and grid size
    dt_diff = dx**2 / (2.0 * D_m)
    dt_adv  = dx / u_mean
    dt = 1e-3#min(dt_diff, dt_adv)
    b=np.max(u)
    a = 0.5
    dt = ((-1+np.sqrt(1+2*a*b))**2)/(2*b**2)
    print("dt_diff =", dt_diff)
    print("dt_adv =", dt_adv)
    print("dt used =", dt)
    total_steps = int(10*L**2)
    print("steps: ", total_steps)

    D, M, positions_for_plot = run_dispersion_sim_physical_test(
        solid, u, steps=total_steps, num_particles=1_000, 
        velocity_strength=1.0, dt=dt, D=D_m, dx=dx)
    
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    def update(i):
        axes[0].cla()
        axes[1].cla()

        axes[0].hist(positions_for_plot[:, 0, i], bins=50)
        axes[0].set_title(f"X distribution (step={i*1e3:.0f})")
        # axes[0].set_xlim(-10,140)
        axes[0].set_ylim(0,100)

        axes[1].hist(positions_for_plot[:, 1, i], bins=50)
        axes[1].set_title(f"Y distribution (step={i*1e3:.0f})")
        # axes[1].set_xlim(-10,1000)
        axes[1].set_ylim(0,100)

        return axes

    ani = FuncAnimation(fig, update, frames=positions_for_plot.shape[2], interval=24, repeat_delay=1000)
    # ani.save("particle_distribution.gif", writer="imagemagick")
    # plt.show()
    # for i in range(positions_for_plot.shape[2]):
    #     plt.subplot(1,2,1)
    #     plt.hist(positions_for_plot[:,0,i], bins=50)
    #     plt.subplot(1,2,2)
    #     plt.hist(positions_for_plot[:,1,i], bins=50)
    #     plt.close()
    # plt.show()

    plt.figure()
    t = np.arange(total_steps)*dt
    plt.subplot(1,2,1)
    plt.plot(t,M[:,0,0], label='M_xx')
    plt.plot(t,np.ones_like(t)*L**2/12, label='L^2/12', linestyle='--',alpha=0.9)
    plt.plot(t,2*D_m*t, label='2 D_m t', linestyle='--',alpha=0.9)
    plt.ylim(-10, L**2/10)
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(t,M[:,1,1], label='M_yy')
    s = total_steps*2//5
    D_sim = (M[-1,1,1]-M[s,1,1])/(2.0*dt*total_steps*3//5)
    intercept = M[s,1,1] - 2.0*D_sim*dt*s

    plt.plot(t,t*D_sim*2 + intercept, label='2 D_sim t', linestyle='--',alpha=0.9)
    plt.plot(t,2*D_m*t, label='2 D_m t', linestyle='--',alpha=0.9)
    plt.legend()
    plt.show()
    # D_sim = D[0, 0]  # Longitudinal dispersion coefficient in physical units [m^2/s]
    
    # print("Dispersion tensor at final step:\n", M_t_all[-1])

    # Theoretical calculation
    L = solid.shape[0] * dx
    g =10  # body force in m/s^2
    nu=1e-6  # physical kinematic viscosity in m^2/s
    u_max = 0.1*g*(L**2)/(8.0*nu) # Poiseuille analytic u-max.
    print("u_max =", u_max)
    # D_phys = u_max * L / target_Pe

    Pe = L * u_mean / D_m
    print("target_Pe =", target_Pe)
    print("Calculated Pe =", Pe)
    # Pe = target_Pe / 3.0  # = (2/3 u_max)*(L/2)/D_phys -> simplifies to target_Pe/3

    kappa = 1.0/210.0
    # D_eff_theory = D_phys * (1.0 + Pe**2 * kappa)
    D_theory = D_m*(1 + (2/105)*Pe**2) # m^2/s + (m/s)^2 * m^2 / (m^2/s) = m^2/s

    print("L =", L)
    print("D_phys =", D_phys)
    print("Pe (used) =", Pe)
    print(f"D_theory ={D_theory:.5e}")
    print(f"D_sim ={D_sim:.5e}")
    # print("D_laminar =", D_laminar)

    # print("D_eff =", D_eff)
    print("relative error theory-sim =", abs(D_theory - D_sim) / D_theory)
    print("D_theory/D_sim =", D_theory / D_sim)
    t_trans = (L)**2 / D_m
    t_sim = total_steps * dt
    print('t_sim / t_trans =', t_sim / t_trans)