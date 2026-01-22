"""
Tracer particle simulation with built-in GIF recording for dispersion in porous flow.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


def run_dispersion_with_recording(
    solid, 
    u, 
    steps=10, 
    num_particles=1000,
    velocity_strength=1.0,
    dt=1e-3,
    D=1.0,
    dx=1.0,
    record_interval=1
):
    """
    Run dispersion simulation and record particle positions at specified intervals.
    
    Parameters:
        solid: boolean array marking solid cells
        u: normalized velocity field
        steps: total simulation steps
        num_particles: number of tracer particles
        velocity_strength: velocity scaling factor
        dt: time step
        D: molecular diffusion coefficient
        dx: lattice spacing
        record_interval: record every N steps (default=1 for all steps)
    
    Returns:
        D_longitudinal: longitudinal dispersion coefficient
        M_transverse: transverse dispersion coefficient
        positions_history: array of particle positions at each recorded step
        record_steps: array of step numbers for each recording
    """
    from src.porousflow.dispersion import run_dispersion_sim_physical_test
    
    # This is a wrapper that calls your existing function
    # If you want to modify the internal recording, you'll need to update
    # the run_dispersion_sim_physical_test function itself
    
    D, M, positions_for_plot = run_dispersion_sim_physical_test(
        solid, u, steps=steps, num_particles=num_particles,
        velocity_strength=velocity_strength, dt=dt, D=D, dx=dx
    )
    
    
    return D, M, positions_for_plot, positions_for_plot.shape[0]


def create_dispersion_gif(
    positions_history,
    record_steps,
    solid,
    filename='dispersion_simulation.gif',
    fps=30,
    particle_size=1,
    particle_color='red',
    figsize=(8, 8),
    show_velocity_field=False,
    u_field=None
):
    
    fig, ax = plt.subplots(figsize=figsize)
    # Ensure positions are a numpy array (accept lists too)
    positions_history = np.asarray(positions_history)

    # Show velocity magnitude background if provided
    if u_field is not None:
        u_mag = np.sqrt(u_field[:, :, 0]**2 + u_field[:, :, 1]**2)
        u_mag_masked = np.where(solid, np.nan, u_mag)
        ax.imshow(u_mag_masked.T, cmap='viridis', origin='lower', alpha=0.6)

    # Initialize scatter plot with empty data
    scat = ax.scatter([], [], s=particle_size, c=particle_color, alpha=0.7)

    # def init():
    #     scat.set_offsets(np.empty((0, 2)))
    #     return scat,

    def update(frame):
        pos = positions_history[frame]
        scat.set_offsets(pos)
        return scat,
    
    # Create animation
    anim = FuncAnimation(
        fig, update, 
        frames=positions_history.shape[0], 
        interval=1000/fps, 
        blit=True
    )
    
    # Save as GIF
    writer = PillowWriter(fps=fps)
    anim.save(filename, writer=writer)
    plt.close(fig)
    print(f"Animation saved as {filename}")



if __name__ == "__main__":
    import zarr
    
    # Load data
    data = zarr.open('data/train.zarr', mode='r')
    solid = data['filled_images']['filled_images'][0]
    ux = data['lbm_results']['ux_physical'][0]
    uy = data['lbm_results']['uy_physical'][0]
    K = data['lbm_results']['K'][0]
    nu = 1e-6
    # Normalize velocity
    # u_mean = np.mean(u_physical[:, :][~solid])
    # u = u_physical / u_mean
    
    # Simulation parameters
    dx = 1.0
    L = solid.shape[0]
    alpha = 0.1
    steps = 100000
    Pe = 10
    # align velicity fields along principal directions
    A = np.linalg.inv(K)*nu
    alpha_x, beta_x = A[0,0], A[0,1]

    u_x_aligned = alpha_x * ux + beta_x * uy


    # --- 1. Normalize velocity fields ---
    fluid_mask = ~solid
    # ux_mean = np.mean(np.abs(ux[fluid_mask]))
    # uy_mean = np.mean(np.abs(uy[fluid_mask]))

    ux_norm = u_x_aligned#ux / ux_mean

    ux_max = np.max(np.abs(ux_norm))

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

    dt = compute_dt(ux_max)

    # total_steps = np.arange(steps) * dt
    
    # Run simulation with recording
    print("\nRunning dispersion simulation...")
    D, M, positions_history, record_steps = run_dispersion_with_recording(
        solid, ux_norm,
        steps=steps,
        num_particles=100,
        velocity_strength=1.0,
        dt=dt,
        D=D_m,
        dx=dx,
        record_interval=10  # Record every step
    )
    
    # plot M
    fig = plt.figure(figsize=(6,6))
    plt.plot(M[:,0,0], label=r'$M_{xx}$')
    plt.plot(M[:,0,1], label=r'$M_{xy}$')
    plt.plot(M[:,1,0], label=r'$M_{yx}$')
    plt.plot(M[:,1,1], label=r'$M_{yy}$')
    plt.xlabel('Steps')
    plt.ylabel('Covariance')
    plt.legend()
    # plt.title('Transverse Dispersion Coefficients over Time')
    plt.grid()
    plt.tight_layout()
    plt.savefig('porelab_junior_plots/covariance.png', dpi=300)
    # plt.close(fig)

    # print(f"Simulation complete: {len(positions_history)} frames recorded")
    # print(f"Longitudinal dispersion coefficient D: {D}")
    # print(f"Transverse dispersion coefficient M: {M}")
    
    # Create standard GIF
    print("\nCreating standard animation...")
    # create_dispersion_gif(
    #     positions_history,
    #     record_steps,
    #     solid,
    #     filename='porelab_junior_plots/dispersion_simulation.gif',
    #     fps=25,
    #     particle_size=20,
    #     particle_color='red',
    #     u_field=ux_norm
    # )
    
    print("\nAll animations created successfully!")