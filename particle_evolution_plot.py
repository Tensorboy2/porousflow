
import numpy as np
import matplotlib.pyplot as plt
import zarr

root = zarr.open('data/train.zarr',mode='r')
solid = root['filled_images']['filled_images'][0]
lbm_results = root['lbm_results']
ux = lbm_results['ux_physical'][0]
uy = lbm_results['uy_physical'][0]
K = lbm_results['K'][0]

def velocity_realignment(K,ux,uy,nu=1e-6):
    
    A = np.linalg.inv(K)*nu # This assumes that f=1 for the right units, which is the case for our LBM simulations.
    alpha_x, beta_x = A[0,0], A[0,1]
    alpha_y, beta_y = A[1,0], A[1,1]

    u_x_aligned = alpha_x * ux + beta_x * uy
    u_y_aligned = alpha_y * ux + beta_y * uy
    return u_x_aligned, u_y_aligned

from src.porousflow.dispersion import run_dispersion_sim_physical_test

dx = 1.0
L = solid.shape[0]
alpha = 0.1
steps = int(50 * L**2)
nu = 1e-6
# align velicity fields along principal directions
A = np.linalg.inv(K)*nu # This assumes that f=1 for the right units, which is the case for our LBM simulations.
alpha_x, beta_x = A[0,0], A[0,1]
alpha_y, beta_y = A[1,0], A[1,1]

u_x_aligned = alpha_x * ux + beta_x * uy
u_y_aligned = alpha_y * ux + beta_y * uy


# --- 1. Normalize velocity fields ---
fluid_mask = ~solid
# ux_mean = np.mean(np.abs(ux[fluid_mask]))
# uy_mean = np.mean(np.abs(uy[fluid_mask]))

ux_norm = u_x_aligned#ux / ux_mean
uy_norm = u_y_aligned#uy / uy_mean

ux_max = np.max(np.abs(ux_norm))
uy_max = np.max(np.abs(uy_norm))

# --- 2. Molecular diffusivity ---
# Pe = (mean_u * L) / D_m, and mean_u=1 after normalization
Pe=10
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

_,_,frames = run_dispersion_sim_physical_test(solid, u_x_aligned,steps=200_000)

# --- Plot snapshots ---
from plotting.ploting import figsize

snapshot_indices = [0, 50, -1]  # or pick specific ones

# Velocity magnitude for background
umag = np.sqrt(u_x_aligned[:,:,0]**2 + u_x_aligned[:,:,1]**2)

# Mask solid regions in umag
umag_masked = np.where(fluid_mask, umag, np.nan)

fig, axes = plt.subplots(1, len(snapshot_indices), figsize=(figsize[0],figsize[1]*0.6))

for ax, idx in zip(axes, snapshot_indices):
    particles = frames[idx]  # assumed shape (N, 2), columns: [x, y]

    im = ax.imshow(
        umag_masked.T,
        cmap="viridis",
        origin="lower",
        interpolation="nearest",
    )

    # Overlay solid mask in gray
    solid_overlay = np.where(solid, 0.0, np.nan)
    ax.imshow(solid_overlay.T, cmap="gray", origin="lower", alpha=0.4, interpolation="nearest")

    # Scatter particle positions
    ax.scatter(
        particles[:, 0],  # x  (column index)
        particles[:, 1],  # y  (row index)
        s=2,
        c="red",
        alpha=0.6,
        linewidths=0,
    )

    # ax.set_title(f"Frame {idx if idx >= 0 else len(frames) + idx}")
    ax.axis("off")

# fig.colorbar(im, ax=axes[-1], label=r"$|\mathbf{u}|$", shrink=0.7)
plt.tight_layout()
plt.savefig("dispersion_snapshots.png", dpi=300)
# plt.show()