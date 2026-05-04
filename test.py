import numpy as np
import matplotlib.pyplot as plt
import zarr
from plotting.ploting import figsize

root = zarr.open('data/train.zarr',mode='r')
filled_images_ds = root['filled_images']['filled_images']
lbm_results = root['lbm_results']
dispersion_results = root['dispersion_results']

def velocity_realignment(K,ux,uy,nu=1e-6):
    
    A = np.linalg.inv(K)*nu # This assumes that f=1 for the right units, which is the case for our LBM simulations.
    alpha_x, beta_x = A[0,0], A[0,1]
    alpha_y, beta_y = A[1,0], A[1,1]

    u_x_aligned = alpha_x * ux + beta_x * uy
    u_y_aligned = alpha_y * ux + beta_y * uy
    return u_x_aligned, u_y_aligned

# Histogram of average velocity magnitudes after realignment
# velocity_magnitudes = []
# for i in range (16000):
#     fluid_mask = ~filled_images_ds[i]
#     K = lbm_results['K'][i]
#     ux = lbm_results['ux_physical'][i]
#     uy = lbm_results['uy_physical'][i]
#     u_x_aligned, u_y_aligned = velocity_realignment(K,ux,uy)
#     velocity_magnitudes.append(np.mean(np.sqrt(u_x_aligned[:,:,0]**2 + u_x_aligned[:,:,1]**2)[fluid_mask]))
#     velocity_magnitudes.append(np.mean(np.sqrt(u_y_aligned[:,:,0]**2 + u_y_aligned[:,:,1]**2)[fluid_mask]))

# plt.figure(figsize=figsize)
# plt.hist(velocity_magnitudes, bins=200)
# plt.xlabel('Average Velocity Magnitude after Realignment')
# plt.ylabel('Frequency')
# plt.title('Histogram of Average Velocity Magnitudes after Realignment')
# plt.tight_layout()
# plt.savefig('velocity_magnitude_histogram.png')

fig, ax = plt.subplots(2,2,figsize=(10,6))
velocity_xx = []
velocity_xy = []
velocity_yx = []
velocity_yy = []
for i in range (16000):
    fluid_mask = ~filled_images_ds[i]
    K = lbm_results['K'][i]
    ux = lbm_results['ux_physical'][i]
    uy = lbm_results['uy_physical'][i]
    u_x_aligned, u_y_aligned = velocity_realignment(K,ux,uy)
    velocity_xx.append(np.mean(u_x_aligned[:,:,0][fluid_mask]))
    velocity_xy.append(np.mean(u_x_aligned[:,:,1][fluid_mask]))
    velocity_yx.append(np.mean(u_y_aligned[:,:,0][fluid_mask]))
    velocity_yy.append(np.mean(u_y_aligned[:,:,1][fluid_mask]))
    # velocity_x.append(np.mean(u_x_aligned[:,:,0][fluid_mask]))
    # velocity_y.append(np.mean(u_y_aligned[:,:,1][fluid_mask]))

ax[0,0].hist(velocity_xx, bins=10)
ax[0,0].set_xlabel('Average ux_x')
ax[0,0].set_ylabel('Frequency')
ax[0,0].set_title('Histogram of Average ux_x')
ax[0,1].hist(velocity_xy, bins=100)
ax[0,1].set_xlabel('Average ux_y')
ax[0,1].set_ylabel('Frequency')
ax[0,1].set_title('Histogram of Average ux_y')
ax[1,0].hist(velocity_yx, bins=100)
ax[1,0].set_xlabel('Average uy_x')
ax[1,0].set_ylabel('Frequency')
ax[1,0].set_title('Histogram of Average uy_x')
ax[1,1].hist(velocity_yy, bins=10)
ax[1,1].set_xlabel('Average uy_y')
ax[1,1].set_ylabel('Frequency')
ax[1,1].set_title('Histogram of Average uy_y')
plt.tight_layout()
plt.savefig('velocity_component_re_aligned_histogram.png',dpi=300)