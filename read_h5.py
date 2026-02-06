import h5py
import numpy as np

file_name = 'rocks_validation.h5'

with h5py.File(file_name, 'r') as f:
    # Inspect dataset shapes
    print("bounds shape:", f['bounds'].shape)
    print("input/fill shape:", f['input/fill'].shape)
    print("output/p shape:", f['output/p'].shape)
    print("name shape:", f['name'].shape)

    # Load data
    bounds = f['bounds'][:]
    X = f['input/fill'][:]     # inputs
    Y = f['output/p'][:]       # targets
    names = f['name'][:]       # sample names (likely bytes)

print("X shape:", X.shape)
print("Y shape:", Y.shape)
print("First name:", names[0])

import matplotlib.pyplot as plt

i = 0


# plt.title("Output (pressure)")
# plt.imshow(Y[i,0], cmap="viridis")
# plt.colorbar()

# plt.tight_layout()
# plt.show()

from src.porousflow.lbm.lbm import LBM_solver
# print('starting sim')
# out = LBM_solver(X[i,0],max_iterations=1000)
# Kxx, Kxy = out[1], out[2]
# print('1/2')
# out_2 = LBM_solver(X[i,0],force_dir=1,max_iterations=1000)
# Kyx, Kyy = out_2[1], out_2[2]
# print('2/2')
# K = np.array([[Kxx,Kxy],[Kyx,Kyy]])

# plt.figure(figsize=(10,4))

# plt.subplot(1,3,1)
# plt.title("Input (fill)")
# plt.imshow(X[i,0], cmap="gray")
# plt.colorbar()
# u = out[0]
# u_mag = np.sqrt(u[:,:,0]**2 + u[:,:,1]**2)
# u_2 = out_2[0]
# u_mag_2 = np.sqrt(u_2[:,:,0]**2 + u_2[:,:,1]**2)

# # plt.figure(figsize=(10,4))

# plt.subplot(1,3,2)
# plt.title("Flow")
# plt.imshow(u_mag, cmap="viridis")
# plt.colorbar()
# plt.subplot(1,3,3)
# plt.title("Flow")
# plt.imshow(u_mag_2, cmap="viridis")
# plt.colorbar()


# plt.tight_layout()
# # plt.show()
# plt.savefig('update_plots/real_media.pdf')

from src.ml.models.convnext import load_convnext_model
import torch
def center_crop(x, size=128):
    _, _, H, W = x.shape
    top = (H - size) // 2
    left = (W - size) // 2
    return x[:, :, top:top+size, left:left+size]

X = torch.from_numpy(X[i,0]).float().unsqueeze(0).unsqueeze(0)
X_crop = center_crop(X)
model = load_convnext_model('v1','atto',pretrained_path='results/convnext_atto/ConvNeXt-Atto_lr-0.001_wd-0.1_bs-4_epochs-50_cosine_warmup-0_clipgrad-True_pe-encoder-None_pe-None_mse.pth')
output= model(X_crop)
print(output)
# print(Kxx, Kxy)
# print(Kyx, Kyy)
L_physical = 1e-3
tau=0.6
force_scaling=1e-1
out = LBM_solver(X_crop[0,0].numpy(), max_iterations=1000, L_physical=L_physical,tau=tau,force_strength=force_scaling)
Kxx, Kxy = out[1], out[2]

out_2 = LBM_solver(X_crop[0,0].numpy(), force_dir=1, max_iterations=1000, L_physical=L_physical,tau=tau,force_strength=force_scaling)
Kyx, Kyy = out_2[1], out_2[2]

K_crop = np.array([[Kxx,Kxy],[Kyx,Kyy]])

print(K_crop/8e-10)