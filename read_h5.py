import matplotlib.pyplot as plt
from src.porousflow.lbm.lbm import LBM_solver
import torch
import torch.nn.functional as F
import h5py
import numpy as np
from src.ml.models.convnext import load_convnext_model

file_name = 'rocks_validation.h5'

with h5py.File(file_name, 'r') as f:
    # Inspect dataset shapes
    # print("bounds shape:", f['bounds'].shape)
    # print("input/fill shape:", f['input/fill'].shape)
    # print("output/p shape:", f['output/p'].shape)
    # print("name shape:", f['name'].shape)

    # Load data
    bounds = f['bounds'][:]
    X = f['input/fill'][:]     # inputs
    Y = f['output/p'][:]       # targets
    names = f['name'][:]       # sample names (likely bytes)

# print("X shape:", X.shape)
# print("Y shape:", Y.shape)
# print("First name:", names[0])


i = 0


def coarsegrain_to_128(x, out_h=128, out_w=128):
    _, _, H, W = x.shape

    factor_h = H // out_h   # 320 -> 128 => 2.5 (not integer!)
    factor_w = W // out_w   # 256 -> 128 => 2

    # Use adaptive pooling to handle non-integer scaling cleanly
    x = F.adaptive_avg_pool2d(x, (out_h, out_w))
    return x >0.5
X = coarsegrain_to_128(torch.tensor(X))[i,0]

L_physical = 1e-3
tau=0.6
force_scaling=1e-1
out = LBM_solver(X.numpy(), max_iterations=10000, L_physical=L_physical,tau=tau,force_strength=force_scaling)
Kxx, Kxy = out[1], out[2]

out_2 = LBM_solver(X.numpy(), force_dir=1, max_iterations=10000, L_physical=L_physical,tau=tau,force_strength=force_scaling)
Kyx, Kyy = out_2[1], out_2[2]

K_crop = np.array([[Kxx,Kxy],[Kyx,Kyy]])

print(K_crop/8e-10)
u = out[0]
u_mag = np.sqrt(u[:,:,0]**2 + u[:,:,1]**2)
u_2 = out_2[0]
u_mag_2 = np.sqrt(u_2[:,:,0]**2 + u_2[:,:,1]**2)


plt.figure(figsize=(10,3))
plt.subplot(1,3,1)
plt.title("Input (fill)")
plt.imshow(X, cmap="gray")
plt.subplot(1,3,2)
plt.title("Flow")
plt.imshow(u_mag, cmap="viridis")
plt.colorbar()
plt.subplot(1,3,3)
plt.title("Flow")
plt.imshow(u_mag_2, cmap="viridis")
plt.colorbar()


plt.tight_layout()
plt.savefig('update_plots/real_media.pdf')


model = load_convnext_model('v1','atto',pretrained_path='results/convnext_atto/ConvNeXt-Atto_lr-0.0008_wd-0.1_bs-128_epochs-100_cosine_warmup-1250_clipgrad-True_pe-encoder-None_pe-None_mse.pth')
with torch.no_grad():
    output= model(X.float().unsqueeze(0).unsqueeze(0))
    print(output.reshape(-1,2,2))
# # print(Kxx, Kxy)
# # print(Kyx, Kyy)
