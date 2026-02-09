import matplotlib.pyplot as plt
from src.porousflow.lbm.lbm import LBM_solver
import torch
import torch.nn.functional as F
import h5py
import numpy as np
from src.ml.models.vit import load_vit_model
from src.ml.models.convnext import load_convnext_model

file_name = 'rough_validation_2D.h5'
# file_name = 'smooth_validation_2D.h5'

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


# i = 0


def coarsegrain_to_128(x, out_h=128, out_w=128):
    _, _, H, W = x.shape

    factor_h = H // out_h   # 320 -> 128 => 2.5 (not integer!)
    factor_w = W // out_w   # 256 -> 128 => 2

    # Use adaptive pooling to handle non-integer scaling cleanly
    x = F.adaptive_avg_pool2d(x, (out_h, out_w))
    return x >0.5

# X = coarsegrain_to_128(torch.tensor(X[:,:,28:290]))[i,0]
# X = X[i,0]

# L_physical = 1e-3
# tau=0.6
# force_scaling=1e-1
# out = LBM_solver(X, max_iterations=1000, L_physical=L_physical,tau=tau,force_strength=force_scaling)
# Kxx, Kxy = out[1], out[2]

# out_2 = LBM_solver(X, force_dir=1, max_iterations=1000, L_physical=L_physical,tau=tau,force_strength=force_scaling)
# Kyx, Kyy = out_2[1], out_2[2]

# K_crop = np.array([[Kxx,Kxy],[Kyx,Kyy]])

# print(K_crop/8e-10)
# u = out[0]
# u_mag = np.sqrt(u[:,:,0]**2 + u[:,:,1]**2)
# u_2 = out_2[0]
# u_mag_2 = np.sqrt(u_2[:,:,0]**2 + u_2[:,:,1]**2)

def sim(X):
    X = X[0]

    L_physical = 1e-3
    tau=0.6
    force_scaling=1e-1
    out = LBM_solver(X, max_iterations=2000, L_physical=L_physical,tau=tau,force_strength=force_scaling)

    out_2 = LBM_solver(X, force_dir=1, max_iterations=2000, L_physical=L_physical,tau=tau,force_strength=force_scaling)
    u = out[0]
    u_mag = np.sqrt(u[:,:,0]**2 + u[:,:,1]**2)
    u_2 = out_2[0]
    u_mag_2 = np.sqrt(u_2[:,:,0]**2 + u_2[:,:,1]**2)
    return u_mag, u_mag_2
num_images=5
fig, ax = plt.subplots(num_images,3,figsize=(10,3*num_images))

for i in range(num_images):
    u_mag, u_mag_2 = sim(X[i])

    ax[i,0].set_title("Input")
    ax[i,0].imshow(X[i,0], cmap="gray")

    ax[i,1].set_title("Flow-x")
    im1 = ax[i,1].imshow(u_mag, cmap="viridis")
    fig.colorbar(im1, ax=ax[i,1])

    ax[i,2].set_title("Flow-y")
    im2 = ax[i,2].imshow(u_mag_2, cmap="viridis")
    fig.colorbar(im2, ax=ax[i,2])



plt.tight_layout()
plt.savefig('update_plots/real_media_rough.pdf')


# model = load_convnext_model('v1','atto',pretrained_path='results/convnext_atto/ConvNeXt-Atto_lr-0.0008_wd-0.1_bs-128_epochs-100_cosine_warmup-1250_clipgrad-True_pe-encoder-None_pe-None_mse.pth')
# model = load_vit_model('L16',pretrained_path='results/convnext_atto/ViT-L16_lr-0.0008_wd-0.1_bs-128_epochs-1500_cosine_warmup-250_clipgrad-True_pe-encoder-None_pe-None.pth')
# with torch.no_grad():
#     output= model(X.float().unsqueeze(0).unsqueeze(0))
#     print(output.reshape(-1,2,2))
# # print(Kxx, Kxy)
# # print(Kyx, Kyy)
