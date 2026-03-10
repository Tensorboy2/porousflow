import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

def binary_blobs(n_dim =2, length=128, volume_fraction=0.5, blob_size_fraction=0.1, seed=0, rng=None, mode='wrap'):
    '''
    Modified from the scipy.ndimage.binary_blobs function to include mode "wrap", for periodic wrapping on structures.
    
    Parameters:
    - shape: tuple, the shape of the output image.
    - density: float, the density of the blobs.
    - sigma: float, the standard deviation for Gaussian smoothing.
    - seed: int, random seed for reproducibility.

    Returns:
    - A binary image with periodic blobs.
    '''
    rs = np.random.default_rng(seed=seed)
    shape = tuple([length] * n_dim)
    mask = np.zeros(shape)
    n_pts = max(int(1.0 / blob_size_fraction) ** n_dim, 1)
    points = (length * rs.random((n_dim, n_pts))).astype(int)
    mask[tuple(indices for indices in points)] = 1
    mask = gaussian_filter(
        mask,
        sigma=0.25 * length * blob_size_fraction,
        # preserve_range=False,
        mode=mode,
    )
    threshold = np.percentile(mask, 100 * (1 - volume_fraction))
    return np.logical_not(mask < threshold)
seed = 0
porosity = 0.4
sigma = 0.15


seed = 0
imgs = [binary_blobs(volume_fraction=1-porosity, blob_size_fraction=sigma,seed=seed,mode='nearest'),binary_blobs(volume_fraction=1-porosity, blob_size_fraction=sigma,seed = seed)]
from ploting import figsize
fig, axs = plt.subplots(1, 2, figsize=figsize)
titles = ['Non-periodic Binary Blobs', 'Periodic Binary Blobs']

for i, ax in enumerate(axs):
    tiled = np.tile(imgs[i], (3,3))
    h, w = imgs[i].shape
    
    ax.imshow(tiled, cmap='gray')
    
    # grid at tile boundaries
    ax.set_xticks(np.arange(0, 3*w+1, w))
    ax.set_yticks(np.arange(0, 3*h+1, h))
    
    ax.grid(color='red', linewidth=1)
    ax.set_title(titles[i])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

plt.savefig('thesis_plots/periodic_vs_non_periodic_example.pdf')