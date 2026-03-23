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
    
    ax.grid(color='red', linewidth=1,alpha=0.3)
    ax.set_title(titles[i])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

plt.savefig('thesis_plots/periodic_vs_non_periodic_example.pdf')

# from src.porousflow.media_generator.utils.precolation_check import fill_non_percolating_fluid, fill_non_percolating_fluid_periodic
import numpy as np
from scipy.ndimage import  binary_dilation, label

STRUCTURE_8 = np.ones((3, 3), dtype=int) # 8-connectivity structure

def detect_percolation(binary_img):
    """Check if fluid clusters percolate in x or y (periodic)."""

    dilated = binary_dilation(binary_img,iterations=2)
    labeled, _ = label(~dilated,STRUCTURE_8)
    h, w = labeled.shape

    # Boundary check:
    x_perc = any(labeled[0, j] != 0 and labeled[0, j] == labeled[-1, j] for j in range(w))
    y_perc = any(labeled[i, 0] != 0 and labeled[i, 0] == labeled[i, -1] for i in range(h))

    return x_perc, y_perc, labeled


def fill_non_percolating_fluid(binary_img):
    """
    # Not in use!
    Turns non-percolating fluid regions into solid.
    """
    labels, _ = label(~binary_img)
    top, bottom = set(labels[0, :]) - {0}, set(labels[-1, :]) - {0}
    left, right = set(labels[:, 0]) - {0}, set(labels[:, -1]) - {0}

    percolating = (top & bottom) | (left & right)
    non_perc_mask = (labels > 0) & ~np.isin(labels, list(percolating))

    filled_img = binary_img.copy()
    filled_img[non_perc_mask] = True  # fill in fluid to solid
    return filled_img

def fill_non_percolating_fluid_periodic(binary_img):
    """Turns non-percolating fluid regions into solid with periodic BCs."""
    h, w = binary_img.shape

    tiled = np.tile(~binary_img, (3, 3))
    labels, _ = label(tiled,STRUCTURE_8)

    center_labels = labels[h:2*h, w:2*w]

    percolating_labels = set()

    for label_id in np.unique(center_labels):
        if label_id == 0:
            continue
        mask = labels == label_id

        left_edge = mask[h:2*h, w] | mask[h:2*h, 2*w-1]
        top_edge  = mask[h, w:2*w] | mask[2*h-1, w:2*w]

        if left_edge.any() or top_edge.any():
            percolating_labels.add(label_id)

    filled_img = binary_img.copy()
    non_perc_mask = (center_labels > 0) & ~np.isin(center_labels, list(percolating_labels))
    filled_img[non_perc_mask] = True  # Fill holes to solid

    return filled_img

fig,ax = plt.subplots(1,3,figsize=(figsize[0],figsize[1]*0.6))
image = binary_blobs(length=128,volume_fraction=1-porosity,blob_size_fraction=sigma,seed=5)
import matplotlib as mpl
def make_phase_image(original, filled):
    """0=solid, 1=open fluid, 2=filled/trapped fluid"""
    phase = np.zeros_like(original, dtype=int)
    phase[~original] = 0          # solid
    phase[original]  = 1          # fluid
    phase[~original & filled] = 2 # trapped (was fluid, now filled)
    return phase
a  =0.2
c1 = mpl.colormaps['Blues'](a)
c2 = mpl.colormaps['Reds'](a)
cmap = mpl.colors.ListedColormap(['black', 'white', c1])
cmap_2 = mpl.colors.ListedColormap(['black', 'white', c2])

filled_std  = fill_non_percolating_fluid(image)
filled_per  = fill_non_percolating_fluid_periodic(image)

ax[0].imshow(image,                            cmap='gray')
ax[0].set_title('Original')
ax[1].imshow(make_phase_image(image, filled_std),  cmap=cmap, vmin=0, vmax=2)
ax[1].set_title('Standard fill')
ax[2].imshow(make_phase_image(image, filled_per),  cmap=cmap_2, vmin=0, vmax=2)
ax[2].set_title('Periodic fill')
for a in ax.flatten():
    a.axis(False)
plt.tight_layout()
plt.savefig('thesis_plots/periodic_filling_demo.pdf')