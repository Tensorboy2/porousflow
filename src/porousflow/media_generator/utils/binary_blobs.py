import numpy as np
from scipy.ndimage import gaussian_filter

def periodic_binary_blobs(n_dim =2, length=128, volume_fraction=0.5, blob_size_fraction=0.1, seed=0, rng=None):
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
    rs = np.random.default_rng(rng)
    shape = tuple([length] * n_dim)
    mask = np.zeros(shape)
    n_pts = max(int(1.0 / blob_size_fraction) ** n_dim, 1)
    points = (length * rs.random((n_dim, n_pts))).astype(int)
    mask[tuple(indices for indices in points)] = 1
    mask = gaussian_filter(
        mask,
        sigma=0.25 * length * blob_size_fraction,
        # preserve_range=False,
        mode='wrap',
    )
    threshold = np.percentile(mask, 100 * (1 - volume_fraction))
    return np.logical_not(mask < threshold)



if __name__ == "__main__":
    # Example usage
    img = periodic_binary_blobs(n_dim=2, length=128,volume_fraction=0.4,blob_size_fraction=0.1, seed=9)
    print("Generated periodic blobs image with shape:", img.shape)
    print("Poroisity (fraction of fluid):", np.mean(~img))  # Calculate the fraction of fluid
    print("Solid fraction:", np.mean(img))  # Calculate the fraction of solid

    import matplotlib.pyplot as plt
    plt.imshow(img, cmap='gray')
    plt.show()