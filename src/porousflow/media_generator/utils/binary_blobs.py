import numpy as np
from scipy.ndimage import gaussian_filter

def periodic_binary_blobs(shape=(128, 128), density=0.5, sigma=2.0, seed=0):
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
    rng = np.random.default_rng(seed)
    noise = rng.random(shape)
    smooth = gaussian_filter(noise, sigma=shape[0]*sigma, mode='wrap')
    threshold = np.percentile(smooth, (1.0 - density) * 100)
    return smooth > threshold  # True = solid, False = fluid

if __name__ == "__main__":
    # Example usage
    img = periodic_binary_blobs(shape=(128, 128), density=0.6, sigma=2.0, seed=9)
    print("Generated periodic blobs image with shape:", img.shape)
    print("Poroisity (fraction of fluid):", np.mean(~img))  # Calculate the fraction of fluid
    print("Solid fraction:", np.mean(img))  # Calculate the fraction of solid