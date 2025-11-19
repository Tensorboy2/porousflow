'''
media_generator.py

Module for generating porouse media structures.
'''
import numpy as np
import os
import pandas as pd

from .utils.binary_blobs import periodic_binary_blobs
from .utils.precolation_check import detect_percolation, fill_non_percolating_fluid

def generate_media(number_of_samples=1, 
                   shape=(128, 128), 
                   porosity_range=(0.1, 0.9),
                   sigma_range=(0.05, 0.15),
                   base_seed=0,
                   save_path=None):
    '''
    Generate porous media structure with periodic binary blobs and ensure percolation.

    Parameters:
    - shape: tuple, the shape of the output image.
    - density: float, the density of the blobs.
    - sigma: float, the standard deviation for Gaussian smoothing.
    - seed: int, random seed for reproducibility.

    Returns:
    - A binary image with percolating porous media structure.
    '''
    images = np.zeros((number_of_samples, *shape), dtype=bool)
    filled_imgs = np.zeros((number_of_samples, *shape), dtype=bool)

    metrics = {'seed': [],
               'porosity': [],
               'sigma': [],
               'percolates_x': [],
               'percolates_y': []}

    total_generated = 0
    total_accepted = 0
    while total_accepted < number_of_samples:
        seed = base_seed + total_generated
        porosity = np.random.uniform(*porosity_range)
        density = 1.0 - porosity
        sigma = np.random.uniform(*sigma_range)

        # img = periodic_binary_blobs(n_dim=2, density=density, sigma=sigma, seed=seed)
        img = periodic_binary_blobs(n_dim=2, length=128,volume_fraction=density, blob_size_fraction=sigma, seed=seed)

        x_perc, y_perc, _ = detect_percolation(img)

        if x_perc and y_perc:
            filled_img = fill_non_percolating_fluid(img)
            filled_imgs[total_accepted] = filled_img
            images[total_accepted] = img
            total_accepted += 1
        total_generated += 1
    
        metrics['seed'].append(seed)
        metrics['porosity'].append(np.mean(~img))
        metrics['sigma'].append(sigma)
        metrics['percolates_x'].append(x_perc)
        metrics['percolates_y'].append(y_perc)
    
    metrics_df = pd.DataFrame(metrics)

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        np.savez_compressed(os.path.join(save_path, 'media_samples.npz'), 
                            images=images, filled_images=filled_imgs)
        metrics_df.to_csv(os.path.join(save_path, 'media_metrics.csv'), index=False)
    
    print(f"Generated {number_of_samples} samples, accepted {total_accepted}, total tried {total_generated}")
    return total_generated

    



# if __name__ == "__main__":

