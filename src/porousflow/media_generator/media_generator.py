'''
media_generator.py

Module for generating porouse media structures.
'''
import numpy as np
import os
import pandas as pd
import zarr

if __name__ == '__main__':
    from utils.binary_blobs import periodic_binary_blobs
    from utils.precolation_check import detect_percolation, fill_non_percolating_fluid_periodic
else:
    from .utils.binary_blobs import periodic_binary_blobs
    from .utils.precolation_check import detect_percolation, fill_non_percolating_fluid_periodic

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
            filled_img = fill_non_percolating_fluid_periodic(img)
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
    
    print(f"Generated {number_of_samples} samples, accepted {total_accepted}, total tried {total_generated}")

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        np.savez_compressed(os.path.join(save_path, 'media_samples.npz'), 
                            images=images, filled_images=filled_imgs)
        metrics_df.to_csv(os.path.join(save_path, 'media_metrics.csv'), index=False)
    
    return total_generated

def generate_media_zarr(zarr_path,
                        number_of_samples=1, 
                        shape=(128, 128), 
                        porosity_range=(0.1, 0.9),
                        sigma_range=(0.05, 0.15),
                        base_seed=0):
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
    root = zarr.open(zarr_path+'.zarr',mode='a')

    # images = root.create_group('images')
    # filled_images = root.create_group('filled_images')

    images_ds = root.create_group('images').create_dataset(
        "images",
        shape=(number_of_samples, shape[0], shape[1]),
        chunks=(1, shape[0], shape[1]),   # optimal for batch reads & writes
        dtype="bool"
    )

    filled_images_ds = root.create_group('filled_images').create_dataset(
        "filled_images",
        shape=(number_of_samples, shape[0], shape[1]),
        chunks=(1, shape[0], shape[1]),   # optimal for batch reads & writes
        dtype="bool"
    )


    metrics_ds = root.create_group("metrics").create_dataset(
        "metrics",
        shape=(number_of_samples,),
        dtype=np.dtype([
            ('seed', 'i8'),
            ('porosity', 'f4'),
            ('effective_porosity', 'f4'),
            ('sigma', 'f4'),
            ('percolates_x', 'b1'),
            ('percolates_y', 'b1')
        ]),
        chunks=(1,)
    )

    root.attrs['N'] = number_of_samples
    root.attrs['H'] = shape[0]
    root.attrs['W'] = shape[1]


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
            filled_img = fill_non_percolating_fluid_periodic(img)
            images_ds[total_accepted] = img
            filled_images_ds[total_accepted] = filled_img
            metrics_ds[total_accepted] = (seed, np.mean(~img), np.mean(~filled_img), sigma, x_perc, y_perc)
            total_accepted += 1

        total_generated += 1

    
    return total_generated

def generate_media_sample(
    shape=(128, 128),
    porosity_range=(0.1, 0.9),
    sigma_range=(0.05, 0.15),
    seed=0,
):
    """
    Generate one porous media sample that percolates in x and y.
    Returns:
        - img:            raw porous media (bool, HxW)
        - filled_img:     filled version (bool, HxW)
        - metrics: dict with porosity, sigma, seed, percolation flags
    """

    while True:
        porosity = np.random.uniform(*porosity_range)
        density = 1.0 - porosity
        sigma = np.random.uniform(*sigma_range)

        img = periodic_binary_blobs(
            n_dim=2,
            length=shape[0],
            volume_fraction=density,
            blob_size_fraction=sigma,
            seed=seed,
        )

        x_perc, y_perc, _ = detect_percolation(img)

        if x_perc and y_perc:
            filled_img = fill_non_percolating_fluid_periodic(img)

            metrics = {
                "seed": seed,
                "porosity": np.mean(~img),
                "sigma": sigma,
                "percolates_x": x_perc,
                "percolates_y": y_perc,
            }

            return img, filled_img, metrics

        # Otherwise try next seed
        seed += 1
    
def generate_media_into_zarr(
    zarr_group,
    num_samples,
    start_seed,
    shape=(128, 128),
    porosity_range=(0.1, 0.9),
    sigma_range=(0.05, 0.15),
):
    """
    Write num_samples porous media samples into a Zarr group.
    Returns the next free seed (last used seed + 1).
    """

    current_seed = start_seed
    idx = 0

    for i in range(num_samples):

        img, filled_img, meta = generate_media_sample(
            shape=shape,
            porosity_range=porosity_range,
            sigma_range=sigma_range,
            seed=current_seed,
        )

        # Store arrays
        zarr_group["images"][i] = img
        zarr_group["filled_images"][i] = filled_img

        # Store metrics
        zarr_group["metrics"]["seed"][i] = meta["seed"]
        zarr_group["metrics"]["porosity"][i] = meta["porosity"]
        zarr_group["metrics"]["sigma"][i] = meta["sigma"]
        zarr_group["metrics"]["percolates_x"][i] = meta["percolates_x"]
        zarr_group["metrics"]["percolates_y"][i] = meta["percolates_y"]

        current_seed = meta["seed"] + 1

    return current_seed


if __name__ == "__main__":
    import sys
    num = float(sys.argv[1])
    save_path = sys.argv[2]

    generate_media(number_of_samples=num, save_path=save_path)
    print(f"Processing number {num}")
    print(f"saving to {save_path}")

