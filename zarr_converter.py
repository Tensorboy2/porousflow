import numpy as np
import zarr

path = 'data/test.zarr'
root = zarr.open(path, mode='r')

# 1. Convert Images
# print("Converting images...")
# images = root['filled_images']['filled_images'][:]
# np.save('data/test_images_raw.npy', images)

# 2. Convert Targets (Dx and Dy)
# print("Converting targets...")
# # These are likely stored as (N, 5) or similar in your Zarr
# targets_x = root['dispersion_results']['Dx'][:]
# targets_y = root['dispersion_results']['Dy'][:]

# np.save('data/test_targets_x.npy', targets_x)
# np.save('data/test_targets_y.npy', targets_y)
print("Saving targets...")
np.save('test_targets_k.npy', root['lbm_results']['K'][:])
print("Conversion complete. Files saved to 'data/'.")