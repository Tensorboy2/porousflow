'''
generate_media.py

script to generate porous media samples and save them along with metrics.
'''

# from src.porousflow.media_generator.media_generator import generate_media, generate_media_sample
# import os

# generate_media.py
# generate_datasets.py

import os
import zarr
# from src.porousflow.media_generator.media_generator import generate_media_into_zarr


from src.porousflow.media_generator.media_generator import generate_media_zarr
# import zarr
# import os

# def generate_into_zarr(zarr_path, n_samples, base_seed):
#     root = zarr.open(zarr_path, mode="r+")
#     images = root["images"]
#     K_arr = root["K"]
#     D_arr = root["D"]

#     for i in range(n_samples):
#         img, K, D = generate_media_sample(
#             shape=(128, 128),
#             porosity_range=(0.2, 0.8),
#             sigma_range=(0.1, 0.1),
#             seed=base_seed + i,
#         )

#         images[i] = img
#         K_arr[i] = K
#         D_arr[i] = D

#     print(f"Wrote {n_samples} samples to {zarr_path}")


# if __name__ == "__main__":
#     root_dir = os.path.dirname(os.path.abspath(__file__))

#     generate_into_zarr(os.path.join(root_dir,'data', "train.zarr"), 512, base_seed=0)
#     generate_into_zarr(os.path.join(root_dir,'data', "val.zarr"), 64, base_seed=512)
#     generate_into_zarr(os.path.join(root_dir,'data', "test.zarr"), 64, base_seed=576)


if __name__ == "__main__":
    
    # generate training data
    base_seed = 0
    root_path = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(root_path,"data","train")
    total_generated = generate_media_zarr(zarr_path=save_path,
        number_of_samples=512,
        shape=(128, 128),
        porosity_range=(0.2,0.8),
        sigma_range=(0.1, 0.1),
        base_seed=0,
    )
    print(f"Generated media samples and saved to '{save_path}'")
    print(f'Generated: {total_generated}')

    # generate validation data
    base_seed=total_generated
    save_path = os.path.join(root_path,"data","validation")
    total_generated += generate_media_zarr(zarr_path=save_path,
        number_of_samples=64,
        shape=(128, 128),
        porosity_range=(0.2,0.8),
        sigma_range=(0.1, 0.1),
        base_seed=base_seed,
    )
    print(f"Generated media samples and saved to '{save_path}'")
    print(f'Generated: {total_generated}')
    # generate test data
    base_seed=total_generated
    save_path = os.path.join(root_path,"data","test")
    total_generated += generate_media_zarr(zarr_path=save_path,
        number_of_samples=64,
        shape=(128, 128),
        porosity_range=(0.2,0.8),
        sigma_range=(0.1, 0.1),
        base_seed=base_seed,
    )


    print(f"Generated media samples and saved to '{save_path}'")
    print(f'Generated: {total_generated}')
