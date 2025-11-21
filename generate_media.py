'''
generate_media.py

script to generate porous media samples and save them along with metrics.
'''

from src.porousflow.media_generator.media_generator import generate_media
import os



if __name__ == "__main__":
    
    # generate training data
    base_seed = 0
    root_path = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(root_path,"data","media_samples_train")
    total_generated = generate_media(
        number_of_samples=512,
        shape=(128, 128),
        porosity_range=(0.2,0.8),
        sigma_range=(0.1, 0.1),
        base_seed=0,
        save_path=save_path
    )
    print(f"Generated media samples and saved to '{save_path}'")

    # generate validation data
    base_seed=total_generated
    save_path = os.path.join(root_path,"data","media_samples_validation")
    total_generated += generate_media(
        number_of_samples=64,
        shape=(128, 128),
        porosity_range=(0.2,0.8),
        sigma_range=(0.1, 0.1),
        base_seed=base_seed,
        save_path=save_path,
    )
    print(f"Generated media samples and saved to '{save_path}'")

    # generate test data
    base_seed=total_generated
    save_path = os.path.join(root_path,"data","media_samples_test")
    total_generated += generate_media(
        number_of_samples=64,
        shape=(128, 128),
        porosity_range=(0.2,0.8),
        sigma_range=(0.1, 0.1),
        base_seed=base_seed,
        save_path=save_path,
    )


    print(f"Generated media samples and saved to '{save_path}'")
