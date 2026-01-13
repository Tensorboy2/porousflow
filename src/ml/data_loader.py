'''
Docstring for data_loader

This module implements the datasets and dataloaders needed to train neural networks on predicting both
permeability and dispersion from porous media images. It also implements the augmentations needed for training.
'''
import os
import zarr
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torchvision.transforms.functional as tf
import random

class PermeabilityTransform:
    def __init__(self):
        self.permutations = [
            ('I', lambda img: img, lambda K: K),
            ('R', lambda img: tf.rotate(img, 90),
                  lambda K: torch.tensor([K[3], -K[2], -K[1], K[0]])),  # 90°
            ('RR', lambda img: tf.rotate(img, 180),
                   lambda K: torch.tensor([K[0], K[1], K[2], K[3]])),  # 180°
            ('RRR', lambda img: tf.rotate(img, 270),
                    lambda K: torch.tensor([K[3], -K[2], -K[1], K[0]])),  # 270°
            ('H', lambda img: tf.hflip(img),
                  lambda K: torch.tensor([K[0], -K[1], -K[2], K[3]])),  # Horizontal flip
            ('HR', lambda img: tf.rotate(tf.hflip(img), 90),
                   lambda K: torch.tensor([K[3], K[2], K[1], K[0]])),  # H + 90°
            ('HRR', lambda img: tf.rotate(tf.hflip(img), 180),
                    lambda K: torch.tensor([K[0], -K[1], -K[2], K[3]])),  # H + 180°
            ('HRRR', lambda img: tf.rotate(tf.hflip(img), 270),
                     lambda K: torch.tensor([K[3], K[2], K[1], K[0]])),  # H + 270°
        ]

    def __call__(self, image, K):
        # Randomly choose one of the 8 permutations
        _, img_transform, k_transform = random.choice(self.permutations)
        image = img_transform(image)
        K = k_transform(K)
        return image, K
    
class DispersionTransform:
    def __init__(self):
        self.permutations = [
            ('I', lambda img: img, lambda K: K),
            ('R', lambda img: tf.rotate(img, 90),
                  lambda K: torch.tensor([K[3], -K[2], -K[1], K[0]])),  # 90°
            ('RR', lambda img: tf.rotate(img, 180),
                   lambda K: torch.tensor([K[0], K[1], K[2], K[3]])),  # 180°
            ('RRR', lambda img: tf.rotate(img, 270),
                    lambda K: torch.tensor([K[3], -K[2], -K[1], K[0]])),  # 270°
            ('H', lambda img: tf.hflip(img),
                  lambda K: torch.tensor([K[0], -K[1], -K[2], K[3]])),  # Horizontal flip
            ('HR', lambda img: tf.rotate(tf.hflip(img), 90),
                   lambda K: torch.tensor([K[3], K[2], K[1], K[0]])),  # H + 90°
            ('HRR', lambda img: tf.rotate(tf.hflip(img), 180),
                    lambda K: torch.tensor([K[0], -K[1], -K[2], K[3]])),  # H + 180°
            ('HRRR', lambda img: tf.rotate(tf.hflip(img), 270),
                     lambda K: torch.tensor([K[3], K[2], K[1], K[0]])),  # H + 270°
        ]

    def __call__(self, image, Dx, Dy):
        # Randomly choose one of the 8 permutations
        _, img_transform, D_transform = random.choice(self.permutations)
        image = img_transform(image)
        # K = _transform(K)
        # return image, K

class PermeabilityDataset(Dataset):
    def __init__(self, file_path, transform=None):
        self.root = zarr.open(file_path, mode='r')
        self.filled_images_ds = self.root['filled_images']['filled_images']
        self.targets_ds = self.root['lbm_results']['K']

        self.transform = transform

    def __len__(self):
        return self.filled_images_ds.shape[0]

    def __getitem__(self, idx):
        # Fetch numpy versions of the data
        image = self.filled_images_ds[idx]
        K = self.targets_ds[idx].flatten()

        # turn into torch tensors
        image = torch.from_numpy(image).float().unsqueeze(0)  # add channel dimension
        K = torch.from_numpy(K).float()*1e9  # convert to mD
        
        # transform if needed
        if self.transform:
            image, K = self.transform(image, K)

        return image, K
    
class DispersionDataset(Dataset):
    def __init__(self, file_path, transform=None, num_training_samples=None):
        self.root = zarr.open(file_path, mode='r')
        self.filled_images_ds = self.root['filled_images']['filled_images']
        self.targets_ds_x = self.root['dispersion_results']['Dx']
        self.targets_ds_y = self.root['dispersion_results']['Dy']

        self.transform = transform

        if num_training_samples is not None:
            self.root.attrs['N'] = min(num_training_samples, self.root.attrs['N'])

    def __len__(self):
        return self.root.attrs['N']

    def __getitem__(self, idx):
        # Fetch numpy versions of the data
        image = self.filled_images_ds[idx]
        Dx = self.targets_ds_x[idx].reshape(5,4) # shape (1,5,2,2)
        Dy = self.targets_ds_y[idx].reshape(5,4)

        # turn into torch tensors
        image = torch.from_numpy(image).float().unsqueeze(0)  # add channel dimension
        target = torch.cat((torch.from_numpy(Dx).float(), torch.from_numpy(Dy).float()), dim=1)

        # transform if needed
        if self.transform:
            image, target = self.transform(image, target)
        return image, target

def get_permeability_dataloader(file_path,config):
    '''
    Docstring for get_permeability_dataloader
    
    :param file_path: Path to the dataset
    :param config: Configuration dictionary with dataloader parameters
    :return: train_loader, val_loader, test_loader
    '''
    # General options
    batch_size = config.get('batch_size',32)
    num_workers = config.get('num_workers',2)

    # Cuda specific options
    persistent_workers = config.get('persistent_workers',False)
    pin_memory = config.get('pin_memory',False)
    pin_memory_device = config.get('pin_memory_device','')
    prefetch_factor = config.get('prefetch_factor',None)

    train_path = os.path.join(file_path,'train.zarr')
    val_path = os.path.join(file_path,'validation.zarr')
    test_path = os.path.join(file_path,'test.zarr')

    train_dataset = PermeabilityDataset(train_path, transform=PermeabilityTransform())
    val_dataset = PermeabilityDataset(val_path)
    test_dataset = PermeabilityDataset(test_path)

    train_loader = DataLoader(train_dataset, 
                              batch_size=batch_size, 
                              shuffle=True, 
                              num_workers=num_workers,
                              persistent_workers=persistent_workers,
                              pin_memory=pin_memory,
                              prefetch_factor=prefetch_factor,
                              pin_memory_device=pin_memory_device)
    val_loader = DataLoader(val_dataset, 
                            batch_size=batch_size, 
                            shuffle=False, 
                            num_workers=num_workers,
                            persistent_workers=persistent_workers,
                            pin_memory=pin_memory,
                            prefetch_factor=prefetch_factor,
                            pin_memory_device=pin_memory_device)
    test_loader = DataLoader(test_dataset, 
                             batch_size=batch_size, 
                             shuffle=False, 
                             num_workers=num_workers,
                             persistent_workers=persistent_workers,
                            pin_memory=pin_memory,
                            prefetch_factor=prefetch_factor,
                            pin_memory_device=pin_memory_device)
    
    return train_loader, val_loader, test_loader

def get_dispersion_dataloader(file_path,config):
    '''
    Docstring for get_dispersion_dataloader
    
    :param file_path: Path to the dataset
    :param config: Configuration dictionary with dataloader parameters
    :return: train_loader, val_loader, test_loader
    '''
    # General options
    batch_size = config.get('batch_size',32)
    num_workers = config.get('num_workers',2)

    # Cuda specific options
    persistent_workers = config.get('persistent_workers',False)
    pin_memory = config.get('pin_memory',False)
    pin_memory_device = config.get('pin_memory_device','')
    prefetch_factor = config.get('prefetch_factor',None)

    train_path = os.path.join(file_path,'train.zarr')
    val_path = os.path.join(file_path,'validation.zarr')
    test_path = os.path.join(file_path,'test.zarr')

    train_dataset = DispersionDataset(train_path,num_training_samples=config.get('num_training_samples',None))
    val_dataset = DispersionDataset(val_path)
    test_dataset = DispersionDataset(test_path)

    train_loader = DataLoader(train_dataset, 
                              batch_size=batch_size, 
                              shuffle=True, 
                              num_workers=num_workers,
                              persistent_workers=persistent_workers,
                              pin_memory=pin_memory,
                              prefetch_factor=prefetch_factor,
                            pin_memory_device=pin_memory_device)
    val_loader = DataLoader(val_dataset, 
                            batch_size=batch_size, 
                            shuffle=False, 
                            num_workers=num_workers,
                            persistent_workers=persistent_workers,
                            pin_memory=pin_memory,
                            prefetch_factor=prefetch_factor,
                            pin_memory_device=pin_memory_device)
    test_loader = DataLoader(test_dataset, 
                             batch_size=batch_size, 
                             shuffle=False, 
                             num_workers=num_workers,
                             persistent_workers=persistent_workers,
                            pin_memory=pin_memory,
                            prefetch_factor=prefetch_factor,
                            pin_memory_device=pin_memory_device)
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    # Example usage
    import time, psutil, os
    config = {
        'batch_size': 1024,
        'num_workers': 0,
        # 'prefetch_factor': 8,
    }

    proc = psutil.Process(os.getpid())
    start_mem = proc.memory_info().rss / 1e6  # MB
    start = time.time()

    train_loader, val_loader, test_loader = get_dispersion_dataloader('data', config)
    for i, (image, target) in enumerate(train_loader):
        print(f"Batch {i}: Image batch shape: {image.shape}, Target batch shape: {target.shape}")
        if i >= 20:  # first 20 batches
            break

    end = time.time()
    end_mem = proc.memory_info().rss / 1e6

    print(f"Time per batch: {(end-start)/20:.3f} s, RAM usage increase: {end_mem-start_mem:.2f} MB")
