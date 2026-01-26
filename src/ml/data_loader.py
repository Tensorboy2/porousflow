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
    
def swap_diag(D):
    # [Kxx, Kxy, Kyx, Kyy] → [Kyy, Kxy, Kyx, Kxx]
    return D[[3, 1, 2, 0]]

class DispersionTransform:
    def __init__(self):
        self.permutations = [
            # Identity
            lambda img, Dx: (img, Dx),

            # R90
            lambda img, Dx: (
                tf.rotate(img, 90),
                # swap_diag(Dy),
                swap_diag(Dx),
            ),

            # R180
            lambda img, Dx: (
                tf.rotate(img, 180),
                Dx,
                # Dy,
            ),

            # R270
            lambda img, Dx: (
                tf.rotate(img, 270),
                # swap_diag(Dy),
                swap_diag(Dx),
            ),

            # H
            lambda img, Dx: (
                tf.hflip(img),
                Dx,
                # Dy,
            ),

            # H + R90
            lambda img, Dx: (
                tf.rotate(tf.hflip(img), 90),
                # swap_diag(Dy),
                swap_diag(Dx),
            ),

            # H + R180
            lambda img, Dx: (
                tf.rotate(tf.hflip(img), 180),
                Dx,
                # Dy,
            ),

            # H + R270
            lambda img, Dx: (
                tf.rotate(tf.hflip(img), 270),
                # swap_diag(Dy),
                swap_diag(Dx),
            ),
        ]

    def __call__(self, image, Dx):
        t = random.choice(self.permutations)
        return t(image, Dx)



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
    def __init__(self, file_path, transform=None, num_samples=None):
        self.root = zarr.open(file_path, mode='r')
        self.filled_images_ds = self.root['filled_images']['filled_images']
        self.targets_ds_x = self.root['dispersion_results']['Dx']
        # self.targets_ds_y = self.root['dispersion_results']['Dy']

        self.transform = transform

        if num_samples is not None:
            self.len = min(num_samples, self.root.attrs['N'])

    def __len__(self):
        return self.len if hasattr(self, 'len') else self.filled_images_ds.shape[0]

    def __getitem__(self, idx):
        # Fetch numpy versions of the data
        image = self.filled_images_ds[idx]
        Dx = self.targets_ds_x[idx][0] # shape (1,5,2,2)
        # Dy = self.targets_ds_y[idx].reshape(5,4)

        # turn into torch tensors
        image = torch.from_numpy(image).float().unsqueeze(0)  # add channel dimension
        Dx = torch.from_numpy(Dx).float().flatten()
        # Dy = torch.from_numpy(Dy).float()
        
        # transform if needed
        if self.transform:
            image, Dx = self.transform(image, Dx)

        return image, Dx
class DispersionDataset_2(Dataset):
    def __init__(self, file_path, transform=None, num_samples=None):
        self.root = zarr.open(file_path, mode='r')
        self.filled_images_ds = self.root['filled_images']['filled_images']
        self.targets_ds_x = self.root['dispersion_results']['Dx']
        self.transform = transform
        if num_samples is not None:
            self.len = min(num_samples, len(self.filled_images_ds))

    def __len__(self):
        return self.len if hasattr(self, 'len') else self.filled_images_ds.shape[0]

    def __getitem__(self, idx):
        image = self.filled_images_ds[idx]
        Dx = self.targets_ds_x[idx] # shape (1,5,2,2)
        image = torch.from_numpy(image).float().unsqueeze(0)  # add channel dimension
        Dx = torch.from_numpy(Dx).float()
        return image, Dx

class DispersionDataset_single_view(Dataset):
    def __init__(self, base_dataset):
        """
        Wrapper that expands a dataset where targets come in slices of 5
        into individual samples (one target per sample).
        
        Args:
            base_dataset: Instance of DispersionDataset_2
        """
        self.base_dataset = base_dataset
        self.samples_per_base = 5  # Number of views/slices per base sample
        self.pe_hash = [0.1,10,50,100,500]
        
    def __len__(self):
        return len(self.base_dataset) * self.samples_per_base
    
    def __getitem__(self, idx):
        # Calculate which base sample and which slice within that sample
        base_idx = idx // self.samples_per_base
        slice_idx = idx % self.samples_per_base
        
        # Get the full sample from base dataset
        image, Dx = self.base_dataset[base_idx]
        
        # Extract the single view: Dx has shape (1, 5, 2, 2)
        # We want to get one of the 5 slices
        # print(f"Base idx: {base_idx}, Slice idx: {slice_idx}")
        # print(f"Dx shape before slicing: {Dx.shape}")
        Dx_single = Dx[slice_idx].flatten()  # Shape: (1, 2, 2)
        # Dx_single = torch.arcsinh(Dx[slice_idx].flatten())  # Shape: (1, 2, 2)
        Pe = torch.tensor([self.pe_hash[slice_idx]], dtype=torch.float32)  # Peclet number as float tensor
        return image, Dx_single, Pe



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

class DispersionDatasetCached(Dataset):
    """
    Ultra-optimized version: cache images in RAM since they're reused 5x.
    Use this if you have enough RAM (total_images * image_size * 4 bytes).
    """
    def __init__(self, file_path, transform=None, num_samples=None, cache_images=True):
        self.root = zarr.open(file_path, mode='r')
        self.transform = transform
        
        self.pe_values = torch.tensor([0.1, 10, 50, 100, 500], dtype=torch.float32)
        self.samples_per_image = 5
        
        # Determine length
        base_len = self.root['filled_images']['filled_images'].shape[0]
        if num_samples is not None:
            base_len = min(num_samples, base_len)
        
        self.base_len = base_len
        self.total_len = base_len * self.samples_per_image
        
        # Cache images in RAM
        if cache_images:
            print(f"Caching {base_len} images in RAM...")
            np_images = self.root['filled_images']['filled_images'][:base_len]
            np_targets = self.root['dispersion_results']['Dx'][:base_len]
            
            self.images = torch.from_numpy(np_images).float()          # Shape e.g. (N, H, W) or (N, C, H, W)
            self.targets_ds_x = torch.from_numpy(np_targets).float()   # Shape (N, 5, ...)
            
            # If images lack channel dim, add it once here
            if self.images.ndim == 3:  # (N, H, W)
                self.images = self.images.unsqueeze(1)  # -> (N, 1, H, W)
    
            print(f"Cached {(self.images.nbytes + self.targets_ds_x.nbytes) / 1e9:.2f} GB of data")
        else:
            self.images = self.root['filled_images']['filled_images']
            self.targets_ds_x = self.root['dispersion_results']['Dx']
    
    def __len__(self):
        return self.total_len
    
    def __getitem__(self, idx):
        base_idx = idx // self.samples_per_image
        pe_idx = idx % self.samples_per_image
        
        # Fast numpy array access (from RAM if cached, else from disk)
        # image = self.images[base_idx]
        # Dx = self.targets_ds_x[base_idx, pe_idx]
        
        # # Convert to tensors
        # image = torch.from_numpy(image).float().unsqueeze(0)
        # Dx_single = torch.from_numpy(Dx).float().flatten()
        # Pe = self.pe_values[pe_idx:pe_idx+1]
        
        # if self.transform:
        #     image = self.transform(image)
        image = self.images[base_idx]          # View, zero-copy
        Dx = self.targets_ds_x[base_idx, pe_idx]
        Dx_single = Dx.flatten()
        Pe = self.pe_values[pe_idx:pe_idx+1]

        if self.transform:
            image = self.transform(image)      # Transform expects torch tensor
        
        return image, Dx_single, Pe

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
    pin_memory = config.get('pin_memory',True)
    pin_memory_device = config.get('pin_memory_device','')
    prefetch_factor = config.get('prefetch_factor',None)

    train_path = os.path.join(file_path,'train.zarr')
    val_path = os.path.join(file_path,'validation.zarr')
    test_path = os.path.join(file_path,'test.zarr')

    if config.get('pe_encoder',None):
        print(f"Pe encoder: {config['pe_encoder']}")
        # base_train_dataset = DispersionDataset_2(train_path,num_samples=config.get('num_training_samples',None))
        # base_val_dataset = DispersionDataset_2(val_path,num_samples=config.get('num_validation_samples',None))
        # base_test_dataset = DispersionDataset_2(test_path)
        # train_dataset = DispersionDataset_single_view(base_train_dataset)
        # val_dataset = DispersionDataset_single_view(base_val_dataset)
        # test_dataset = DispersionDataset_single_view(base_test_dataset)
        train_dataset = DispersionDatasetCached(train_path,num_samples=config.get('num_training_samples',None),cache_images=False)
        val_dataset = DispersionDatasetCached(val_path,num_samples=config.get('num_validation_samples',None),cache_images=False)
        test_dataset = DispersionDatasetCached(test_path,cache_images=False)
    else:
        train_dataset = DispersionDataset(train_path,num_samples=config.get('num_training_samples',None))#, transform=DispersionTransform())
        val_dataset = DispersionDataset(val_path,num_samples=config.get('num_validation_samples',None))
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
                            num_workers=0,
                            persistent_workers=False,
                            pin_memory=False,
                            prefetch_factor=None,
                            pin_memory_device=pin_memory_device)
    test_loader = DataLoader(test_dataset, 
                             batch_size=batch_size, 
                             shuffle=False, 
                             num_workers=0,
                             persistent_workers=False,
                            pin_memory=False,
                            prefetch_factor=None,
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
