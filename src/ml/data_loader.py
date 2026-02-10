'''
Docstring for data_loader

This module implements the datasets and dataloaders needed to train neural networks on predicting both
permeability and dispersion from porous media images. It also implements the augmentations needed for training.
'''
import os
from torch.utils.data import DataLoader
import torch
torch.manual_seed(0)

if __name__ == "__main__":
    from dataaugmentations import PermeabilityTransform, DispersionTransform
    from datasets import PermeabilityDataset, DispersionDataset, DispersionDatasetCached, DispersionDatasetFull
else:
    from src.ml.dataaugmentations import PermeabilityTransform, DispersionTransform
    from src.ml.datasets import PermeabilityDataset, DispersionDataset, DispersionDatasetCached, DispersionDatasetFull


    
   





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

    train_dataset = PermeabilityDataset(train_path, transform=PermeabilityTransform(),num_samples=config.get('num_training_samples'))
    val_dataset = PermeabilityDataset(val_path,num_samples=config.get('num_validation_samples'))
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
    pin_memory = config.get('pin_memory',True)
    pin_memory_device = config.get('pin_memory_device','')
    prefetch_factor = config.get('prefetch_factor',None)

    train_path = os.path.join(file_path,'train.zarr')
    val_path = os.path.join(file_path,'validation.zarr')
    test_path = os.path.join(file_path,'test.zarr')
    
    pe = config['pe']
    aug = DispersionTransform() if config.get("transform",True) else None
    if pe['pe_encoder']:
        # print(f"Pe encoder: {pe['pe_encoder']}")
        if pe['include_direction']:
            train_dataset = DispersionDatasetFull(train_path,num_samples=config.get('num_training_samples',None),transform=aug)
            val_dataset = DispersionDatasetFull(val_path,num_samples=config.get('num_validation_samples',None))
            test_dataset = DispersionDatasetFull(test_path)
        else:
            train_dataset = DispersionDatasetCached(train_path,num_samples=config.get('num_training_samples',None),cache_images=False,transform=aug)
            val_dataset = DispersionDatasetCached(val_path,num_samples=config.get('num_validation_samples',None),cache_images=False)
            test_dataset = DispersionDatasetCached(test_path,cache_images=False)
    else:
        Pe = config.get('Pe',0)
        train_dataset = DispersionDataset(train_path,num_samples=config.get('num_training_samples',None),Pe=Pe, transform=aug)
        val_dataset = DispersionDataset(val_path,num_samples=config.get('num_validation_samples',None),Pe=Pe)
        test_dataset = DispersionDataset(test_path,Pe=Pe)

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
    configs = [
        {'pe':{'pe_encoder':False, 'pe':0, 'include_direction':False}},
        {'pe':{'pe_encoder':True, 'pe':0, 'include_direction':False}},
        {'pe':{'pe_encoder':True, 'pe':0, 'include_direction':True}},
        ]
    for config in configs:
        t,v,_ = get_dispersion_dataloader('data', config=config)
        for i, batch in enumerate(t):
            print(f'{config}, {len(t)}, {len(batch)}')
            break

    # # Example usage
    # import time, psutil, os
    # config = {
    #     'batch_size': 1024,
    #     'num_workers': 0,
    #     # 'prefetch_factor': 8,
    # }

    # proc = psutil.Process(os.getpid())
    # start_mem = proc.memory_info().rss / 1e6  # MB
    # start = time.time()

    # train_loader, val_loader, test_loader = get_dispersion_dataloader('data', config)
    # for i, (image, target) in enumerate(train_loader):
    #     print(f"Batch {i}: Image batch shape: {image.shape}, Target batch shape: {target.shape}")
    #     if i >= 20:  # first 20 batches
    #         break

    # end = time.time()
    # end_mem = proc.memory_info().rss / 1e6

    # print(f"Time per batch: {(end-start)/20:.3f} s, RAM usage increase: {end_mem-start_mem:.2f} MB")
