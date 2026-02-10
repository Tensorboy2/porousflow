
import zarr
from torch.utils.data import Dataset
import torch
torch.manual_seed(0)

class PermeabilityDataset(Dataset):
    def __init__(self, file_path, transform=None, num_samples=None):
        self.root = zarr.open(file_path, mode='r')
        self.filled_images_ds = self.root['filled_images']['filled_images']
        self.targets_ds = self.root['lbm_results']['K']

        self.transform = transform
        self.len = self.filled_images_ds.shape[0]
        if num_samples is not None:
            self.len = min(self.len,num_samples)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        # Fetch numpy versions of the data
        image = self.filled_images_ds[idx]
        K = self.targets_ds[idx].flatten()

        # turn into torch tensors
        image = torch.from_numpy(image).float().unsqueeze(0)  # add channel dimension
        K = torch.from_numpy(K).float()/8e-10  # convert to mD
        
        # transform if needed
        if self.transform:
            image, K = self.transform(image, K)

        return image, K
    
if __name__ == '__main__':
    dataset = PermeabilityDataset(file_path='data/train.zarr')
    print(f"Permeability dataset with len: {len(dataset)}")
    # root = zarr.open('data/train.zarr', mode='r')
    # targets_ds = np.amax(root['lbm_results']['K'][:])
    # print(f"Max value {targets_ds/9.187899330242999e-10}")


class DispersionDataset(Dataset):
    def __init__(self, file_path, transform=None, num_samples=None, Pe=0):
        self.root = zarr.open(file_path, mode='r')
        self.filled_images_ds = self.root['filled_images']['filled_images']
        self.targets_ds_x = self.root['dispersion_results']['Dx']
        self.targets_ds_y = self.root['dispersion_results']['Dy']
        self.Pe = Pe
        self.pe_hash = [0.1,10,50,100,500]
        self.pe_out = torch.tensor([self.pe_hash[Pe]], dtype=torch.float32)
        self.transform = transform

        if num_samples is not None:
            self.len = min(num_samples, self.root.attrs['N'])

    def __len__(self):
        return self.len if hasattr(self, 'len') else self.filled_images_ds.shape[0]

    def __getitem__(self, idx):
        # Fetch numpy versions of the data
        image = self.filled_images_ds[idx]
        Dx = self.targets_ds_x[idx][self.Pe] # shape (1,5,2,2)
        Dy = self.targets_ds_y[idx][self.Pe]

        # turn into torch tensors
        image = torch.from_numpy(image).float().unsqueeze(0)  # add channel dimension
        Dx = torch.from_numpy(Dx).float().flatten()
        Dy = torch.from_numpy(Dy).float().flatten()

        # transform if needed
        if self.transform:
            image, Dx = self.transform(image, Dx, Dy)
            # _, D
        else:
            D = torch.tensor([Dx[0],Dx[3]]).float()

        return image, D, self.Pe 
if __name__ == '__main__':
    dataset = DispersionDataset(file_path='data/train.zarr',Pe=2)
    print(f"Dispersion dataset with len: {len(dataset)}") 

    
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
            np_targets_2 = self.root['dispersion_results']['Dy'][:base_len]
            
            self.images = torch.from_numpy(np_images).float() .unsqueeze(0)         # Shape e.g. (N, H, W) or (N, C, H, W)
            self.targets_ds_x = torch.from_numpy(np_targets).float().flatten()   # Shape (N, 5, ...)
            self.targets_ds_y = torch.from_numpy(np_targets_2).float().flatten()   # Shape (N, 5, ...)
            
            # If images lack channel dim, add it once here
            # if self.images.ndim == 3:  # (N, H, W)
            # self.images = self.images  # -> (N, 1, H, W)
    
            print(f"Cached {(self.images.nbytes + self.targets_ds_x.nbytes) / 1e9:.2f} GB of data")
        else:
            self.images = self.root['filled_images']['filled_images']
            self.targets_ds_x = self.root['dispersion_results']['Dx']
            self.targets_ds_y = self.root['dispersion_results']['Dy']
    
    def __len__(self):
        return self.total_len
    
    def __getitem__(self, idx):
        base_idx = idx // self.samples_per_image
        pe_idx = idx % self.samples_per_image
        
        # Fast numpy array access (from RAM if cached, else from disk)
        image = self.images[base_idx]
        Dx = self.targets_ds_x[base_idx, pe_idx]
        Dy = self.targets_ds_y[base_idx, pe_idx]
        
        # Convert to tensors
        image = torch.from_numpy(image).float().unsqueeze(0)
        Dx = torch.from_numpy(Dx).float().flatten()
        Dy = torch.from_numpy(Dy).float().flatten()
        Pe = self.pe_values[pe_idx:pe_idx+1]
        
        if self.transform:
            image, D = self.transform(image, Dx, Dy)
        else:
            D = torch.tensor([Dx[0],Dx[3]]).float()
        return image, D, Pe
    
class DispersionDatasetFull(Dataset):
    """
    Ultra-optimized version: cache images in RAM since they're reused 10x.
    Each base image produces 10 samples: 5 Pe values × 2 directions (x/y).
    """
    def __init__(self, file_path, transform=None, num_samples=None):
        self.root = zarr.open(file_path, mode='r')
        self.transform = transform
        
        self.pe_values = torch.tensor([0.1, 10, 50, 100, 500], dtype=torch.float32)
        self.directions = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        
        self.num_pe = len(self.pe_values)          # 5
        self.num_dir = self.directions.shape[0]    # 2
        self.samples_per_image = self.num_pe * self.num_dir  # 10
        
        # Determine length
        base_len = self.root['filled_images']['filled_images'].shape[0]
        if num_samples is not None:
            base_len = min(num_samples, base_len)
        
        self.base_len = base_len
        self.total_len = base_len * self.samples_per_image
        
        # Cached array references (Zarr loads on-demand but keeps in RAM if possible)
        self.images = self.root['filled_images']['filled_images']
        self.Dx_array = self.root['dispersion_results']['Dx']   # shape: [N, 5]
        self.Dy_array = self.root['dispersion_results']['Dy']   # shape: [N, 5]
    
    def __len__(self):
        return self.total_len
    
    def __getitem__(self, idx):
        # Map global idx → base image + Pe + direction
        base_idx = idx // self.samples_per_image
        inner_idx = idx % self.samples_per_image
        
        pe_idx = inner_idx % self.num_pe
        dir_idx = inner_idx // self.num_pe   # 0 or 1
        
        # Load data
        image_np = self.images[base_idx]                     # numpy array
        if dir_idx == 0:  # x-direction
            target_np = self.Dx_array[base_idx, pe_idx]
            direction = self.directions[0]
        else:             # y-direction
            target_np = self.Dy_array[base_idx, pe_idx]
            direction = self.directions[1]
        
        # Convert to tensors
        image = torch.from_numpy(image_np).float().unsqueeze(0)  # add channel dim
        target = torch.from_numpy(target_np).float()
        if target.ndim > 0:
            target = target.flatten()  # ensure 1D vector
        pe = self.pe_values[pe_idx].unsqueeze(0)  # shape (1,)
        
        if self.transform:
            # Assuming transform takes (image, target) and returns the same
            image, target = self.transform(image, target)
        
        return image, target, pe, direction

if __name__ == '__main__':
    ds = DispersionDatasetFull('data/train.zarr')
    img, tgt, pe, dir_vec = ds[0]
    print(f"Dataset ready — len = {len(ds):,}")
    print(f"Content: image {img.shape} | D {tgt.shape} | pe {pe} | dir {dir_vec}")

