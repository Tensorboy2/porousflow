import zarr
import sys
import numpy as np

def init_lbm_zarr(path):
    """
    Initializes the Zarr groups and pre-allocates all LBM result datasets 
    to their final size.
    
    This function should be run once before starting parallel simulations.
    """

    root = zarr.open(path, mode='a')
    
    lbm_results = root.create_group('lbm_results',overwrite=True)

    
    N = root.attrs['N']
    print(f"Initializing LBM datasets for {N} samples...")
    W = root.attrs['W']
    H = root.attrs['H']
    
    velocity_shape = (N, H, W,2)
    velocity_chunks = (1, H, W,2)

    # Datasets for 2D field results
    lbm_results.create_dataset(
        name='ux_physical', 
        shape=velocity_shape, 
        chunks=velocity_chunks,
        dtype='f4',
        overwrite=True # Overwrite if existing dataset has wrong size
    )
    lbm_results.create_dataset(
        name='uy_physical', 
        shape=velocity_shape, 
        chunks=velocity_chunks,
        dtype='f4',
        overwrite=True
    )
    
    # Permeability Tensor K (shape: N, 2, 2)
    lbm_results.create_dataset(
        name='K',
        shape=(N, 2, 2),
        chunks=(1, 2, 2),
        dtype='f4',
        overwrite=True,
        fill_value=np.nan
    )
    
    # Scalar/Metric results (shape: N)
    scalar_shape = (N,)
    scalar_chunks = (1,)
    scalar_dtype = 'f4'

    # Scalar datasets
    lbm_results.create_dataset(name='iteration_x', shape=scalar_shape, chunks=scalar_chunks, dtype='i4', overwrite=True)
    lbm_results.create_dataset(name='iteration_y', shape=scalar_shape, chunks=scalar_chunks, dtype='i4', overwrite=True)
    lbm_results.create_dataset(name='Ma_x', shape=scalar_shape, chunks=scalar_chunks, dtype=scalar_dtype, overwrite=True)
    lbm_results.create_dataset(name='Ma_y', shape=scalar_shape, chunks=scalar_chunks, dtype=scalar_dtype, overwrite=True)
    lbm_results.create_dataset(name='Re_lattice_x', shape=scalar_shape, chunks=scalar_chunks, dtype=scalar_dtype, overwrite=True)
    lbm_results.create_dataset(name='Re_lattice_y', shape=scalar_shape, chunks=scalar_chunks, dtype=scalar_dtype, overwrite=True)
    lbm_results.create_dataset(name='dt', shape=scalar_shape, chunks=scalar_chunks, dtype=scalar_dtype, overwrite=True)
    lbm_results.create_dataset(name='tau', shape=scalar_shape, chunks=scalar_chunks, dtype=scalar_dtype, overwrite=True)
    lbm_results.create_dataset(name='Re_phys_x', shape=scalar_shape, chunks=scalar_chunks, dtype=scalar_dtype, overwrite=True)
    lbm_results.create_dataset(name='Re_phys_y', shape=scalar_shape, chunks=scalar_chunks, dtype=scalar_dtype, overwrite=True)
    lbm_results.create_dataset(name='dx', shape=scalar_shape, chunks=scalar_chunks, dtype=scalar_dtype, overwrite=True)
    lbm_results.create_dataset(name='F_lattice', shape=scalar_shape, chunks=scalar_chunks, dtype=scalar_dtype, overwrite=True)
    
    print("Initialization complete. Datasets are ready for parallel writing.")


def init_dispersion_zarr(path):
    root = zarr.open(path, mode='a')
    # root.attrs['Pe'] = 
    disp_results = root.create_group('dispersion_results',overwrite=True)

    N = root.attrs['N']
    print(f"Initializing LBM datasets for {N} samples...")
    W = root.attrs['W']
    H = root.attrs['H']

    disp_results.create_dataset(
        'Dx', 
        shape=(N, 5, 2, 2), 
        chunks=(1, 1, 2, 2), 
        dtype='f4', 
        overwrite=True,
        fill_value=np.nan
        )
    
    disp_results.create_dataset(
        'Dy', 
        shape=(N, 5, 2, 2), 
        chunks=(1, 1, 2, 2), 
        dtype='f4', 
        overwrite=True,
        fill_value=np.nan
        )
    
    # disp_results.create_dataset(
    #     'Pe',
    #     data = np.array([0,10,50,100,500]),
    #     shape=(5,),
    #     dtype='f4',
    #     chunks=(1,)
    # )
    print("Dispersion initialization complete.")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python init_zarr.py <data_type> <mode> <num_samples>")
        sys.exit(1)
        
    data_type = sys.argv[1]
    mode = sys.argv[2]
    
    # Construct the full Zarr path
    path = f"./data/{data_type}.zarr"
    
    if mode == 'lbm':
        init_lbm_zarr(path)
    elif mode == 'dispersion':
        init_dispersion_zarr(path)
    else:
        print(f"Unknown mode: {mode}")