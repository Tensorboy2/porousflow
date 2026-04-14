from src.porousflow.lbm.lbm import LBM_solver
from src.porousflow.media_generator.utils.binary_blobs import periodic_binary_blobs
import time
if __name__ == "__main__":
    # jit warmup:
    geometry = periodic_binary_blobs(n_dim=2, length=32,volume_fraction=0.4,blob_size_fraction=0.1, seed=9)
    results = LBM_solver(solid=geometry, max_iterations=1)

    sizes = [128, 256, 512, 1024, 2048, 4096]
    for s in sizes:
        start= time.time()
        geometry = periodic_binary_blobs(n_dim=2, length=s,volume_fraction=0.4,blob_size_fraction=0.1, seed=9)
        results = LBM_solver(solid=geometry, max_iterations=500)
        print(f"Time taken for 4 iteration size {s}: ", time.time() - start)