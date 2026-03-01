import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI
from pathlib import Path
from src.porousflow.media_generator.utils.precolation_check import detect_percolation, fill_non_percolating_fluid_periodic
from scipy.ndimage import gaussian_filter
from src.porousflow.lbm.lbm import LBM_solver

# ---------------------------------------------------------------------------
# MPI setup
# ---------------------------------------------------------------------------
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

CACHE_DIR = Path('thesis_plots/cache')

# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------
def cache_path(sizes, N):
    """Deterministic cache filename based on run parameters."""
    size_tag = f"s{sizes[0]}-{sizes[-1]}-n{len(sizes)}"
    return CACHE_DIR / f"rev_N{N}_{size_tag}.npz"

def save_cache(path, K_all, phi_all, sizes):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, K_all=K_all, phi_all=phi_all, sizes=sizes)
    print(f"[Rank 0] Cache saved to {path}")

def load_cache(path):
    data = np.load(path)
    print(f"[Rank 0] Loaded cache from {path}")
    return data['K_all'], data['phi_all'], data['sizes']

# ---------------------------------------------------------------------------
# Domain generation
# ---------------------------------------------------------------------------
def periodic_binary_blobs(n_dim=2, length=128, volume_fraction=0.4,
                           blob_size_fraction=0.1, seed=None):
    base_length = 128
    sigma = 0.25 * base_length * blob_size_fraction
    base_n_pts = max(int(1.0 / blob_size_fraction) ** n_dim, 1)
    scale = (length / base_length) ** n_dim
    n_pts = max(int(base_n_pts * scale), 1)
    rs = np.random.default_rng(seed)
    shape = (length,) * n_dim
    mask = np.zeros(shape)
    points = (length * rs.random((n_dim, n_pts))).astype(int)
    mask[tuple(points)] = 1
    mask = gaussian_filter(mask, sigma=sigma, mode="wrap")
    threshold = np.percentile(mask, 100 * (1 - volume_fraction))
    return mask >= threshold

def generate_all_sizes_for_seed(sizes, seed):
    imgs = []
    for length in sizes:
        img = periodic_binary_blobs(length=length, seed=seed)
        x, y, _ = detect_percolation(img)
        if not (x and y):
            return None
        img = fill_non_percolating_fluid_periodic(img)
        imgs.append(img)
    return imgs

def find_percolating_set(sizes, base_seed=0, max_tries=100000):
    for k in range(max_tries):
        seed = base_seed + k
        imgs = generate_all_sizes_for_seed(sizes, seed)
        if imgs is not None:
            return imgs, seed
    raise RuntimeError("No global percolating seed found")

def find_N_percolating_sets(sizes, N, base_seed=0, max_tries=100000):
    all_imgs = []
    seeds_used = []
    current_seed = base_seed
    while len(all_imgs) < N:
        if current_seed - base_seed > max_tries:
            raise RuntimeError(f"Could only find {len(all_imgs)}/{N} percolating seeds")
        imgs, seed = find_percolating_set(sizes, base_seed=current_seed, max_tries=max_tries)
        all_imgs.append(imgs)
        seeds_used.append(seed)
        current_seed = seed + 1
    return all_imgs, seeds_used

# ---------------------------------------------------------------------------
# LBM
# ---------------------------------------------------------------------------
def run_lbm(imgs):
    K = np.zeros((len(imgs), 2, 2))
    for i, img in enumerate(imgs):
        results_x = LBM_solver(img, force_dir=0, max_iterations=10_000, tau=0.6)
        results_y = LBM_solver(img, force_dir=1, max_iterations=10_000, tau=0.6)
        K[i] = np.array([[results_x[1], results_x[2]], [results_y[2], results_y[1]]])
    return K

def compute_porosity(imgs):
    return np.array([np.sum(~img) / img.size for img in imgs])

# ---------------------------------------------------------------------------
# Plotting (rank 0 only)
# ---------------------------------------------------------------------------
from plotting.ploting import figsize
def plot_K_with_errorbars(K_all, sizes, phi_all=None):
    K_mean = K_all.mean(axis=0)
    K_std  = K_all.std(axis=0)

    n_rows = 4 
    fig, ax = plt.subplots(n_rows, 1, sharex=True, figsize=(figsize[0], figsize[1]*1.5))

    components = [
        (0, 0, 'C0', r'$K_{xx}$'),
        (0, 1, 'C1', r'$K_{xy}$'),
        (1, 1, 'C3', r'$K_{yx}$'),
        (1, 0, 'C2', r'$K_{yy}$'),
    ]
    for idx, (i, j, color, label) in enumerate(components):
        ax[idx].plot(sizes, K_mean[:, i, j], c=color, label=label)
        ax[idx].fill_between(
            sizes,
            K_mean[:, i, j] - K_std[:, i, j],
            K_mean[:, i, j] + K_std[:, i, j],
            alpha=0.3, color=color, label='±1 std'
        )
        ax[idx].vlines(x=128,
                       ymin=K_mean[:, i, j].min(),
                       ymax=K_mean[:, i, j].max(),
                       linestyle='dashed', color='k', alpha=0.5)
        ax[idx].set_ylabel('Permeability')
        ax[idx].legend()
        ax[idx].grid(alpha=0.3)

    # if phi_all is not None:
    #     phi_mean = phi_all.mean(axis=0)
    #     phi_std  = phi_all.std(axis=0)
    #     ax[4].plot(sizes, phi_mean, c='C4', label=r'$\phi$')
    #     ax[4].fill_between(sizes, phi_mean - phi_std, phi_mean + phi_std,
    #                        alpha=0.3, color='k', label='±1 std')
    #     ax[4].set_ylabel('Porosity')
    #     ax[4].legend()
    #     ax[4].grid(alpha=0.3)

    ax[-1].set_xlabel('Domain size (px)')
    plt.tight_layout()
    plt.savefig('thesis_plots/rev_errorbars.pdf')
    print("Saved to thesis_plots/rev_errorbars.pdf")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    sizes = np.linspace(16, 256, 20, dtype=int)
    N = 16

    # ------------------------------------------------------------------
    # Rank 0 checks cache, broadcasts whether computation is needed
    # ------------------------------------------------------------------
    skip_compute = True
    K_all = phi_all = None

    if rank == 0:
        cpath = cache_path(sizes, N)
        if cpath.exists():
            K_all, phi_all, sizes = load_cache(cpath)
            skip_compute = True
        else:
            print(f"No cache found at {cpath}, running computation...")

    skip_compute = comm.bcast(skip_compute, root=0)

    if skip_compute:
        # Only rank 0 has data — just plot and exit
        if rank == 0:
            plot_K_with_errorbars(K_all, sizes, phi_all=phi_all)
        comm.Barrier()  # keep all ranks alive until rank 0 finishes
    else:
        # ------------------------------------------------------------------
        # Full computation
        # ------------------------------------------------------------------
        SEED_BLOCK = 10_000
        rank_base_seed = rank * SEED_BLOCK
        counts = [(N // size) + (1 if r < N % size else 0) for r in range(size)]
        local_N = counts[rank]

        if rank == 0:
            print(f"Running {N} realizations across {size} MPI ranks")
            print(f"Realizations per rank: {counts}")

        local_imgs, local_seeds = find_N_percolating_sets(
            sizes, N=local_N, base_seed=rank_base_seed
        )
        print(f"[Rank {rank}] Found seeds: {local_seeds}")

        local_K   = np.zeros((local_N, len(sizes), 2, 2))
        local_phi = np.zeros((local_N, len(sizes)))
        for n, imgs in enumerate(local_imgs):
            print(f"[Rank {rank}] LBM realization {n+1}/{local_N}...")
            local_K[n]   = run_lbm(imgs)
            local_phi[n] = compute_porosity(imgs)

        all_K   = comm.gather(local_K,   root=0)
        all_phi = comm.gather(local_phi, root=0)

        if rank == 0:
            K_all   = np.concatenate(all_K,   axis=0)
            phi_all = np.concatenate(all_phi, axis=0)
            save_cache(cache_path(sizes, N), K_all, phi_all, sizes)
            plot_K_with_errorbars(K_all, sizes, phi_all=phi_all)