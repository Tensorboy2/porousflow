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
POROSITIES = [0.8, 0.65]   # volume_fraction = 1 - porosity

# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------
def cache_path(sizes, N, porosity):
    size_tag = f"s{sizes[0]}-{sizes[-1]}-n{len(sizes)}"
    phi_tag  = f"phi{int(porosity*100):03d}"
    return CACHE_DIR / f"rev_N{N}_{size_tag}_{phi_tag}.npz"

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

def generate_all_sizes_for_seed(sizes, seed, volume_fraction):
    imgs = []
    for length in sizes:
        img = periodic_binary_blobs(length=length, seed=seed,
                                    volume_fraction=volume_fraction)
        x, y, _ = detect_percolation(img)
        if not (x and y):
            return None
        img = fill_non_percolating_fluid_periodic(img)
        imgs.append(img)
    return imgs

def find_percolating_set(sizes, volume_fraction, base_seed=0, max_tries=100000):
    for k in range(max_tries):
        seed = base_seed + k
        imgs = generate_all_sizes_for_seed(sizes, seed, volume_fraction)
        if imgs is not None:
            return imgs, seed
    raise RuntimeError("No global percolating seed found")

def find_N_percolating_sets(sizes, N, volume_fraction, base_seed=0, max_tries=100000):
    all_imgs = []
    seeds_used = []
    current_seed = base_seed
    while len(all_imgs) < N:
        if current_seed - base_seed > max_tries:
            raise RuntimeError(f"Could only find {len(all_imgs)}/{N} percolating seeds")
        imgs, seed = find_percolating_set(sizes, volume_fraction,
                                          base_seed=current_seed,
                                          max_tries=max_tries)
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

def plot_K_multi_porosity(results, sizes):
    """
    results : list of (porosity, K_all, phi_all) tuples,
              one per target porosity
    """
    components = [
        (0, 0, r'$K_{xx}$'),
        (0, 1, r'$K_{xy}$'),
        (1, 0, r'$K_{yx}$'),
        (1, 1, r'$K_{yy}$'),
    ]
    colors_per_phi = ['C0', 'C1', 'C2']   # one colour family per porosity

    fig, axes = plt.subplots(4, 1, sharex=True,
                             figsize=(figsize[0], figsize[1] * 1.25))

    for ax_idx, (i, j, comp_label) in enumerate(components):
        ax = axes[ax_idx]

        for phi_idx, (porosity, K_all, phi_all) in enumerate(results):
            color = colors_per_phi[phi_idx]
            K_mean = K_all.mean(axis=0)   # shape: (n_sizes, 2, 2)
            K_std  = K_all.std(axis=0)

            label = f"{comp_label}, φ={porosity:.2f}"
            ax.plot(sizes, K_mean[:, i, j], c=color, label=label)
            ax.fill_between(
                sizes,
                K_mean[:, i, j] - K_std[:, i, j],
                K_mean[:, i, j] + K_std[:, i, j],
                alpha=0.25, color=color,
            )

        ax.vlines(x=128,
                  ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1],
                  linestyle='dashed', color='k', alpha=0.5)
        ax.set_ylabel('Permeability')
        ax.legend(loc='upper right', fontsize='small')
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel('Domain size (px)')
    plt.tight_layout()
    plt.savefig('thesis_plots/rev_errorbars_multiporosity.pdf')
    print("Saved to thesis_plots/rev_errorbars_multiporosity.pdf")

# ---------------------------------------------------------------------------
# Per-porosity compute/cache helper
# ---------------------------------------------------------------------------
def get_or_compute(sizes, N, porosity, rank, size, comm):
    """Return (K_all, phi_all) for one porosity value, using cache when possible."""
    # volume_fraction of solid  = 1 - porosity
    volume_fraction = 1.0 - porosity

    skip_compute = False
    K_all = phi_all = None

    if rank == 0:
        cpath = cache_path(sizes, N, porosity)
        if cpath.exists():
            K_all, phi_all, _ = load_cache(cpath)
            skip_compute = True
        else:
            print(f"[Rank 0] No cache for φ={porosity}, running computation...")

    skip_compute = comm.bcast(skip_compute, root=0)

    if not skip_compute:
        SEED_BLOCK = 10_000
        # Offset seed block by porosity index so ranks don't collide across runs
        phi_offset  = int(porosity * 1000) * SEED_BLOCK * size
        rank_base_seed = phi_offset + rank * SEED_BLOCK

        counts    = [(N // size) + (1 if r < N % size else 0) for r in range(size)]
        local_N   = counts[rank]

        if rank == 0:
            print(f"  φ={porosity}: {N} realizations across {size} MPI ranks "
                  f"(local_N per rank: {counts})")

        local_imgs, local_seeds = find_N_percolating_sets(
            sizes, N=local_N, volume_fraction=volume_fraction,
            base_seed=rank_base_seed
        )
        print(f"[Rank {rank}] φ={porosity} seeds: {local_seeds}")

        local_K   = np.zeros((local_N, len(sizes), 2, 2))
        local_phi = np.zeros((local_N, len(sizes)))
        for n, imgs in enumerate(local_imgs):
            print(f"[Rank {rank}] φ={porosity} LBM realization {n+1}/{local_N}...")
            local_K[n]   = run_lbm(imgs)
            local_phi[n] = compute_porosity(imgs)

        all_K   = comm.gather(local_K,   root=0)
        all_phi = comm.gather(local_phi, root=0)

        if rank == 0:
            K_all   = np.concatenate(all_K,   axis=0)
            phi_all = np.concatenate(all_phi, axis=0)
            cpath   = cache_path(sizes, N, porosity)
            save_cache(cpath, K_all, phi_all, sizes)

    # Broadcast results to all ranks so they stay in sync
    K_all   = comm.bcast(K_all,   root=0)
    phi_all = comm.bcast(phi_all, root=0)
    return K_all, phi_all

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    sizes = np.linspace(16, 256, 20, dtype=int)
    N = 16

    results = []
    for porosity in POROSITIES:
        K_all, phi_all = get_or_compute(sizes, N, porosity, rank, size, comm)
        results.append((porosity, K_all, phi_all))

    if rank == 0:
        plot_K_multi_porosity(results, sizes)

    comm.Barrier()