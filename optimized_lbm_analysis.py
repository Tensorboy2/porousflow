import matplotlib.pyplot as plt
from src.porousflow.lbm.lbm import LBM_solver
import torch
import torch.nn.functional as F
import h5py
import numpy as np
from src.ml.models.vit import load_vit_model
from src.ml.models.convnext import load_convnext_model
from sklearn.metrics import mean_squared_error
import os
import argparse
from pathlib import Path

try:
    from mpi4py import MPI
    HAS_MPI = True
except ImportError:
    HAS_MPI = False
    print("Warning: mpi4py not available. Running in serial mode.")


# ===============================
# Config
# ===============================

device = "cuda" if torch.cuda.is_available() else "cpu"

L_physical = 1e-3
tau = 0.6
force_scaling = 1e-1

save_dir = "update_plots/"
results_cache_dir = "lbm_cache/"

datasets = {
    "smooth": "smooth_validation_2D.h5",
    "rocks": "rocks_validation.h5",
    "rough": "rough_validation_2D.h5",
}

component_names = ["Kxx", "Kxy", "Kyx", "Kyy"]

dataset_colors = {
    "rocks": "tab:blue",
    "rough": "tab:orange",
    "smooth": "tab:green"
}


# ===============================
# Helper Functions
# ===============================

def coarsegrain_to_128(x, out_h=128, out_w=128):
    """Coarsen input to 128x128 resolution"""
    x = F.adaptive_avg_pool2d(x, (out_h, out_w))
    return (x > 0.5).float()


def get_cache_filename(dataset_tag):
    """Generate cache filename for LBM results"""
    return os.path.join(results_cache_dir, f"lbm_results_{dataset_tag}.h5")


def save_lbm_results(dataset_tag, K_gt, flows, X_torch):
    """Save LBM results to HDF5 file"""
    os.makedirs(results_cache_dir, exist_ok=True)
    cache_file = get_cache_filename(dataset_tag)
    
    with h5py.File(cache_file, 'w') as f:
        f.create_dataset('K_gt', data=K_gt, compression='gzip')
        f.create_dataset('X_torch', data=X_torch.numpy(), compression='gzip')
        
        # Save flow fields (separate datasets for each sample)
        flow_grp = f.create_group('flows')
        for i, (u_mag, u_mag_2) in enumerate(flows):
            sample_grp = flow_grp.create_group(f'sample_{i}')
            sample_grp.create_dataset('u_mag', data=u_mag, compression='gzip')
            sample_grp.create_dataset('u_mag_2', data=u_mag_2, compression='gzip')
    
    print(f"Saved LBM results to {cache_file}")


def load_lbm_results(dataset_tag):
    """Load LBM results from HDF5 file"""
    cache_file = get_cache_filename(dataset_tag)
    
    if not os.path.exists(cache_file):
        return None
    
    with h5py.File(cache_file, 'r') as f:
        K_gt = f['K_gt'][:]
        X_torch = torch.tensor(f['X_torch'][:])
        
        flows = []
        flow_grp = f['flows']
        for i in range(len(K_gt)):
            sample_grp = flow_grp[f'sample_{i}']
            u_mag = sample_grp['u_mag'][:]
            u_mag_2 = sample_grp['u_mag_2'][:]
            flows.append((u_mag, u_mag_2))
    
    print(f"Loaded LBM results from {cache_file}")
    return K_gt, flows, X_torch


# ===============================
# LBM Simulation Functions
# ===============================

def run_lbm_single(x):
    """Run LBM for a single sample"""
    out = LBM_solver(
        x,
        max_iterations=100_000,
        L_physical=L_physical,
        tau=tau,
        force_strength=force_scaling
    )

    out_2 = LBM_solver(
        x,
        force_dir=1,
        max_iterations=100_000,
        L_physical=L_physical,
        tau=tau,
        force_strength=force_scaling
    )

    Kxx, Kxy = out[1], out[2]
    Kyx, Kyy = out_2[1], out_2[2]

    K = np.array([[Kxx, Kxy],
                  [Kyx, Kyy]])

    u_mag = np.linalg.norm(out[0], axis=-1)
    u_mag_2 = np.linalg.norm(out_2[0], axis=-1)

    return K, (u_mag, u_mag_2)


def run_lbm_batch_serial(X_batch):
    """Run LBM in serial mode"""
    K_list = []
    flow_list = []

    for i in range(len(X_batch)):
        x = X_batch[i, 0].numpy()
        K, flows = run_lbm_single(x)
        K_list.append(K)
        flow_list.append(flows)

        if i % 10 == 0:
            print(f"LBM progress: {i}/{len(X_batch)}")

    return np.stack(K_list), flow_list


def run_lbm_batch_mpi(X_batch):
    """Run LBM with MPI parallelization"""
    if not HAS_MPI:
        print("MPI not available, falling back to serial mode")
        return run_lbm_batch_serial(X_batch)
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Distribute work across MPI processes
    n_samples = len(X_batch)
    samples_per_rank = n_samples // size
    remainder = n_samples % size
    
    # Calculate start and end indices for this rank
    if rank < remainder:
        start_idx = rank * (samples_per_rank + 1)
        end_idx = start_idx + samples_per_rank + 1
    else:
        start_idx = rank * samples_per_rank + remainder
        end_idx = start_idx + samples_per_rank
    
    # Process assigned samples
    local_K_list = []
    local_flow_list = []
    
    for i in range(start_idx, end_idx):
        x = X_batch[i, 0].numpy()
        K, flows = run_lbm_single(x)
        local_K_list.append(K)
        local_flow_list.append(flows)
        
        if (i - start_idx) % 5 == 0:
            print(f"Rank {rank}: LBM progress {i - start_idx}/{end_idx - start_idx}")
    
    # Gather results from all ranks
    all_K_lists = comm.gather(local_K_list, root=0)
    all_flow_lists = comm.gather(local_flow_list, root=0)
    
    if rank == 0:
        # Combine results in correct order
        K_list = []
        flow_list = []
        
        for rank_idx in range(size):
            K_list.extend(all_K_lists[rank_idx])
            flow_list.extend(all_flow_lists[rank_idx])
        
        return np.stack(K_list), flow_list
    else:
        return None, None


# ===============================
# Model Prediction Functions
# ===============================

def load_models():
    """Load all trained models"""
    models = {
        "convnext_atto": load_convnext_model(
            'v1','atto',
            pretrained_path='results/convnext_atto/ConvNeXt-Atto_lr-0.0008_wd-0.1_bs-128_epochs-100_cosine_warmup-1250_clipgrad-True_pe-encoder-None_pe-None_mse.pth'
        ).to(device),

        "vit_B16_last_epoch": load_vit_model(
            'B16',
            pretrained_path='results/convnext_atto/ViT-B16_lr-0.0008_wd-0.1_bs-128_epochs-2000_cosine_warmup-250_clipgrad-True_pe-encoder-None_pe-None_last_model.pth'
        ).to(device)
    }
    return models


def run_models(models, X):
    """Run model predictions"""
    preds = {}
    X_in = X.float().to(device)

    with torch.no_grad():
        for name, model in models.items():
            model.eval()
            preds[name] = model(X_in).cpu().numpy()
            print(f"Ran model: {name}")

    return preds


# ===============================
# Main Processing Functions
# ===============================

def process_dataset(dataset_tag, file_name, use_mpi=False, force_recompute=False):
    """Process a single dataset - load or compute LBM results"""
    print(f"\n{'='*60}")
    print(f"Processing dataset: {dataset_tag}")
    print(f"{'='*60}")
    
    # Try to load cached results
    if not force_recompute:
        cached = load_lbm_results(dataset_tag)
        if cached is not None:
            K_gt, flows, X_torch = cached
            print(f"Using cached LBM results for {dataset_tag}")
            return K_gt, flows, X_torch
    
    # Load and preprocess input data
    print(f"Loading dataset from {file_name}...")
    with h5py.File(file_name, 'r') as f:
        X = f['input/fill'][:]
    
    X_torch = torch.tensor(X[:, :, 27:291])
    X_torch = coarsegrain_to_128(X_torch)
    print(f"Preprocessed X shape: {X_torch.shape}")
    
    # Run LBM simulation
    print("Running LBM simulation...")
    if use_mpi:
        K_gt, flows = run_lbm_batch_mpi(X_torch)
    else:
        K_gt, flows = run_lbm_batch_serial(X_torch)
    
    # Only save on rank 0 (or in serial mode)
    if K_gt is not None:
        save_lbm_results(dataset_tag, K_gt, flows, X_torch)
        print("LBM simulation complete.")
    
    return K_gt, flows, X_torch


def compute_all_predictions(all_results, models):
    """Compute model predictions for all datasets"""
    print(f"\n{'='*60}")
    print("Computing model predictions")
    print(f"{'='*60}")
    
    for tag, result in all_results.items():
        print(f"\nRunning models on {tag}...")
        X_torch = result['X_torch']
        preds = run_models(models, X_torch)
        all_results[tag]['preds'] = preds
    
    return all_results


# ===============================
# Plotting Functions
# ===============================

def plot_scatter_comparisons(all_results, models, save_dir):
    """Generate scatter plots comparing predictions to ground truth"""
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("Generating scatter plots")
    print(f"{'='*60}")
    
    for model_name in models.keys():
        fig, axes = plt.subplots(2, 2, figsize=(8, 8))
        axes = axes.flatten()

        for i, ax in enumerate(axes):
            lo_vals = []
            hi_vals = []

            for tag, result in all_results.items():
                K_gt = result["K_gt"]
                pred = result["preds"][model_name]

                K_gt_flat = K_gt.reshape(len(K_gt), 4)
                pred_flat = pred.reshape(len(pred), 4)

                if model_name == "convnext_atto":
                    scale = 8e-10
                else:
                    scale = 1e-9

                gt_vals = K_gt_flat[:, i] / scale
                pred_vals = pred_flat[:, i]

                ax.scatter(
                    gt_vals,
                    pred_vals,
                    s=12,
                    alpha=0.5,
                    color=dataset_colors[tag],
                    label=tag if i == 0 else None
                )

                lo_vals.append(gt_vals.min())
                hi_vals.append(gt_vals.max())
                lo_vals.append(pred_vals.min())
                hi_vals.append(pred_vals.max())

            lo = min(lo_vals)
            hi = max(hi_vals)

            ax.plot([lo, hi], [lo, hi], linestyle="--", color="black", alpha=0.6)

            ax.set_title(component_names[i])
            ax.set_xlabel("Ground truth")
            ax.set_ylabel("Prediction")
            # ax.set_yscale('log')
            # ax.set_xscale('log')

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right")

        fig.suptitle(f"Truth vs Prediction — {model_name}")
        plt.tight_layout()
        
        plot_file = f"{save_dir}/scatter_{model_name}.pdf"
        plt.savefig(plot_file)
        plt.close()
        print(f"Saved: {plot_file}")

    print("\nAll scatter plots saved.")


# ===============================
# Main Execution
# ===============================

def main():
    parser = argparse.ArgumentParser(description='Run LBM analysis with optional MPI parallelization')
    parser.add_argument('--mpi', action='store_true', help='Use MPI parallelization')
    parser.add_argument('--force-recompute', action='store_true', help='Force recomputation of LBM (ignore cache)')
    parser.add_argument('--plot-only', action='store_true', help='Only generate plots from cached results')
    parser.add_argument('--datasets', nargs='+', choices=list(datasets.keys()), 
                        default=list(datasets.keys()), help='Datasets to process')
    args = parser.parse_args()
    
    # Initialize MPI if requested
    if args.mpi and HAS_MPI:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
    else:
        rank = 0
    
    all_results = {}
    
    # Process datasets (LBM simulation or load cache)
    if not args.plot_only:
        for tag in args.datasets:
            file_name = datasets[tag]
            K_gt, flows, X_torch = process_dataset(
                tag, file_name, 
                use_mpi=args.mpi, 
                force_recompute=args.force_recompute
            )
            
            if rank == 0 or not args.mpi:
                all_results[tag] = {
                    "K_gt": K_gt,
                    "flows": flows,
                    "X_torch": X_torch
                }
    else:
        # Load all cached results for plotting
        print("Loading cached results for plotting...")
        for tag in args.datasets:
            cached = load_lbm_results(tag)
            if cached is None:
                print(f"ERROR: No cached results found for {tag}. Run without --plot-only first.")
                return
            K_gt, flows, X_torch = cached
            all_results[tag] = {
                "K_gt": K_gt,
                "flows": flows,
                "X_torch": X_torch
            }
    
    # Only rank 0 does plotting
    if rank == 0:
        # Load models and compute predictions
        print("\nLoading models...")
        models = load_models()
        
        all_results = compute_all_predictions(all_results, models)
        
        # Generate plots
        plot_scatter_comparisons(all_results, models, save_dir)
        
        print(f"\n{'='*60}")
        print("Analysis complete!")
        print(f"Results cached in: {results_cache_dir}")
        print(f"Plots saved in: {save_dir}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()