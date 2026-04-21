from src.ml.trainer import Trainer
# from src.ml.models.resnet import 
# from src.ml.models.vit import 
from src.ml.models.convnext import load_convnext_model
from src.ml.models.vit import load_vit_model
from src.ml.models.resnet import load_resnet_model
from src.ml.models.SHViT import load_hierarchical_shvit
from src.ml.models.swin import load_swin_model
from src.ml.data_loader import get_permeability_dataloader, get_dispersion_dataloader
from src.ml.trainer import Trainer
from torch import optim
import torch
torch.manual_seed(0)
import torch.nn as nn
import yaml
import os
import argparse
import zarr
import numpy as np
from tqdm import tqdm
from sklearn.metrics import r2_score
import h5py
import torch.nn.functional as F


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

class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps # Added for numerical stability
    
    def forward(self, x, y):
        # Calculate MSE and take the square root
        loss = torch.sqrt(self.mse(x, y) + self.eps)
        return loss

    
loss_functions = {
    'mse': nn.MSELoss(),
    'rmse': RMSELoss(),
}

def zarr_writer(path,data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    root = zarr.open(path, mode='w')
    for key, values in data.items():
        root.create_dataset(name=key, data=np.array(values), dtype='f4')

from plotting.ploting import figsize
def similarity_plot(targets,preds,path,R2):
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
    })

    titles = [r'$K_{xx}$', r'$K_{xy}$', r'$K_{yx}$', r'$K_{yy}$']
    fig, ax = plt.subplots(1,2,figsize=(figsize[0],figsize[1]*0.8))
    index = [[0,3],[1,2]]
    targets=targets*8e-10
    preds = preds*8e-10
    for i in range(2):
        ax[i].plot([targets[:, index[i][1]].min(), targets[:, index[i][1]].max()], [targets[:, index[i][1]].min(), targets[:, index[i][1]].max()], 'k--', alpha=0.2)
        ax[i].plot(targets[:, index[i][0]], preds[:, index[i][0]],
                        linestyle='',
                        marker='o',
                        markerfacecolor='C9',
                        markeredgecolor='C9',
                        markersize=1.,
                        markeredgewidth=0.8,
                        alpha=0.7,
                    #   c='C0', 
                    #   alpha=0.5, 
                    #   s=1.5, 
                        # label=titles[index[i][0]])
                        label=rf"{titles[index[i][0]]}, $R^2=${R2[index[i][0]]:.5f}")
        ax[i].plot(targets[:, index[i][1]], preds[:, index[i][1]],
                        marker='o',
                        linestyle='',
                        markerfacecolor='C6',
                        markeredgecolor='C6',
                        markersize=1.,
                        markeredgewidth=0.8,
                        alpha=0.5,
                #    c='C1', alpha=0.5, s=1.5, 
                        label=rf"{titles[index[i][1]]}, $R^2=${R2[index[i][1]]:.5f}")
        

        ax[i].grid(alpha=0.3)
        ax[i].set_xlabel(r'Ground truth ($m^2$)')
        ax[i].legend(markerscale=4)
    ax[0].set_ylabel(r'Predicted ($m^2$)')
    plt.tight_layout()
    plt.savefig(f'thesis_plots/{path}_similarity_plot_v2.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

def similarity_plot_dispersion(targets, preds, path, R2, pe):
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib
    import matplotlib.colors as mcolors
    from matplotlib.gridspec import GridSpec

    plt.rcParams.update({
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
    })

    titles = [r'$D_{\parallel}$', r'$D_{\bot}$']
    fig = plt.figure(figsize=(figsize[0], figsize[1] * 0.8))
    outer = GridSpec(1, 2, width_ratios=[20, 0.4], wspace=0.05)
    inner = outer[0].subgridspec(1, 2, wspace=0.225)
    axs = [fig.add_subplot(inner[i]) for i in range(2)]
    cax = fig.add_subplot(outer[1])

    # PE coloring — map to integer indices for clean discrete bands
    pe_vals = pe[:, 0]
    unique_pe = np.array([0.1, 10, 50, 100, 500])
    n = len(unique_pe)
    # pe_idx = np.searchsorted(unique_pe, pe_vals)
    pe_idx = np.argmin(np.abs(pe_vals[:, None] - unique_pe[None, :]), axis=1)
    # print(pe_idx)

    cmap = matplotlib.colormaps.get_cmap('viridis')
    norm = mcolors.BoundaryNorm(np.arange(-0.5, n), cmap.N)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)

    for i in range(2):
        lo = min(targets[:, i].min(), preds[:, i].min())
        hi = max(targets[:, i].max(), preds[:, i].max())
        axs[i].plot([lo, hi], [lo, hi], 'k--', alpha=0.2)

        for j in range(5):
            mask = pe_idx == j
            color = sm.to_rgba(j)  # sample cmap correctly as float in [0,1]
            axs[i].plot(
                targets[mask, i],
                preds[mask, i],
                marker='o',
                linestyle='',
                markerfacecolor=color,
                markeredgecolor=color,
                markersize=0.5,
                markeredgewidth=0.5,
                alpha=0.7#/(j+1),
            )
        axs[i].grid(alpha=0.2)
        axs[i].set_xlabel(r'Ground truth')
        axs[i].set_yscale('log')
        axs[i].set_xscale('log')
        # axs[i].set_aspect('equal')
        # axs[i].set_title(rf"{titles[i]}, $R^2={R2[i]:.5f}$")

    axs[0].set_ylabel(r'Predicted')
    axs[0].annotate(fr'$D_\parallel$, $R^2={R2[0]:.5f}$',(0.05,0.9),xycoords='axes fraction',bbox=dict(boxstyle="round,pad=0.3", fc="white",alpha=0.5, ec="gray", lw=0.5))
    axs[1].annotate(fr'$D_\bot$, $R^2={R2[1]:.5f}$',(0.05,0.9),xycoords='axes fraction',bbox=dict(boxstyle="round,pad=0.3", fc="white",alpha=0.5, ec="gray", lw=0.5))

    # Colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label('Péclet number (Pe)')
    cbar.set_ticks(np.arange(n))
    cbar.set_ticklabels([str(v) for v in unique_pe])

    plt.savefig(f'thesis_plots/{path}_similarity_dispersion_plot_v2.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def run_test(model, test_loader, device, criterion, config=None):
    model.eval()
    model_name = config['model']['name']
    task = config['task']
    cache_path = preds_cache_path(model_name, task, 'test')

    if os.path.exists(cache_path):
        print(f"Loading test predictions from cache: {cache_path}")
        with np.load(cache_path, allow_pickle=True) as d:
            all_preds = d['preds']
            all_targets = d['targets']
            all_pe = d['pe'] if 'pe' in d.files and d['pe'].size else None
            average_loss = float(d['test_loss']) if 'test_loss' in d.files else None
            average_r2 = float(d['R2']) if 'R2' in d.files else None
            R2 = d['R2_components'].tolist() if 'R2_components' in d.files else []
        print(f"Loaded cached test results. Test Loss: {average_loss:.5f} | Test R2: {average_r2:.5f}")
    else:
        total_loss = 0.0
        all_preds, all_targets, all_pe = [], [], []
        has_pe = False

        with torch.no_grad():
            for batch in tqdm(test_loader, leave=True):
                if len(batch) == 2:
                    inputs, targets = batch
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                else:
                    inputs, targets, pe = batch
                    inputs, targets, pe = inputs.to(device), targets.to(device), pe.to(device)
                    outputs = model(inputs, pe)
                    outputs = torch.sinh(outputs)
                    # outputs = torch.exp(outputs)
                    all_pe.append(pe.cpu().numpy())
                    has_pe = True
                loss = criterion(outputs, targets)
                total_loss += loss.item() * inputs.size(0)
                all_preds.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        all_preds   = np.concatenate(all_preds,   axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        all_pe      = np.concatenate(all_pe, axis=0) if has_pe else None
        average_loss = total_loss / len(test_loader.dataset)
        average_r2   = r2_score(all_targets, all_preds)
        print(f"Test Loss: {average_loss:.5f} | Test R2: {average_r2:.5f}")

        # --- Per-component R² ---
        R2 = []
        n_components = all_targets.shape[1]
        print(all_targets.shape, all_preds.shape)
        for i in range(n_components):
            r2_i = r2_score(all_targets[:, i], all_preds[:, i])
            print(f"  Component {i} R²: {r2_i:.5f}")
            R2.append(r2_i)

        # --- Per-Pe R² (overall and per component) ---
        if all_pe is not None:
            # all_pe may be shape (N,) or (N,1) — flatten to 1D
            pe_vals = all_pe.squeeze() if all_pe.ndim > 1 else all_pe
            unique_pes = np.unique(pe_vals)
            print(f"\nPer-Pe R² ({len(unique_pes)} unique Pe values):")
            for pe_val in unique_pes:
                mask = pe_vals == pe_val
                r2_pe = r2_score(all_targets[mask], all_preds[mask])
                comp_strs = "  |  ".join(
                    f"C{i}: {r2_score(all_targets[mask, i], all_preds[mask, i]):.4f}"
                    for i in range(n_components)
                )
                print(f"  Pe={pe_val:>8.2f}  overall={r2_pe:.5f}  [{comp_strs}]")

        # Save cache
        try:
            np.savez_compressed(
                cache_path,
                preds=all_preds, targets=all_targets,
                pe=all_pe if all_pe is not None else np.array([]),
                test_loss=average_loss, R2=average_r2,
                R2_components=np.array(R2),
            )
            print(f"Saved test predictions cache: {cache_path}")
        except Exception as e:
            print(f"Warning: failed to save test cache {cache_path}: {e}")

    # Save results and plotting as before
    data = {
        'test_loss': average_loss,
        'R2_test': average_r2,
        'predictions': all_preds,
    }
    path = f'{config["model"]["name"]}_test'
    zarr_writer(f'test_results/{path}_results.zarr', data)
    if config['task']=='permeability':
        similarity_plot(all_targets, all_preds,path, R2)
    elif config['task']=='dispersion':
        similarity_plot_dispersion(all_targets, all_preds,path, R2, pe = all_pe)


def get_cache_filename(dataset_tag):
    """Generate cache filename for LBM results"""
    results_cache_dir = "lbm_cache/"
    return os.path.join(results_cache_dir, f"lbm_results_{dataset_tag}.h5")

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

def coarsegrain_to_128(x, out_h=128, out_w=128):
    """Coarsen input to 128x128 resolution"""
    x = F.adaptive_avg_pool2d(x, (out_h, out_w))
    return (x > 0.5).float()

def run_models(model, X):
    """Run model predictions"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_in = X.float().to(device)

    with torch.no_grad():
        model.eval()
        preds = model(X_in).cpu().numpy()
        print(f"Ran model predictions on input shape {X.shape}, got output shape {preds.shape}")

    return preds


def get_model_name(model):
    """Derive a readable model name for cache filenames."""
    name = getattr(model, 'name', None)
    if not name and hasattr(model, 'config') and isinstance(getattr(model, 'config'), dict):
        name = model.config.get('name')
    if not name:
        name = model.__class__.__name__
    return str(name)


def preds_cache_path(model_name, task, dataset_tag):
    cache_dir = os.path.join('test_results', 'preds_cache')
    os.makedirs(cache_dir, exist_ok=True)
    safe_name = f"{model_name}_{task}_{dataset_tag}_preds.npz"
    return os.path.join(cache_dir, safe_name)


def save_preds_cache(path, preds):
    np.savez_compressed(path, preds=preds)


def load_preds_cache(path):
    if not os.path.exists(path):
        return None
    with np.load(path) as data:
        return data['preds']

def compute_all_predictions(all_results, model, config, use_cache=True):
    """Compute model predictions for all datasets with optional caching.

    If `use_cache` is True (default) predictions are loaded from
    `test_results/preds_cache/{model_name}_{dataset_tag}_preds.npz` when present.
    """
    print(f"\n{'='*60}")
    print("Computing model predictions")
    print(f"{'='*60}")

    model_name = config['model']['name']#get_model_name(model)
    task = config['task']

    for tag, result in all_results.items():
        cache_path = preds_cache_path(model_name, task, tag) if use_cache else None

        if use_cache and os.path.exists(cache_path):
            preds = load_preds_cache(cache_path)
            print(f"Loaded predictions from cache {cache_path} for {tag}")
        else:
            print(f"\nRunning model on {tag}...")
            X_torch = result['X_torch']
            preds = run_models(model, X_torch)
            if use_cache:
                try:
                    save_preds_cache(cache_path, preds)
                    print(f"Saved predictions to cache {cache_path}")
                except Exception as e:
                    print(f"Warning: failed to save cache {cache_path}: {e}")

        all_results[tag]['preds'] = preds

    return all_results

def run_real_media_test(model, path_to_h5=None, config=None, dataset_tags=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Fall back to all known datasets if none specified
    if dataset_tags is None:
        dataset_tags = list(datasets.keys())  # ["smooth", "rocks", "rough"]

    all_results = {}
    print("Loading cached results for plotting...")
    for tag in dataset_tags:
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

    all_results = compute_all_predictions(all_results, model)

    print(all_results.keys())
    all_abs_error = {}
    all_porosity = {}
    # compute metrics and save results
    for tag, result in all_results.items():
        print(f"\nEvaluating results for {tag}...")
        print(f"Ground truth K shape: {result['K_gt'].shape} | Preds shape: {result['preds'].shape}")
        K_gt = result['K_gt'].reshape(-1,4)  # Flatten to 1D array for metric computation
        preds = result['preds']*8e-10  # Scale predictions back to physical units

        r2 = r2_score(K_gt, preds)
        mse = np.mean((K_gt - preds) ** 2)
        rmse = np.sqrt(mse)

        # component-wise R2:
        for i in range(K_gt.shape[1]):
            r2_i = r2_score(K_gt[:, i], preds[:, i])
            print(f"  Component {i} R2: {r2_i:.5f}")

        all_results[tag]['metrics'] = {
            'R2': r2,
            'MSE': mse,
            'RMSE': rmse
        }

        print(f"\nDataset: {tag} | R2: {r2:.5f} | MSE: {mse:.5e} | RMSE: {rmse:.5e}")
        similarity_plot(K_gt, preds, f'{tag}_{config["model"]["name"]}')

        # absolute error over porosity:
        porosity = 1-result['X_torch'].mean(dim=[-1,-2]).cpu().numpy()

        from skimage.measure import regionprops_table
        smoothness_values = []
        for i in range(result['X_torch'].shape[0]):
            props = regionprops_table(result['X_torch'].cpu().numpy().astype(int)[i,0], properties=['area', 'perimeter'])
            
            smoothness = props['perimeter'] / (2 * np.sqrt(np.pi * props['area'] + 1e-6))  # +1e-6 to avoid division by zero
            # smoothness_variance = np.var(smoothness)
            smoothness_values.append(smoothness)
            # print(f"Sample {i} | Smoothness (perimeter-to-area ratio): {smoothness}")
        smoothness_values = np.array(smoothness_values)
        abs_error = np.abs(K_gt - preds)
        all_abs_error[tag] = abs_error / (0.5*(np.abs(K_gt) + np.abs(preds)))  # relative absolute error
        all_porosity[tag] = (smoothness_values)/porosity
        # print(abs_error.shape, porosity.shape)

    # all_abs_error = np.concatenate(all_abs_error, axis=0)
    # all_porosity = np.concatenate(all_porosity, axis=0)
    error_plot(all_abs_error, all_porosity, config)
        
    porosity_histogram(all_results)

    return all_results

def porosity_histogram(all_results):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6.4,4.8))
    tag_colors = {
        'rocks': 'tab:blue',
        'rough': 'tab:orange',
        'smooth': 'tab:green',
    }
    for tag, result in all_results.items():
        porosity = 1-result['X_torch'].mean(dim=[-1,-2]).cpu().numpy()
        plt.hist(porosity, bins=50, alpha=0.5, label=tag, color=tag_colors[tag])
    plt.xlabel('Porosity')
    plt.ylabel('Frequency')
    plt.title('Porosity Distribution of Datasets')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'thesis_plots/porosity_histogram.png', dpi=300)
    plt.close()

def error_plot(abs_error, porosity, config):
    import matplotlib.pyplot as plt
    fig, ax= plt.subplots(2,2,figsize=(6.4,4.8))
    ax = ax.flatten()
    titles = [r'$K_{xx}$', r'$K_{xy}$', r'$K_{yx}$', r'$K_{yy}$']
    tag_colors = {
        'rocks': 'tab:blue',
        'rough': 'tab:orange',
        'smooth': 'tab:green',
    }
    for i in range(4):
        for tag in abs_error.keys():
            ax[i].scatter(porosity[tag], abs_error[tag][:,i], alpha=0.8, s=5, label=f'Component {i}', color=tag_colors[tag])
        ax[i].set_xlabel('Smoothness/Porosity')
        ax[i].set_ylabel('Relative Absolute Error')
        ax[i].set_title(titles[i])
        ax[i].grid(alpha=0.3)

    handles = [plt.Line2D([0], [0], marker='o', color='w', label=tag, markerfacecolor=color, markersize=8) for tag, color in tag_colors.items()]
    fig.legend(handles=handles, loc='upper right')
    plt.tight_layout()
    plt.savefig(f'thesis_plots/{config["model"]["name"]}_error_vs_porosity.png', dpi=300)
    plt.close(fig)

def main(config,pretrained_path=None):
    '''
    Docstring for main
    
    :param config: Run specific config dictionary
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # clear cache:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Load model from config dict. The generator emits a structured `config['model']`.
    model_cfg = config.get('model')
    if model_cfg is None:
        raise ValueError("Config must contain a 'model' section")

    model_type = model_cfg.get('type')
    task = config.get('task', 'permeability')
    pe_encoder = config.get('pe_encoder', None)
    include_direction = config.get('pe',{}).get('include_direction',False)
    if model_type == 'convnext':
        model = load_convnext_model(model_cfg,task=task,Pe_encoder=pe_encoder,include_direction=include_direction,pretrained_path=pretrained_path)
    elif model_type == 'vit':
        model = load_vit_model(model_cfg,task=task,Pe_encoder=pe_encoder,include_direction=include_direction,pretrained_path=pretrained_path)
    elif model_type == 'resnet':
        model = load_resnet_model(model_cfg,task=task,Pe_encoder=pe_encoder,include_direction=include_direction,pretrained_path=pretrained_path)
    elif model_type == 'shvit':
        model = load_hierarchical_shvit(model_cfg,task=task,Pe_encoder=pe_encoder,include_direction=include_direction,pretrained_path=pretrained_path)
    elif model_type == 'swin':
        model = load_swin_model(model_cfg,task=task,Pe_encoder=pe_encoder,include_direction=include_direction,pretrained_path=pretrained_path)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Move to device
    try:
        model = model.to(device)
    except Exception:
        print("Warning: couldn't move model to device; continuing on CPU")

    print(f"Loaded model: {model_cfg.get('name', model_type)} | Type: {model_type}")

    # Setup data loaders
    batch_size = config.get('batch_size', 8)
    if task == 'permeability':
        train_loader, val_loader, test_loader = get_permeability_dataloader(file_path='data',config=config)
    elif task == 'dispersion':
        train_loader, val_loader, test_loader = get_dispersion_dataloader(file_path='data',config=config)
    else:
        raise ValueError(f"Unsupported task: {task}")
    # print(config)
    print(f"Data loaders set up for task: {task} | Batch size: {batch_size}")
    loss_function = loss_functions[config.get('loss_function','rmse')]

    run_test(model,test_loader,device,loss_function,config=config)
    # run_real_media_test(model, config=config, dataset_tags=args.datasets if hasattr(args, 'datasets') else None)
   
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model test with specified config")
    parser.add_argument('--model', type=str, default='convnext', help='Model name to test (e.g., resnet, vit, convnext)')
    parser.add_argument('--model_name', type=str, default='ConvNeXt-Atto', help='Specific model variant (e.g., ConvNeXt-Atto, ViT-T16, ResNet-18)')
    parser.add_argument('--size', type=str, default='atto', help='Model size (e.g., Atto, Small for convnext; T16, S16 for vit; 18, 50 for resnet)')
    parser.add_argument('--version', type=str, default='v1', help='Model version (e.g., v1, v2, RMS)')
    parser.add_argument('--task', type=str, default='permeability', help='Task to test on (e.g., permeability, dispersion)')
    parser.add_argument('--loss_function', type=str, default='mse', help='Loss function to use (e.g., mse, rmse)')
    parser.add_argument('--pe_encoder', type=str, default=None, help='PE encoder type (e.g., straight, log, vector)')
    parser.add_argument('--pretrained_path', type=str, default=None, help='Path to pretrained model weights (optional)')
    parser.add_argument(
                        '--datasets',
                        type=str,
                        nargs='+',
                        default=list(datasets.keys()),  # ["smooth", "rocks", "rough"]
                        help='Dataset tags to evaluate on (e.g., smooth rocks rough)'
                    )
    args = parser.parse_args()

    if args.task == 'permeability' and args.pe_encoder is not None:
        args.pe_encoder = None  # PE encoders are not used for permeability task
        print("Note: PE encoders are not applicable for permeability task. Ignoring --pe_encoder argument.")

    config = {
        'model' : {
            'in_channels': 1,
            'name': args.model_name,
            'type': args.model,
            'size': args.size,
            'version': args.version,
        },
        'batch_size': 128,
        'task': args.task,
        'pe_encoder': args.pe_encoder,
        'pe': {
            'pe_encoder': args.pe_encoder,
            "pe": 4,
            'include_direction': False,
        },
        'Pe': 4,
        'loss_function': args.loss_function,
    }

    main(config,pretrained_path=args.pretrained_path)