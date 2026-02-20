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

def similarity_plot(targets,preds,path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2,2,figsize=(6.4,6.4))
    ax = ax.flatten()
    titles = [r'$K_{xx}$', r'$K_{xy}$', r'$K_{yx}$', r'$K_{yy}$']
    for i in range(targets.shape[1]):
        ax[i].scatter(targets[:, i], preds[:, i], alpha=0.5, s=1.5, label=f'Component {i}')
        ax[i].plot([targets[:, i].min(), targets[:, i].max()], [targets[:, i].min(), targets[:, i].max()], 'k--', alpha=0.3)
        # ax[i].legend()
        ax[i].set_xlabel('True Values')
        ax[i].set_ylabel('Predicted Values')
        # if i == 0 or i == 3:
        #     ax[i].set_yscale('log')
        #     ax[i].set_xscale('log')


        ax[i].set_title(titles[i])
        ax[i].set_aspect('equal', adjustable='box')
        ax[i].grid(alpha=0.3)
    # plt.title('Similarity Plot: True vs Predicted')
    plt.tight_layout()
    plt.savefig(f'thesis_plots/{path}_similarity_plot.png', dpi=300)
    plt.close(fig)

def run_test(model, test_loader, device, criterion,config=None):
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(test_loader):
            # print(batch)
            if len(batch) == 2:
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
            else:
                inputs, targets, pe = batch
                inputs = inputs.to(device)
                targets = targets.to(device)
                pe = pe.to(device)
                outputs = model(inputs, pe)

            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)

            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())



    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    average_loss = total_loss / len(test_loader.dataset)
    average_r2 = r2_score(all_targets, all_preds)

    print(f"Test Loss: {average_loss:.5f} | Test R2: {average_r2:.5f}")

    # individual R2 per component of the (4) targets:
    print(all_targets.shape, all_preds.shape)
    for i in range(all_targets.shape[1]):
        r2_i = r2_score(all_targets[:, i], all_preds[:, i])
        print(f"  Component {i} R2: {r2_i:.5f}")

    data = {
        'test_loss': average_loss,
        'R2_test': average_r2,
        'predictions': all_preds,
    }
    path = f'{config["model"]["name"]}_test'
    zarr_writer(f'test_results/{path}_results.zarr', data)
    similarity_plot(all_targets, all_preds,path)

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

    # run_test(model,test_loader,device,loss_function,config=config)
    run_real_media_test(model, config=config, dataset_tags=args.datasets if hasattr(args, 'datasets') else None)
   
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