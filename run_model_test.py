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
    for i in range(targets.shape[1]):
        ax[i].scatter(targets[:, i], preds[:, i], alpha=0.5, s=1.5, label=f'Component {i}')
        ax[i].plot([targets[:, i].min(), targets[:, i].max()], [targets[:, i].min(), targets[:, i].max()], 'k--', alpha=0.3)
        # ax[i].legend()
        ax[i].set_xlabel('True Values')
        ax[i].set_ylabel('Predicted Values')
        # if i == 0 or i == 3:
        #     ax[i].set_yscale('log')
        #     ax[i].set_xscale('log')


        ax[i].set_title(f'Component {i} Similarity')
        ax[i].set_aspect('equal', adjustable='box')
        ax[i].grid(alpha=0.3)
    # plt.title('Similarity Plot: True vs Predicted')
    plt.tight_layout()
    plt.savefig(f'thesis_plots/{path}_similarity_plot.png', dpi=300)

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