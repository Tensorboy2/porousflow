'''
Docstring for run_model_training

This script sets up and runs the training process for the permeability and dispersion models.
It initializes the Trainer class and manages the training loop.
'''
from src.ml.trainer import Trainer
# from src.ml.models.resnet import 
# from src.ml.models.vit import 
from src.ml.models.convnext import load_convnext_model
from src.ml.models.vit import load_vit_model
from src.ml.models.resnet import load_resnet_model
from src.ml.data_loader import get_permeability_dataloader, get_dispersion_dataloader
from src.ml.trainer import Trainer
from torch import optim
import torch
import torch.nn as nn
import yaml
import os
import argparse

class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps # Added for numerical stability
    
    def forward(self, x, y):
        # Calculate MSE and take the square root
        loss = torch.sqrt(self.mse(x, y) + self.eps)
        return loss
class LogCoshLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        error = y_pred - y_true
        return torch.mean(torch.log(torch.cosh(error + 1e-12)))

class RelativeSquaredErrorLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, y_pred, y_true):
        y_true_flat = y_true.view(y_true.shape[0], -1)
        y_pred_flat = y_pred.view(y_pred.shape[0], -1)

        num = torch.sum((y_pred_flat - y_true_flat) ** 2, dim=1)
        mean = torch.mean(y_true_flat, dim=1, keepdim=True)
        denom = torch.sum((y_true_flat - mean) ** 2, dim=1) + self.eps

        return torch.mean(num / denom)
    
loss_functions = {
    'mse': nn.MSELoss(),
    'L1': nn.L1Loss(),
    'huber': nn.HuberLoss(),
    'rmse': RMSELoss(),
    'log-cosh': LogCoshLoss(),
    'rse': RelativeSquaredErrorLoss(),
}

def main(config):
    '''
    Docstring for main
    
    :param config: Run specific config dictionary
    '''
    device = config.get('device',torch.device("cuda" if torch.cuda.is_available() else "cpu"))
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
        model = load_convnext_model(model_cfg,task=task,Pe_encoder=pe_encoder,include_direction=include_direction)
    elif model_type == 'vit':
        model = load_vit_model(model_cfg,task=task,Pe_encoder=pe_encoder,include_direction=include_direction)
    elif model_type == 'resnet':
        model = load_resnet_model(model_cfg,task=task,Pe_encoder=pe_encoder,include_direction=include_direction)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Move to device
    try:
        model = model.to(device)
    except Exception:
        print("Warning: couldn't move model to device; continuing on CPU")

    print(f"Loaded model: {model_cfg.get('name', model_type)} | Type: {model_type}")

    # Setup data loaders
    batch_size = config.get('batch_size', 32)
    if task == 'permeability':
        train_loader, val_loader, test_loader = get_permeability_dataloader(file_path='data',config=config)
    elif task == 'dispersion':
        train_loader, val_loader, test_loader = get_dispersion_dataloader(file_path='data',config=config)
    else:
        raise ValueError(f"Unsupported task: {task}")
    # print(config)
    print(f"Data loaders set up for task: {task} | Batch size: {batch_size}")

    # Setup optimizer
    learning_rate = config.get('learning_rate', 1e-3)
    weight_decay = config.get('weight_decay', 0.0)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(0.9, 0.999))
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
    print(f"Optimizer: AdamW | LR: {learning_rate} | Weight Decay: {weight_decay}")
    print(f"Dataset sizes - Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")
    print(f"Warmup Steps: {config.get('warmup_steps', 0)} | Decay: {config.get('decay', 'None')}")
    # Initialize Trainer
    loss_function = loss_functions[config.get('loss_function','log-cosh')]
    print(f"Loss function: {config.get('loss_function','log-cosh')}")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        config=config,
        device=device,
        criterion=loss_function
    )
    print("Trainer initialized.")

    # Run training
    num_epochs = config.get('num_epochs', 10)
    print(f"Starting training for {num_epochs} epochs...")
    trainer.train(num_epochs=num_epochs)
    print("Training completed.\n")

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model training for permeability and dispersion models.")
    parser.add_argument('--config', type=str, default=None, help='Path to the configuration YAML file.')
    args = parser.parse_args()
    path_to_config = os.path.join(os.path.dirname(__file__), args.config)

    with open(path_to_config, 'r') as f:
        config = yaml.safe_load(f)

    for exp_config in config.get('experiments', []):
        main(exp_config)

