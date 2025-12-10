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
import yaml
import os
import argparse

def main(config):
    '''
    Docstring for main
    
    :param config: Run specific config dictionary
    '''
    device = config.get('device',torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    # Load model from config dict. The generator emits a structured `config['model']`.
    model_cfg = config.get('model')
    if model_cfg is None:
        raise ValueError("Config must contain a 'model' section")

    model_type = model_cfg.get('type')
    if model_type == 'convnext':
        model = load_convnext_model(model_cfg)
    elif model_type == 'vit':
        model = load_vit_model(model_cfg)
    elif model_type == 'resnet':
        model = load_resnet_model(model_cfg)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Move to device
    try:
        model = model.to(device)
    except Exception:
        print("Warning: couldn't move model to device; continuing on CPU")

    print(f"Loaded model: {model_cfg.get('name', model_type)} | Type: {model_type}")

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model training for permeability and dispersion models.")
    parser.add_argument('--config', type=str, default=None, help='Path to the configuration YAML file.')
    args = parser.parse_args()
    path_to_config = os.path.join(os.path.dirname(__file__), args.config)

    with open(path_to_config, 'r') as f:
        config = yaml.safe_load(f)
    main(config)