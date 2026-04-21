import matplotlib.pyplot as plt
import numpy as np
import zarr
from matplotlib.lines import Line2D
from ploting import figsize
import os
import matplotlib as mpl
mpl.rcParams['axes.formatter.use_mathtext'] = True

folder = 'results/epoch_sweep_all_models/'
model_families = {
    'resnet': ['ResNet-18', 'ResNet-34', 'ResNet-50', 'ResNet-101', 'ResNet-152'],
    'swin': ['Swin-T', 'Swin-S', 'Swin-B', 'Swin-L'],
    'vit': ['ViT-T16', 'ViT-S16', 'ViT-B16', 'ViT-L16'],
    'convnext': ['ConvNeXt-Atto', 'ConvNeXt-Femto', 'ConvNeXt-Pico', 'ConvNeXt-Nano', 
                 'ConvNeXt-Tiny', 'ConvNeXt-Small', 'ConvNeXt-Base', 'ConvNeXt-Large'],
    'convnext-v2': ['ConvNeXt-V2-Atto', 'ConvNeXt-V2-Femto', 'ConvNeXt-V2-Pico', 'ConvNeXt-V2-Nano', 
                 'ConvNeXt-V2-Tiny', 'ConvNeXt-V2-Small', 'ConvNeXt-V2-Base', 'ConvNeXt-V2-Large'],
    'convnext-rms': ['ConvNeXt-RMS-Atto', 'ConvNeXt-RMS-Femto', 'ConvNeXt-RMS-Pico', 'ConvNeXt-RMS-Nano', 
                 'ConvNeXt-RMS-Tiny', 'ConvNeXt-RMS-Small', 'ConvNeXt-RMS-Base', 'ConvNeXt-RMS-Large'],
}
sizes = {
    "convnext-v2": [3388604, 4849684, 8555204, 14985844, 27871588, 49561444, 87708804, 196443844],
    "convnext-rms": [3371724, 4829428, 8528196, 14946324, 27811204, 49438852, 87545348, 196198660],
    "convnext": [3373884, 4832020, 8531652, 14951284, 27818596, 49453156, 87564420, 196227268],
    "resnet": [11172292, 21280452, 23509956, 42502084, 58145732],
    "vit": [5401156, 21419140, 85305604, 302644228],
    "swin": [27504334, 48804958, 86700156, 194930872]
}

split_styles = {
    'train': '--',
    'val': '-'
}
length_colors = {
    1500: 'C5',
    1000: 'C4',
    600: 'C3',
    500: 'C2',
    200: 'C1',
    100: 'C0',
}

plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
})

family_markers = {
    'resnet': 'o',
    'swin': 's',
    'vit': 'D',
    'convnext': '^',
    'convnext-v2': 'v',
    'convnext-rms': 'P',
}

import re

def parse_zarr_filename(path):
    name = Path(path).name  # e.g. ConvNeXt-Atto_lr-0.005_wd-0.01_..._mse_metrics.zarr
    # Strip suffix
    stem = name.replace('_metrics.zarr', '')
    
    # Extract known hyperparams
    patterns = {
        'lr':     r'_lr-([\d.eE+-]+)',
        'wd':     r'_wd-([\d.eE+-]+)',
        'bs':     r'_bs-(\d+)',
        'epochs': r'_epochs-(\d+)',
        'loss':   r'_(mse|log-cosh|rmse|huber)$',
    }
    
    result = {}
    for key, pat in patterns.items():
        m = re.search(pat, stem)
        result[key] = m.group(1) if m else None
    
    # Model name is everything before the first _lr-
    result['model'] = re.split(r'_lr-', stem)[0]
    
    return result

from pathlib import Path
folder = 'results/dispersion_all_models'
family_cmaps = {
    'resnet': 'C0',
    'swin': 'C1',
    'vit': 'C2',
    'convnext': 'C3',
    'convnext-v2': 'C4',
    'convnext-rms': 'C9'
}
family_markers = {
    'resnet': 'o',
    'swin': 's',
    'vit': 'D',
    'convnext': '^',
    'convnext-v2': 'v',
    'convnext-rms': 'P',
}
family_map = {
    'resnet': 'ResNet', 'swin': 'Swin', 'vit': 'ViT',
    'convnext': 'ConvNeXt', 'convnext-v2': 'ConvNeXt-V2', 'convnext-rms': 'ConvNeXt-RMS'
}
# model_families = dict(models)
legend_model_handles = []
FAMILY_TO_MODEL_ARG = {
    'resnet':       'resnet',
    'vit':          'vit',
    'swin':         'swin',
    'convnext':     'convnext',
    'convnext-v2':  'convnext',
    'convnext-rms': 'convnext',
}

# --version arg per family
FAMILY_TO_VERSION = {
    'resnet':       'None',
    'vit':          'None',
    'swin':         'None',
    'convnext':     'v1',
    'convnext-v2':  'v2',
    'convnext-rms': 'rms',
}


def build_test_cmd(local_path: str, model_name: str, model_family: str, task: str = 'dispersion') -> list[str]:
    """Build the run_model_test.py command from loop variables."""
    # Extract size: last hyphen-separated token, e.g. "Large", "18", "B16"
    size = model_name.split('-')[-1].lower()

    return [
        "python3", "run_model_test.py",
        "--pretrained_path", local_path,
        "--model",           FAMILY_TO_MODEL_ARG[model_family],
        "--model_name",      model_name,
        "--size",            size,
        "--version",         FAMILY_TO_VERSION[model_family],
        "--task",            task,
        "--pe_encoder",      "log",
        "--loss_function",   "mse",
    ]


data = {
    'convnext': {
        'ConvNeXt-Atto': {
            'params': 3373884,
            # 'metrics_path': '',
            'metrics_path': 'results/dispersion_all_models_2/ConvNeXt-Atto_lr-0.001_wd-0.01_bs-128_epochs-1000_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_mse_metrics.zarr',
            'state_dict_path': 'results/dispersion_all_models_2/ConvNeXt-Atto_lr-0.001_wd-0.01_bs-128_epochs-1000_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_mse.pth',
        },
        'ConvNeXt-Femto': {
            'params': 4832020,
            'metrics_path': 'results/dispersion_all_models_2/ConvNeXt-Femto_lr-0.001_wd-0.01_bs-128_epochs-1000_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_mse_metrics.zarr',
            'state_dict_path': 'results/dispersion_all_models_2/ConvNeXt-Femto_lr-0.001_wd-0.01_bs-128_epochs-1000_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_mse.pth',
        },
        'ConvNeXt-Pico': {
            'params': 8531652,
            'metrics_path': 'results/dispersion_all_models_2/ConvNeXt-Pico_lr-0.001_wd-0.01_bs-128_epochs-1000_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_mse_metrics.zarr',
            'state_dict_path': 'results/dispersion_all_models_2/ConvNeXt-Pico_lr-0.001_wd-0.01_bs-128_epochs-1000_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_mse.pth',
        },
        'ConvNeXt-Nano': {
            'params': 14951284,
            'metrics_path': 'results/dispersion_all_models_2/ConvNeXt-Nano_lr-0.001_wd-0.01_bs-128_epochs-1000_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_mse_metrics.zarr',
            'state_dict_path': 'results/dispersion_all_models_2/ConvNeXt-Nano_lr-0.001_wd-0.01_bs-128_epochs-1000_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_mse.pth',
        },
        'ConvNeXt-Tiny': {
            'params': 27818596,
            'metrics_path': 'results/dispersion_all_models_2/ConvNeXt-Tiny_lr-0.001_wd-0.01_bs-128_epochs-1000_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_mse_metrics.zarr',
            'state_dict_path': '',
        },
        'ConvNeXt-Small': {
            'params': 49453156,
            'metrics_path': '',
            'state_dict_path': '',
        },
        'ConvNeXt-Base': {
            'params': 87564420,
            'metrics_path': '',
            'state_dict_path': '',
        },
        'ConvNeXt-Large': {
            'params': 196227268,
            'metrics_path': '',
            'state_dict_path': '',
        },
    },
    'convnext-v2': {
        'ConvNeXt-V2-Atto': {
            'params': 3373884,
            'metrics_path': 'results/dispersion_all_models_2/ConvNeXt-V2-Atto_lr-0.001_wd-0.01_bs-128_epochs-1000_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_mse_metrics.zarr',
            'state_dict_path': 'results/dispersion_all_models_2/ConvNeXt-V2-Atto_lr-0.001_wd-0.01_bs-128_epochs-1000_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_mse.pth',
        },
        'ConvNeXt-V2-Femto': {
            'params': 4832020,
            'metrics_path': 'results/dispersion_all_models_2/ConvNeXt-V2-Femto_lr-0.001_wd-0.01_bs-128_epochs-1000_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_mse_metrics.zarr',
            'state_dict_path': 'results/dispersion_all_models_2/ConvNeXt-V2-Femto_lr-0.001_wd-0.01_bs-128_epochs-1000_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_mse.pth',
        },
        'ConvNeXt-V2-Pico': {
            'params': 8531652,
            'metrics_path': 'results/dispersion_all_models_2/ConvNeXt-V2-Pico_lr-0.001_wd-0.01_bs-128_epochs-1000_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_mse_metrics.zarr',
            'state_dict_path': 'results/dispersion_all_models_2/ConvNeXt-V2-Pico_lr-0.001_wd-0.01_bs-128_epochs-1000_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_mse.pth',
        },
        'ConvNeXt-V2-Nano': {
            'params': 14951284,
            'metrics_path': 'results/dispersion_all_models_2/ConvNeXt-V2-Nano_lr-0.001_wd-0.01_bs-128_epochs-1000_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_mse_metrics.zarr',
            'state_dict_path': 'results/dispersion_all_models_2/ConvNeXt-V2-Nano_lr-0.001_wd-0.01_bs-128_epochs-1000_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_mse.pth',
        },
        'ConvNeXt-V2-Tiny': {
            'params': 27818596,
            # 'metrics_path': '',
            'metrics_path': 'results/dispersion_all_models_2/ConvNeXt-V2-Tiny_lr-0.001_wd-0.01_bs-128_epochs-1000_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_mse_metrics.zarr',
            # 'state_dict_path': 'results/dispersion_all_models_2/ConvNeXt-V2-Tiny_lr-0.001_wd-0.01_bs-128_epochs-1000_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_mse.pth',
            'state_dict_path': '',
        },
        'ConvNeXt-V2-Small': {
            'params': 49453156,
            # 'metrics_path': '',
            'metrics_path': 'results/dispersion_all_models_2/ConvNeXt-V2-Small_lr-0.001_wd-0.01_bs-128_epochs-1000_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_mse_metrics.zarr',
            # 'state_dict_path': 'results/dispersion_all_models_2/ConvNeXt-V2-Small_lr-0.001_wd-0.01_bs-128_epochs-1000_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_mse.pth',
            'state_dict_path': '',
        },
        'ConvNeXt-V2-Base': {
            'params': 87564420,
            'metrics_path': '',
            'state_dict_path': '',
        },
        'ConvNeXt-V2-Large': {
            'params': 196227268,
            'metrics_path': '',
            'state_dict_path': '',
        },
    },
    'convnext-rms': {
        'ConvNeXt-RMS-Atto': {
            'params': 3373884,
            'metrics_path': 'results/dispersion_all_models_2/ConvNeXt-RMS-Atto_lr-0.001_wd-0.01_bs-128_epochs-1000_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_mse_metrics.zarr',
            'state_dict_path': 'results/dispersion_all_models_2/ConvNeXt-RMS-Atto_lr-0.001_wd-0.01_bs-128_epochs-1000_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_mse.pth',
        },
        'ConvNeXt-RMS-Femto': {
            'params': 4832020,
            'metrics_path': 'results/dispersion_all_models_2/ConvNeXt-RMS-Femto_lr-0.001_wd-0.01_bs-128_epochs-1000_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_mse_metrics.zarr',
            'state_dict_path': 'results/dispersion_all_models_2/ConvNeXt-RMS-Femto_lr-0.001_wd-0.01_bs-128_epochs-1000_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_mse.pth',
        },
        'ConvNeXt-RMS-Pico': {
            'params': 8531652,
            'metrics_path': 'results/dispersion_all_models_2/ConvNeXt-RMS-Pico_lr-0.001_wd-0.01_bs-128_epochs-1000_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_mse_metrics.zarr',
            'state_dict_path': 'results/dispersion_all_models_2/ConvNeXt-RMS-Pico_lr-0.001_wd-0.01_bs-128_epochs-1000_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_mse.pth',
        },
        'ConvNeXt-RMS-Nano': {
            'params': 14951284,
            'metrics_path': 'results/dispersion_all_models_2/ConvNeXt-RMS-Nano_lr-0.001_wd-0.01_bs-128_epochs-1000_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_mse_metrics.zarr',
            'state_dict_path': 'results/dispersion_all_models_2/ConvNeXt-RMS-Nano_lr-0.001_wd-0.01_bs-128_epochs-1000_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_mse.pth',
        },
        'ConvNeXt-RMS-Tiny': {
            'params': 27818596,
            'metrics_path': '',
            'state_dict_path': '',
        },
        'ConvNeXt-RMS-Small': {
            'params': 49453156,
            'metrics_path': '',
            'state_dict_path': '',
        },
        'ConvNeXt-RMS-Base': {
            'params': 87564420,
            'metrics_path': '',
            'state_dict_path': '',
        },
        'ConvNeXt-RMS-Large': {
            'params': 196227268,
            'metrics_path': '',
            'state_dict_path': '',
        },
    },
    'resnet': {
        'ResNet-18': {
            'params': 11172292,
            'metrics_path': 'results/dispersion_all_models/ResNet-18_lr-0.005_wd-0.01_bs-128_epochs-1000_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_mse_metrics.zarr',
            'state_dict_path': 'results/dispersion_all_models/ResNet-18_lr-0.005_wd-0.01_bs-128_epochs-1000_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_mse.pth',
        },
        'ResNet-34': {
            'params': 21280452,
            'metrics_path': 'results/dispersion_all_models/ResNet-34_lr-0.005_wd-0.01_bs-128_epochs-1000_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_mse_metrics.zarr',
            'state_dict_path': 'results/dispersion_all_models/ResNet-34_lr-0.005_wd-0.01_bs-128_epochs-1000_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_mse.pth',
        },
        'ResNet-50': {
            'params': 23509956,
            'metrics_path': 'results/dispersion_all_models/ResNet-50_lr-0.005_wd-0.01_bs-128_epochs-1000_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_mse_metrics.zarr',
            'state_dict_path': 'results/dispersion_all_models/ResNet-50_lr-0.005_wd-0.01_bs-128_epochs-1000_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_mse.pth',
        },
        'ResNet-101': {
            'params': 42502084,
            'metrics_path': 'results/dispersion_all_models/ResNet-101_lr-0.005_wd-0.01_bs-128_epochs-1000_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_mse_metrics.zarr',
            'state_dict_path': 'results/dispersion_all_models/ResNet-101_lr-0.005_wd-0.01_bs-128_epochs-1000_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_mse.pth',
        },
        'ResNet-152': {
            'params': 58145732,
            'metrics_path': 'results/dispersion_all_models_2/ResNet-152_lr-0.001_wd-0.01_bs-128_epochs-1000_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_mse_metrics.zarr',
            'state_dict_path': 'results/dispersion_all_models_2/ResNet-152_lr-0.001_wd-0.01_bs-128_epochs-1000_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_mse.pth',
        },
    },
    'vit': {
        'ViT-T16': {
            'params': 5401156,
            'metrics_path': 'results/dispersion_all_models/ViT-T16_lr-0.005_wd-0.01_bs-128_epochs-600_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_mse_metrics.zarr',
            'state_dict_path': 'results/dispersion_all_models/ViT-T16_lr-0.005_wd-0.01_bs-128_epochs-600_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_mse.pth',
        },
        'ViT-S16': {
            'params': 21419140,
            'metrics_path': 'results/dispersion_all_models/ViT-S16_lr-0.0005_wd-0.05_bs-128_epochs-600_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_rmse_metrics.zarr',
            'state_dict_path': 'results/dispersion_all_models/ViT-S16_lr-0.0005_wd-0.05_bs-128_epochs-600_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_rmse.pth',
        },
        'ViT-B16': {
            'params': 85305604,
            'metrics_path': 'results/dispersion_all_models/ViT-B16_lr-0.0005_wd-0.05_bs-128_epochs-600_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_rmse_metrics.zarr',
            'state_dict_path': 'results/dispersion_all_models/ViT-B16_lr-0.0005_wd-0.05_bs-128_epochs-600_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_rmse.pth',
        },
        'ViT-L16': {
            'params': 302644228,
            'metrics_path': 'results/dispersion_all_models/ViT-L16_lr-0.0005_wd-0.05_bs-128_epochs-600_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_rmse_metrics.zarr',
            'state_dict_path': 'results/dispersion_all_models/ViT-L16_lr-0.0005_wd-0.05_bs-128_epochs-600_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_rmse.pth',
        },
    },
    'swin': {
        'Swin-T': {
            'params': 27504334,
            'metrics_path': 'results/dispersion_all_models/Swin-T_lr-0.0001_wd-0.05_bs-128_epochs-1000_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_rmse_metrics.zarr',
            'state_dict_path': 'results/dispersion_epoch_sweep/Swin-T_lr-0.0001_wd-0.05_bs-128_epochs-1000_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_rmse.pth',
        },
        'Swin-S': {
            'params': 48804958,
            'metrics_path': 'results/dispersion_all_models_2/Swin-S_lr-0.0001_wd-0.05_bs-128_epochs-1000_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_rmse_metrics.zarr',
            'state_dict_path': 'results/dispersion_all_models_2/Swin-S_lr-0.0001_wd-0.05_bs-128_epochs-1000_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_rmse.pth',
        },
        'Swin-B': {
            'params': 86700156,
            'metrics_path': 'results/dispersion_all_models_2/Swin-B_lr-0.0001_wd-0.05_bs-128_epochs-1000_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_rmse_metrics.zarr',
            'state_dict_path': 'results/dispersion_all_models_2/Swin-B_lr-0.0001_wd-0.05_bs-128_epochs-1000_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_rmse.pth',
        },
        'Swin-L': {
            'params': 194930872,
            'metrics_path': 'results/dispersion_all_models_2/Swin-L_lr-0.0001_wd-0.05_bs-128_epochs-1000_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_rmse_metrics.zarr',
            'state_dict_path': 'results/dispersion_all_models_2/Swin-L_lr-0.0001_wd-0.05_bs-128_epochs-1000_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_rmse.pth',
        },
    },
}





convnext_comparisont = {
    'original': {
        'ConvNeXt-Atto': {
            'params': 3373884,
            'metrics_path': 'results/dispersion_all_models_2/ConvNeXt-Atto_lr-0.001_wd-0.01_bs-128_epochs-1000_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_mse_metrics.zarr',
            'state_dict_path': '',
        },
        'ConvNeXt-Femto': {
            'params': 4832020,
            'metrics_path': 'results/dispersion_all_models_2/ConvNeXt-Femto_lr-0.001_wd-0.01_bs-128_epochs-1000_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_mse_metrics.zarr',
            'state_dict_path': '',
        },
        'ConvNeXt-Pico': {
            'params': 8531652,
            'metrics_path': 'results/dispersion_all_models_2/ConvNeXt-Pico_lr-0.001_wd-0.01_bs-128_epochs-1000_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_mse_metrics.zarr',
            'state_dict_path': '',
        },
        'ConvNeXt-Nano': {
            'params': 14951284,
            'metrics_path': 'results/dispersion_all_models_2/ConvNeXt-Nano_lr-0.001_wd-0.01_bs-128_epochs-1000_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_mse_metrics.zarr',
            'state_dict_path': '',
        },
        # 'ConvNeXt-Tiny': {
        #     'params': 27818596,
        #     'metrics_path': '',
        #     'state_dict_path': '',
        # },
    },

    'modified': {
        'ConvNeXt-Atto': {
            'params': 3373884,
            'metrics_path': 'results/dispersion_all_models/ConvNeXt-Atto_lr-0.005_wd-0.01_bs-128_epochs-600_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_mse_metrics.zarr',
            'state_dict_path': '',
        },
        'ConvNeXt-Femto': {
            'params': 4832020,
            'metrics_path': 'results/dispersion_all_models/ConvNeXt-Femto_lr-0.005_wd-0.005_bs-128_epochs-600_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_mse_metrics.zarr',
            'state_dict_path': '',
        },
        'ConvNeXt-Pico': {
            'params': 8531652,
            'metrics_path': 'results/dispersion_all_models/ConvNeXt-Pico_lr-0.005_wd-0.01_bs-128_epochs-600_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_mse_metrics.zarr',
            'state_dict_path': '',
        },
        'ConvNeXt-Nano': {
            'params': 14951284,
            'metrics_path': 'results/dispersion_all_models/ConvNeXt-Nano_lr-0.005_wd-0.01_bs-128_epochs-600_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_mse_metrics.zarr',
            'state_dict_path': '',
        },
        'ConvNeXt-Tiny': {
            'params': 27818596,
            'metrics_path': 'results/dispersion_all_models/ConvNeXt-Tiny_lr-0.001_wd-0.005_bs-128_epochs-600_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_mse_metrics.zarr',
            'state_dict_path': '',
        },
        'ConvNeXt-Small': {
            'params': 49453156,
            'metrics_path': 'results/dispersion_all_models/ConvNeXt-Small_lr-0.001_wd-0.005_bs-128_epochs-600_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_mse_metrics.zarr',
            'state_dict_path': '',
        },
        'ConvNeXt-Base': {
            'params': 87564420,
            'metrics_path': 'results/dispersion_all_models/ConvNeXt-Base_lr-0.005_wd-0.005_bs-128_epochs-600_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_mse_metrics.zarr',
            'state_dict_path': '',
        },
        'ConvNeXt-Large': {
            'params': 196227268,
            'metrics_path': 'results/dispersion_all_models/ConvNeXt-Large_lr-0.005_wd-0.005_bs-128_epochs-600_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_mse_metrics.zarr',
            'state_dict_path': '',
        },
    },
}


scale_comparisont = {
    'asinh': {
        'ConvNeXt-RMS-Atto': {
            'params': 3373884,
            'metrics_path': 'results/dispersion_all_models_2/ConvNeXt-Nano_lr-0.001_wd-0.01_bs-128_epochs-1000_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_mse_metrics.zarr',
            'state_dict_path': 'results/dispersion_all_models_2/ConvNeXt-RMS-Atto_lr-0.001_wd-0.01_bs-128_epochs-1000_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_mse.pth',
        },
    },
    'no_scale': {
        'ConvNeXt-RMS-Atto': {
            'params': 3373884,
            'metrics_path': 'results/convnext_no_asinh_test/ConvNeXt-RMS-Atto_lr-0.0001_wd-0.01_bs-128_epochs-1000_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_mse_metrics.zarr',
            'state_dict_path': 'results/convnext_no_asinh_test/ConvNeXt-RMS-Atto_lr-0.0001_wd-0.01_bs-128_epochs-1000_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_mse.pth',
        },
    },
    'log': {
        'ConvNeXt-RMS-Atto': {
            'params': 3373884,
            'metrics_path': 'results/convnext_log_test/ConvNeXt-RMS-Atto_lr-0.001_wd-0.01_bs-128_epochs-1000_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_mse_metrics.zarr',
            'state_dict_path': 'results/convnext_log_test/ConvNeXt-RMS-Atto_lr-0.001_wd-0.01_bs-128_epochs-1000_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_mse.pth',
        },
    },
}


# Log test
data_log = {
    'convnext': {
        'ConvNeXt-Atto': {
            'params': 3373884,
            'metrics_path': 'results/log_dispersion_all_models/ConvNeXt-Atto_lr-0.001_wd-0.01_bs-128_epochs-1000_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_mse_metrics.zarr',
            'state_dict_path': 'results/log_dispersion_all_models/ConvNeXt-Atto_lr-0.001_wd-0.01_bs-128_epochs-1000_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_mse.pth',
        },
        'ConvNeXt-Femto': {
            'params': 4832020,
            'metrics_path': 'results/log_dispersion_all_models/ConvNeXt-Femto_lr-0.001_wd-0.01_bs-128_epochs-1000_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_mse_metrics.zarr',
            'state_dict_path': '',
        },
        'ConvNeXt-Pico': {
            'params': 8531652,
            'metrics_path': 'results/log_dispersion_all_models/ConvNeXt-Pico_lr-0.001_wd-0.01_bs-128_epochs-1000_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_mse_metrics.zarr',
            'state_dict_path': '',
        },
        'ConvNeXt-Nano': {
            'params': 14951284,
            'metrics_path': 'results/log_dispersion_all_models/ConvNeXt-Nano_lr-0.001_wd-0.01_bs-128_epochs-1000_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_mse_metrics.zarr',
            'state_dict_path': 'results/log_dispersion_all_models/ConvNeXt-Nano_lr-0.001_wd-0.01_bs-128_epochs-1000_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_mse.pth',
        },
        'ConvNeXt-Tiny': {
            'params': 27818596,
            'metrics_path': 'results/log_dispersion_all_models/ConvNeXt-Tiny_lr-0.001_wd-0.01_bs-128_epochs-1000_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_mse_metrics.zarr',
            'state_dict_path': '',
        },
        'ConvNeXt-Small': {
            'params': 49453156,
            'metrics_path': 'results/log_dispersion_all_models/ConvNeXt-Small_lr-0.001_wd-0.01_bs-128_epochs-1000_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_mse_metrics.zarr',
            'state_dict_path': '',
        },
    },
    'ConvNeXt-V2':{
        'ConvNeXt-V2-Atto': {
            'params': 3373884,
            'metrics_path': 'results/log_dispersion_all_models/ConvNeXt-V2-Atto_lr-0.001_wd-0.01_bs-128_epochs-1000_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_mse_metrics.zarr',
            'state_dict_path': '',
        },
        'ConvNeXt-V2-Femto': {
            'params': 4832020,
            'metrics_path': 'results/log_dispersion_all_models/ConvNeXt-V2-Femto_lr-0.001_wd-0.01_bs-128_epochs-1000_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_mse_metrics.zarr',
            'state_dict_path': '',
        },
        'ConvNeXt-V2-Pico': {
            'params': 8531652,
            'metrics_path': 'results/log_dispersion_all_models/ConvNeXt-V2-Pico_lr-0.001_wd-0.01_bs-128_epochs-1000_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_mse_metrics.zarr',
            'state_dict_path': '',
        },
        'ConvNeXt-V2-Nano': {
            'params': 14951284,
            'metrics_path': 'results/log_dispersion_all_models/ConvNeXt-V2-Nano_lr-0.001_wd-0.01_bs-128_epochs-1000_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_mse_metrics.zarr',
            'state_dict_path': 'results/log_dispersion_all_models/ConvNeXt-V2-Nano_lr-0.001_wd-0.01_bs-128_epochs-1000_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_mse.pth',
        },
        'ConvNeXt-V2-Tiny': {
            'params': 27818596,
            'metrics_path': 'results/log_dispersion_all_models/ConvNeXt-V2-Tiny_lr-0.001_wd-0.01_bs-128_epochs-1000_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_mse_metrics.zarr',
            'state_dict_path': '',
        },
        'ConvNeXt-V2-Small': {
            'params': 49453156,
            'metrics_path': 'results/log_dispersion_all_models/ConvNeXt-V2-Small_lr-0.001_wd-0.01_bs-128_epochs-1000_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_mse_metrics.zarr',
            'state_dict_path': '',
        },
    },
    'ConvNeXt-RMS':{
        'ConvNeXt-RMS-Atto': {
            'params': 3373884,
            'metrics_path': 'results/log_dispersion_all_models/ConvNeXt-V2-Atto_lr-0.001_wd-0.01_bs-128_epochs-1000_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_mse_metrics.zarr',
            'state_dict_path': '',
        },
        'ConvNeXt-RMS-Femto': {
            'params': 4832020,
            'metrics_path': 'results/log_dispersion_all_models/ConvNeXt-RMS-Femto_lr-0.001_wd-0.01_bs-128_epochs-1000_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_mse_metrics.zarr',
            'state_dict_path': '',
        },
        'ConvNeXt-RMS-Pico': {
            'params': 8531652,
            'metrics_path': 'results/log_dispersion_all_models/ConvNeXt-RMS-Pico_lr-0.001_wd-0.01_bs-128_epochs-1000_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_mse_metrics.zarr',
            'state_dict_path': '',
        },
        'ConvNeXt-RMS-Nano': {
            'params': 14951284,
            'metrics_path': 'results/log_dispersion_all_models/ConvNeXt-V2-Nano_lr-0.001_wd-0.01_bs-128_epochs-1000_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_mse_metrics.zarr',
            'state_dict_path': '',
        },
        'ConvNeXt-RMS-Tiny': {
            'params': 27818596,
            'metrics_path': 'results/log_dispersion_all_models/ConvNeXt-RMS-Tiny_lr-0.001_wd-0.01_bs-128_epochs-1000_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_mse_metrics.zarr',
            'state_dict_path': '',
        },
        'ConvNeXt-RMS-Small': {
            'params': 49453156,
            'metrics_path': 'results/log_dispersion_all_models/ConvNeXt-RMS-Small_lr-0.001_wd-0.01_bs-128_epochs-1000_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_mse_metrics.zarr',
            'state_dict_path': '',
        },
    },
    'ResNet': {
        'ResNet-18': {
            'params': 11172292,
            'metrics_path': 'results/log_dispersion_all_models/ResNet-18_lr-0.001_wd-0.01_bs-128_epochs-1000_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_mse_metrics.zarr',
            'state_dict_path': '',
        },
        'ResNet-34': {
            'params': 21280452,
            'metrics_path': 'results/log_dispersion_all_models/ResNet-34_lr-0.001_wd-0.01_bs-128_epochs-1000_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_mse_metrics.zarr',
            'state_dict_path': '',
        },
        'ResNet-50': {
            'params': 23509956,
            'metrics_path': 'results/log_dispersion_all_models/ResNet-50_lr-0.001_wd-0.01_bs-128_epochs-1000_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_mse_metrics.zarr',
            'state_dict_path': '',
        },
        'ResNet-101': {
            'params': 42502084,
            'metrics_path': 'results/log_dispersion_all_models/ResNet-101_lr-0.001_wd-0.01_bs-128_epochs-1000_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_mse_metrics.zarr',
            'state_dict_path': '',
        },
        'ResNet-152': {
            'params': 58145732,
            'metrics_path': 'results/log_dispersion_all_models/ResNet-152_lr-0.001_wd-0.01_bs-128_epochs-1000_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_mse_metrics.zarr',
            'state_dict_path': '',
        },
    },
    'ViT': {
        'ViT-T16': {
            'params': 5401156,
            'metrics_path': 'results/log_dispersion_all_models/ViT-T16_lr-0.001_wd-0.01_bs-128_epochs-1000_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_mse_metrics.zarr',
            'state_dict_path': '',
        },
        'ViT-S16': {
            'params': 21419140,
            'metrics_path': 'results/log_dispersion_all_models/ViT-S16_lr-0.001_wd-0.01_bs-128_epochs-1000_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_mse_metrics.zarr',
            'state_dict_path': '',
        },
        'ViT-B16': {
            'params': 85305604,
            'metrics_path': '',
            'state_dict_path': '',
        },
        'ViT-L16': {
            'params': 302644228,
            'metrics_path': '',
            'state_dict_path': '',
        },
    },
    'Swin': {
        'Swin-T': {
            'params': 27504334,
            'metrics_path': '',
            'state_dict_path': '',
        },
        'Swin-S': {
            'params': 48804958,
            'metrics_path': '',
            'state_dict_path': '',
        },
        'Swin-B': {
            'params': 86700156,
            'metrics_path': '',
            'state_dict_path': '',
        },
        'Swin-L': {
            'params': 194930872,
            'metrics_path': '',
            'state_dict_path': '',
        },
    },
}
family_markers = {
    'resnet': 'o',
    'swin': 's',
    'vit': 'D',
    'convnext': '^',
    'convnext-v2': 'v',
    'convnext-rms': 'P',
}
family_cmaps = {
    'resnet': 'C0',
    'swin': 'C1',
    'vit': 'C2',
    'convnext': 'C3',
    'convnext-v2': 'C4',
    'convnext-rms': 'C9'
}
# show comparison of log vs original (asinh)
fig, ax = plt.subplots(2,1, figsize=(figsize[0], figsize[1]*1.2), sharex=True, sharey=True)
legend_model_handles = []
# plot data log
print("Plotting log data...")
for family, models in data_log.items():
    marker = family_markers.get(family.lower(), 'o')
    color = family_cmaps.get(family.lower(), 'C0')
    xs, ys = [], []
    for model, info in models.items():
        params = info['params']
        metrics_path = info['metrics_path']
        val_r2 = None

        if metrics_path and os.path.exists(metrics_path):
            try:
                root = zarr.open(metrics_path, mode='r')
                val_r2 = root['R2_val'][:]
                val_r2 = np.max(val_r2)
                print(f"Loaded val R2 for {model} ({family}) from metrics: {val_r2:.5f}")
            except Exception as e:
                print(f"Error loading metrics for {model} ({family}): {e}")
        else:
            print(f"No metrics path for {model} ({family}), skipping.")

        if val_r2 is not None:
            ax[0].plot(params, 1 - val_r2, color=color, marker=marker,
                       markersize=7, fillstyle='none', linestyle='', zorder=2)
            xs.append(params)
            ys.append(1 - val_r2)
            # ax[0].text(params, 1 - val_r2, model.split('-')[0], fontsize=6, ha='center', va='bottom')
    ax[0].plot(xs, ys, color=color, linestyle='-', alpha=0.5, zorder=1)
    legend_model_handles.append(plt.Line2D([0], [0], color=color, marker=marker,
                                               markersize=7, fillstyle='none', linestyle='', label=family))
print("\nPlotting asinh data...")
# plot data 
for family, models in data.items():
    marker = family_markers.get(family.lower(), 'o')
    color = family_cmaps.get(family.lower(), 'C0')
    xs, ys = [], []
    for model, info in models.items():
        params = info['params']
        metrics_path = info['metrics_path']
        val_r2 = None

        if metrics_path and os.path.exists(metrics_path):
            try:
                root = zarr.open(metrics_path, mode='r')
                val_r2 = root['R2_val'][:]
                val_r2 = np.max(val_r2)
                print(f"Loaded val R2 for {model} ({family}) from metrics: {val_r2:.5f}")
            except Exception as e:
                print(f"Error loading metrics for {model} ({family}): {e}")
        else:
            print(f"No metrics path for {model} ({family}), skipping.")

        if val_r2 is not None:
            ax[1].plot(params, 1 - val_r2, color=color, marker=marker,
                       markersize=7, fillstyle='none', linestyle='', zorder=2)
            xs.append(params)
            ys.append(1 - val_r2)
            # ax[0].text(params, 1 - val_r2, model.split('-')[0], fontsize=6, ha='center', va='bottom')
    ax[1].plot(xs, ys, color=color, linestyle='-', alpha=0.5, zorder=1)

# test plot
# for family, models in data_log.items():
#     for model, info in models.items():
#         params = info['params']
#         state_dict_path = info['state_dict_path']
#         test_r2 = None

#         if False:#state_dict_path and os.path.exists(state_dict_path):
#             print(f"State dict found for {model} ({family}), running test script...")
#             # cmd = build_test_cmd(state_dict_path, model, family)
#             cmd = [
#                 "python3", "run_model_test.py",
#                 "--pretrained_path", state_dict_path,
#                 "--model", family.lower(),
#                 "--model_name", model+'_'+'log',
#                 "--size", model.split('-')[1],
#                 "--version", "rms",
#                 "--task", "dispersion",
#                 "--pe_encoder", "log",
#                 "--loss_function", "mse"
#             ]
#             print("Running:", " ".join(cmd))
            
#             result = subprocess.run(cmd, check=True, capture_output=True, text=True)
#             match = re.search(r"Test R2:\s*([\d.]+)", result.stdout)
            
#             if match:
#                 test_r2 = float(match.group(1))
#                 print(f"Test R2 for {model} ({family}): {test_r2:.5f}")
#                 ax[1].plot(params, 1 - test_r2, color=family_cmaps.get(family, 'C0'), marker=family_markers.get(family, 'o'),
#                            markersize=7, fillstyle='none', linestyle='', zorder=2)
#                 ax[1].text(params, 1 - test_r2, model.split('-')[0], fontsize=6, ha='center', va='bottom')
#             else:
#                 print(f"No Test R2 found in output for {model} ({family}).")
#         else:
#             print(f"No state dict for {model} ({family}), skipping test R2.")

ax[0].legend(handles=legend_model_handles, title="Model Family", fontsize=6, title_fontsize=7, loc='upper right')

ax[0].set_xscale('log')
ax[0].set_yscale('log')
ax[0].set_ylabel(r'Log $1 - R^2$')
ax[0].grid(True, which="both", ls="-", alpha=0.15)
ax[1].set_xscale('log')
ax[1].set_yscale('log')
ax[1].set_xlabel('Total Parameters')
ax[1].set_ylabel(r'Asinh $1 - R^2$')
ax[1].grid(True, which="both", ls="-", alpha=0.15)
plt.tight_layout()
plt.savefig('thesis_plots/log_comparison_dispersion.pdf')
plt.close()
print("Saved thesis_plots/log_comparison_dispersion.pdf")