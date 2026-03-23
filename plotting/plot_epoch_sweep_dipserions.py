import matplotlib.pyplot as plt
import numpy as np
import zarr
from matplotlib.lines import Line2D
from ploting import figsize
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

length = [200,600,1000]
models = ['ConvNeXt-Atto', 'ResNet-18', 'ViT-T16', 'Swin-T']
colors= ['C9', 'C6', 'C2', 'C1']
markers = ['o','s','D','^']
# colors

fig,axs = plt.subplots(1,1,figsize= (figsize[0],figsize[1]*0.8))
legend_model_handles = []
for i, m in enumerate(models):
    xs, ys = [], []
    color = colors[i]
    marker = markers[i]
    for l in length:
        if l ==200:
            if m=='Swin-T':
                path = (
                    f'results/dispersion_lr_wd_sweep/{m}_lr-0.005_wd-0.05_bs-128_epochs-200_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_rmse_metrics.zarr'
                )
            else:
                path = (
                    f'results/dispersion_lr_wd_sweep/{m}_lr-0.005_wd-0.05_bs-128_epochs-200_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_mse_metrics.zarr'
                )
        else: 
            if m=='Swin-T':
                path = (
                    f'results/dispersion_epoch_sweep/{m}_lr-0.0001_wd-0.05_bs-128_epochs-{l}_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_rmse_metrics.zarr'
                )
            # elif m=='ViT-T16':
            #     path = (
            #         f'results/dispersion_epoch_sweep/{m}_lr-0.005_wd-0.01_bs-128_epochs-{l}_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_log-cosh_metrics.zarr'
            #     )
            else:
                path = (
                    f'results/dispersion_epoch_sweep/{m}_lr-0.005_wd-0.01_bs-128_epochs-{l}_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_mse_metrics.zarr'
                )
        # print(path)
        try:
            root = zarr.open(path, mode='r')
            val_r2 = root['R2_val'][:]
            best = 1 - np.max(val_r2)
            xs.append(l)
            ys.append(best)
        except Exception as e:
            print(f"Skipping {path}: {e}")
            continue

    axs.plot(xs, ys, color=color, linewidth=0.9, alpha=0.5)
    axs.plot(xs, ys, color=color, linestyle='',#fillstyle='none', 
             marker=marker, markersize=8)
    legend_model_handles.append(
                Line2D([0], [0],
                    color=color,
                    marker=marker,
                    linestyle='-',
                    linewidth=1.2,
                    markersize=5,
                    label=m)
            )
axs.set_yscale('log')
# axs.set_xscale('log')
axs.set_xlabel('Epoch')
axs.set_ylabel(r'Lowest $1-R^2$')
axs.set_xticks(length)
axs.grid(alpha=0.3)
axs.grid(which='minor', alpha=0.15)
axs.minorticks_on()

axs.legend(
        handles=legend_model_handles,
        title='Model',
        # loc='upper right',   # fixed position = consistent layout
        frameon=True,
        framealpha=0.3,
        edgecolor='#cccccc',
        fontsize=7,
        labelspacing=0.3,
        handlelength=1.5,
        handletextpad=0.4,
    )
plt.tight_layout()
plt.savefig('thesis_plots/dispersion_epoch_sweep.pdf')

lfs = ['mse','log-cosh']
colors = ['C2', 'C3']

fig,axs = plt.subplots(1,1,figsize=(figsize[0],figsize[1]*0.7))
legend_model_handles = []
m='ViT-T16'
for i, lf in enumerate(lfs):
    xs, ys = [], []
    color = colors[i]
    marker = markers[i]
    for l in length:
        if l ==200:
            if lf=='log-cosh':
                path = (
                    f'results/dispersion_lr_wd_sweep_2/{m}_lr-0.005_wd-0.01_bs-128_epochs-200_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_log-cosh_metrics.zarr'
                )
            else:
                path = (
                    f'results/dispersion_lr_wd_sweep/{m}_lr-0.005_wd-0.01_bs-128_epochs-200_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_mse_metrics.zarr'
                )
        else:
            if lf=='mse':
                path = (
                    f'results/dispersion_epoch_sweep/{m}_lr-0.005_wd-0.01_bs-128_epochs-{l}_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_mse_metrics.zarr'
                )
            else:
                path = (
                    f'results/dispersion_epoch_sweep/{m}_lr-0.005_wd-0.01_bs-128_epochs-{l}_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_log-cosh_metrics.zarr'
                )
        try:
            root = zarr.open(path, mode='r')
            val_r2 = root['R2_val'][:]
            best = 1 - np.max(val_r2)
            xs.append(l)
            ys.append(best)
        except Exception as e:
            print(f"Skipping {path}: {e}")
            continue
    axs.plot(xs, ys, color=color, linewidth=0.9, alpha=0.5)
    axs.plot(xs, ys, color=color, linestyle='',#fillstyle='none', 
             marker=marker, markersize=8)
    legend_model_handles.append(
                Line2D([0], [0],
                    color=color,
                    marker=marker,
                    linestyle='-',
                    linewidth=1.2,
                    markersize=5,
                    label=lf)
            )
axs.set_yscale('log')
# axs.set_xscale('log')
axs.set_xlabel('Epoch')
axs.set_ylabel(r'Lowest $1-R^2$')
axs.set_xticks(length)
axs.grid(alpha=0.3)
axs.grid(which='minor', alpha=0.15)
axs.minorticks_on()

axs.legend(
        handles=legend_model_handles,
        title='ViT-T16 with:',
        # loc='upper right',   # fixed position = consistent layout
        frameon=True,
        framealpha=0.3,
        edgecolor='#cccccc',
        fontsize=7,
        labelspacing=0.3,
        handlelength=1.5,
        handletextpad=0.4,
    )
plt.tight_layout()
plt.tight_layout()
plt.savefig('thesis_plots/dispersion_epoch_sweep_vit.pdf')


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
        "--loss_function",   "mse",
    ]
model_to_family = {m: fam for fam, mlist in model_families.items() for m in mlist}
fig, ax = plt.subplots(2, 1, figsize=(figsize[0], figsize[1]*1.45))
legend_model_handles = []
import os
import subprocess
import re
for family, models_list in model_families.items():
    color  = family_cmaps[family]
    marker = family_markers[family]

    for idx, model in enumerate(models_list):
        param_count = sizes[family][idx]
        best_path = None
        # Try each loss function variant
        for loss in ['mse', 'log-cosh', 'rmse', 'huber']:
            # Construct glob or a known path pattern
            candidates = list(Path(folder).glob(
                f'{model}_*_{loss}_metrics.zarr'
            ))
            if not candidates:
                continue
            # Take the best (or just first) match
            path = candidates[0]
            try:
                root = zarr.open(path, mode='r')
                val_r2 = root['R2_val'][:]
                best = 1 - np.max(val_r2)
                best_path = path
            except Exception as e:
                print(f"Skipping {path}: {e}")
                continue

            if best_path:
                
                path='results/dispersion_all_models/'+Path(best_path).name.replace('_metrics.zarr','')+'.pth'
                local=path
                servers = [
                    ("bigfacet", "bigfacet:/home/users/sigursv/porousflow/"),
                    ("herbie",   "herbie-jump:/home/sigursv/porousflow/"),
                ]
                if not os.path.exists(local):
                    for name, base in servers:
                        try:
                            subprocess.run(["rsync", "-av", f"{base}{path}", local], check=True)
                            print(f"Fetched from {name}")
                            break
                        except subprocess.CalledProcessError:
                            print(f"Not found on {name}, trying next...")
                    else:
                        print("File does not exist on any familiar cluster.")

                if os.path.exists(local):
                    cmd = build_test_cmd(local, m, family)
                    print("Running:", " ".join(cmd))
                    
                    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                    match = re.search(r"Test R2:\s*([\d.]+)", result.stdout)
                    
                    if match:
                        test_r2_for_model = float(match.group(1))
                        print(f"Test R2:       {test_r2_for_model:.5f}")
                    print(f"Validation R2: {best:.5f}")

            ax[0].plot(param_count, best, linestyle='', marker=marker,
                       markersize=6, color=color)
            break  # use first loss that exists
ax[0].set_yscale('log')
ax[0].set_xscale('log')

for key, item in model_families.items():
    if key =='convnext-v2' or key=='convnext-rms':
        continue
    # print(key)
    color = family_cmaps[key]
    marker = family_markers[key]
    name = family_map[key]
    legend_model_handles.append(
                Line2D([0], [0],
                    color=color,
                    marker=marker,
                    linestyle='-',
                    linewidth=1.2,
                    markersize=5,
                    label=name)
            )
# # Test:

#     # ax[0].legend()
ax[0].grid(True, which="both", ls="-", alpha=0.15)
ax[0].legend(
        handles=legend_model_handles,
        title='Architecture:',
        # loc='upper right',   # fixed position = consistent layout
        frameon=True,
        framealpha=0.3,
        edgecolor='#cccccc',
        fontsize=7,
        labelspacing=0.3,
        handlelength=1.5,
        handletextpad=0.4,
    )
plt.tight_layout()
plt.savefig('thesis_plots/scaling_laws_r2_vs_params_dispersion.pdf')