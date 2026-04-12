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
            # elif m=='ConvNeXt-Atto':
            #     path = (
            #         f'results/dispersion_epoch_sweep_2/{m}_lr-0.005_wd-0.01_bs-128_epochs-{l}_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_mse_metrics.zarr'
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
        "--pe_encoder",      "log",
        "--loss_function",   "mse",
    ]
# model_to_family = {m: fam for fam, mlist in model_families.items() for m in mlist}
# fig, ax = plt.subplots(2, 1, figsize=(figsize[0], figsize[1]*1.45))
# legend_model_handles = []
# import os
# import subprocess
# import re
# import json
# CACHE_FILE = "thesis_plots/all_family_data_dispersion_cache.json"
# if os.path.exists(CACHE_FILE):
#     print("Loading cached family data...")
#     with open(CACHE_FILE, 'r') as f:
#         all_family_data = json.load(f)
#     # json doesn't preserve tuples, convert lists back
#     all_family_data = {
#         fam: ([(p, v, c, name, t) for p, v, c, name, t in entries], marker)
#         for fam, (entries, marker) in all_family_data.items()
#     }
# else:
#     all_family_data = {}

#     for family, models_list in model_families.items():
#         color  = family_cmaps[family]
#         marker = family_markers[family]
#         family_data = []
#         for idx, model in enumerate(models_list):
#             param_count = sizes[family][idx]
#             best_path = None
#             best = -np.inf
#             # Try each loss function variant
#             for loss in ['mse', 'log-cosh', 'rmse', 'huber']:
#                 # Construct glob or a known path pattern
#                 candidates = list(Path(folder).glob(
#                     f'{model}_*_{loss}_metrics.zarr'
#                 ))
#                 if not candidates:
#                     continue
#                 # Take the best (or just first) match
#                 path = candidates[0]
#                 try:
#                     root = zarr.open(path, mode='r')
#                     val_r2 = root['R2_val'][:]
#                     best = 1 - np.max(val_r2)
#                     best_path = path
#                 except Exception as e:
#                     print(f"Skipping {path}: {e}")
#                     continue

#                 if best_path:
                    
#                     path='results/dispersion_all_models/'+Path(best_path).name.replace('_metrics.zarr','')+'.pth'
#                     local=path
#                     servers = [
#                         ("bigfacet", "bigfacet:/home/users/sigursv/porousflow"),
#                         ("herbie",   "herbie-jump:/home/sigursv/porousflow"),
#                     ]
#                     if not os.path.exists(local):
#                         for name, base in servers:
#                             print(f'Trying {name}, with {path}')
#                             print(f'{base}/{path}')
#                             try:
#                                 subprocess.run(["rsync", "-av", f"{base}/{path}", local], check=True)
#                                 print(f"Fetched from {name}")
#                                 break
#                             except subprocess.CalledProcessError:
#                                 print(f"Not found on {name}, trying next...")
#                         else:
#                             print("File does not exist on any familiar cluster.")

#                     if os.path.exists(local):
#                         print(model,family)
#                         cmd = build_test_cmd(local, model, family)
#                         print("Running:", " ".join(cmd))
                        
#                         result = subprocess.run(cmd, check=True, capture_output=True, text=True)
#                         match = re.search(r"Test R2:\s*([\d.]+)", result.stdout)
                        
#                         if match:
#                             test_r2_for_model = float(match.group(1))
#                             print(f"Test R2:       {test_r2_for_model:.5f}")
#                         print(f"Validation R2: {1-best:.5f}")
#                 if best != -np.inf:
#                     test_error_val = (float(1 - test_r2_for_model)) if test_r2_for_model is not None else None
#                     family_data.append((
#                         int(param_count),
#                         float(1 - best),
#                         color,
#                         m,
#                         test_error_val
#                     ))

#                 ax[0].plot(param_count, best, linestyle='', marker=marker,
#                         markersize=6, color=color)
#             all_family_data[family] = (family_data, marker)

#     os.makedirs("thesis_plots", exist_ok=True)
#     with open(CACHE_FILE, 'w') as f:
#         json.dump(all_family_data, f, indent=2)
#     print(f"Saved family data to {CACHE_FILE}")

# def make_scaling_plot(all_family_data, path: str):
#     fig, axes = plt.subplots(2, 1, figsize=(figsize[0], figsize[1]*1.45),
#                              sharex=True, sharey=True)

#     for ax, use_test in zip(axes, [False, True]):
#         for model_family, (family_data, marker) in all_family_data.items():
#             color = family_cmaps.get(model_family, 'C0')

#             if use_test:
#                 plot_data = [(x, te, name) for x, ve, c, name, te in family_data if te is not None]
#             else:
#                 plot_data = [(x, ve, name) for x, ve, c, name, te in family_data]

#             if not plot_data:
#                 continue

#             plot_data.sort(key=lambda x: x[0])
#             xs, ys, names = zip(*plot_data)

#             ax.plot(xs, ys, color=color, alpha=0.3, zorder=1)
#             for x, y, name in plot_data:
#                 ax.plot(x, y, color=color, linestyle='', marker=marker,
#                         markersize=7, fillstyle='none', zorder=2)

#         ax.set_xscale('log')
#         ax.set_yscale('log')
#         ax.set_ylabel(r'Test $1 - R^2$' if use_test else r'Validation $1 - R^2$')
#         ax.grid(True, which="both", ls="-", alpha=0.15)

#     axes[1].set_xlabel('Total Parameters')

#     legend_elements = [
#         Line2D([0], [0], marker=family_markers[f], color=family_cmaps[f],
#                label=family_map[f], markersize=8, fillstyle='none', linestyle='-')
#         for f in models.keys()
#     ]
#     axes[0].legend(handles=legend_elements, title="Architectures",
#                    loc='best', fontsize=8)

#     plt.tight_layout()
#     plt.savefig(path)
#     plt.close()
#     print(f"Saved {path}")

# make_scaling_plot(all_family_data, 'thesis_plots/scaling_laws_r2_vs_params_dispersion.pdf')

            # break  # use first loss that exists
# ax[0].set_yscale('log')
# ax[0].set_xscale('log')

# for key, item in model_families.items():
#     if key =='convnext-v2' or key=='convnext-rms':
#         continue
#     # print(key)
#     color = family_cmaps[key]
#     marker = family_markers[key]
#     name = family_map[key]
#     legend_model_handles.append(
#                 Line2D([0], [0],
#                     color=color,
#                     marker=marker,
#                     linestyle='-',
#                     linewidth=1.2,
#                     markersize=5,
#                     label=name)
#             )
# # # Test:

# #     # ax[0].legend()
# ax[0].grid(True, which="both", ls="-", alpha=0.15)
# ax[0].legend(
#         handles=legend_model_handles,
#         title='Architecture:',
#         # loc='upper right',   # fixed position = consistent layout
#         frameon=True,
#         framealpha=0.3,
#         edgecolor='#cccccc',
#         fontsize=7,
#         labelspacing=0.3,
#         handlelength=1.5,
#         handletextpad=0.4,
#     )
# plt.tight_layout()
# plt.savefig('thesis_plots/scaling_laws_r2_vs_params_dispersion.pdf')

data = {
    'convnext': {
        'ConvNeXt-Atto': {
            'params': 3373884,
            # 'metrics_path': '',
            'metrics_path': 'results/dispersion_convnext_long_test/ConvNeXt-Atto_lr-0.0005_wd-0.05_bs-128_epochs-1500_cosine_warmup-6250.0_clipgrad-True_pe-encoder-log_pe-4_mse_metrics.zarr',
            'state_dict_path': 'results/dispersion_convnext_long_test/ConvNeXt-Atto_lr-0.0005_wd-0.05_bs-128_epochs-1500_cosine_warmup-6250.0_clipgrad-True_pe-encoder-log_pe-4_mse.pth',
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
            'metrics_path': '',
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
            'metrics_path': '',
            # 'metrics_path': 'results/dispersion_all_models_2/ConvNeXt-V2-Tiny_lr-0.001_wd-0.01_bs-128_epochs-1000_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_mse_metrics.zarr',
            'state_dict_path': 'results/dispersion_all_models_2/ConvNeXt-V2-Tiny_lr-0.001_wd-0.01_bs-128_epochs-1000_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_mse.pth',
        },
        'ConvNeXt-V2-Small': {
            'params': 49453156,
            'metrics_path': '',
            # 'metrics_path': 'results/dispersion_all_models_2/ConvNeXt-V2-Small_lr-0.001_wd-0.01_bs-128_epochs-1000_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_mse_metrics.zarr',
            'state_dict_path': 'results/dispersion_all_models_2/ConvNeXt-V2-Small_lr-0.001_wd-0.01_bs-128_epochs-1000_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_mse.pth',
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
            'state_dict_path': '',
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

for_plotting = {}

import os
import json
import subprocess
CACHE_FILE = "thesis_plots/all_family_data_dispersion_cache.json"
# Look for chache
if os.path.exists(CACHE_FILE):
    print("Loading for chached test errors...") 
    with open(CACHE_FILE, 'r') as f:
        cached_data = json.load(f)
else:
    cached_data = {}
    print("No cache found for test errors, will compute from state dicts if available.")

# Loop through all models, fetch val R2 from metrics, and test R2 from cache or by running test script if state dict exists
# cached data has format: {family: {size: {"params": int, "val_r2": float, "test_r2": float}}}
for family, sizes in data.items():
    for model, info in sizes.items():
        params = info['params']
        metrics_path = info['metrics_path']
        state_dict_path = info['state_dict_path']
        val_r2 = None
        test_r2 = None

        # look for val_r2 in cache
        if family in cached_data and model in cached_data[family] and cached_data[family][model]['val_r2'] is not None:
            val_r2 = cached_data[family][model]['val_r2']
            print(f"Loaded cached R2 for {model}: val {val_r2}")
        else: 
            # Load val R2 from metrics zarr
            if metrics_path and os.path.exists(metrics_path):
                try:
                    root = zarr.open(metrics_path, mode='r')
                    val_r2 = root['R2_val'][:]
                    val_r2 = np.max(val_r2)
                    print(f"Loaded val R2 for {model} from metrics: {val_r2:.5f}")
                except Exception as e:
                    print(f"Error loading metrics for {model}: {e}")
            else:
                print(f"No metrics path for {model}, skipping val R2.")
            
        
        # look for test_r2 in cache, if not found, check if state dict exists to run test script
        if family in cached_data and model in cached_data[family] and 'test_r2' in cached_data[family][model] and cached_data[family][model]['test_r2'] is not None:
            test_r2 = cached_data[family][model]['test_r2']
            print(f"Loaded cached test R2 for {model}: {test_r2}")
        elif state_dict_path and os.path.exists(state_dict_path):
            print(f"State dict found for {model}, running test script...")
            cmd = build_test_cmd(state_dict_path, model, family)
            print("Running:", " ".join(cmd))
            
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            match = re.search(r"Test R2:\s*([\d.]+)", result.stdout)
            
            if match:
                test_r2 = float(match.group(1))
                print(f"Test R2 for {model}: {test_r2:.5f}")
            else:
                print(f"No Test R2 found in output for {model}.")
        else:
            print(f"No state dict for {model}, skipping test R2.")

        # write results to cache
        if family not in cached_data:
            cached_data[family] = {}
        cached_data[family][model] = {
            "params": int(params),
            "val_r2": float(val_r2) if val_r2 is not None else None,
            "test_r2": float(test_r2) if test_r2 is not None else None,
        }

        for_plotting[family] = for_plotting.get(family, {})
        for_plotting[family][model] = (params, val_r2, test_r2)

# Save updated cache
with open(CACHE_FILE, 'w') as f:
    json.dump(cached_data, f, indent=2)
print(f"Saved updated cache to {CACHE_FILE}")


# Plotting
fig, axes = plt.subplots(2, 1, figsize=(figsize[0], figsize[1]*1.45), sharex=True, sharey=True)
legend_model_handles = []
for family, models in for_plotting.items():
    color = family_cmaps.get(family, 'C0')
    marker = family_markers.get(family, 'o')
    name = family_map.get(family, family)
    xs_val, ys_val = [], []
    xs_test, ys_test = [], []
    for model, (params, val_r2, test_r2) in models.items():
        if val_r2 is not None:
            xs_val.append(params)
            ys_val.append(1 - val_r2)
        if test_r2 is not None:
            xs_test.append(params)
            ys_test.append(1 - test_r2)

    if xs_val and ys_val:
        axes[0].plot(xs_val, ys_val, color=color, alpha=0.3, zorder=1)
        for x, y in zip(xs_val, ys_val):
            axes[0].plot(x, y, color=color, linestyle='', marker=marker,
                         markersize=7, fillstyle='none', zorder=2)

    if xs_test and ys_test:
        axes[1].plot(xs_test, ys_test, color=color, alpha=0.3, zorder=1)
        for x, y in zip(xs_test, ys_test):
            axes[1].plot(x, y, color=color, linestyle='', marker=marker,
                         markersize=7, fillstyle='none', zorder=2)

    legend_model_handles.append(
        Line2D([0], [0], marker=marker, color=color,
               label=name, markersize=8, fillstyle='none', linestyle='-')
    )
    
for ax in axes:
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, which="both", ls="-", alpha=0.15)

axes[0].set_ylabel(r'Validation $1 - R^2$')
axes[1].set_ylabel(r'Test $1 - R^2$')
axes[1].set_xlabel('Total Parameters')

axes[0].legend(handles=legend_model_handles, title="Architectures",
               loc='best', fontsize=8)
plt.tight_layout()
plt.savefig('thesis_plots/scaling_laws_r2_vs_params_dispersion.pdf')
plt.close()
print("Saved thesis_plots/scaling_laws_r2_vs_params_dispersion.pdf")