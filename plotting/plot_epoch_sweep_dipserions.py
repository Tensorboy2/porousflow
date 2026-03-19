import matplotlib.pyplot as plt
import numpy as np
import zarr
from matplotlib.lines import Line2D
from ploting import figsize
import matplotlib as mpl
mpl.rcParams['axes.formatter.use_mathtext'] = True

folder = 'results/epoch_sweep_all_models/'
models = {
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
    "ConvNeXt-V2": [3388604, 4849684, 8555204, 14985844, 27871588, 49561444, 87708804, 196443844],
    "ConvNeXt-RMS": [3371724, 4829428, 8528196, 14946324, 27811204, 49438852, 87545348, 196198660],
    "ConvNeXt": [3373884, 4832020, 8531652, 14951284, 27818596, 49453156, 87564420, 196227268],
    "ResNet": [11172292, 21280452, 23509956, 42502084, 58145732],
    "ViT": [5401156, 21419140, 85305604, 302644228],
    "Swin": [27504334, 48804958, 86700156, 194930872]
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
            else:
                path = (
                    f'results/dispersion_epoch_sweep/{m}_lr-0.005_wd-0.01_bs-128_epochs-{l}_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_mse_metrics.zarr'
                )
        print(path)
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