import matplotlib.pyplot as plt
import numpy as np
import zarr
from matplotlib.lines import Line2D

folder = 'results//'
models = ['ConvNeXt-RMS-Atto']
paths= [
    'results/rms_atto_dispersion/ConvNeXt-RMS-Atto_lr-0.0005_wd-0.05_bs-128_epochs-1000_cosine_warmup-0.0_clipgrad-True_pe-encoder-straight_pe-4_rmse_metrics.zarr',
    'results/rms_atto_dispersion/ConvNeXt-RMS-Atto_lr-0.00025_wd-0.005_bs-128_epochs-400_cosine_warmup-0.0_clipgrad-True_pe-encoder-straight_pe-4_rmse_metrics.zarr',
    'results/rms_atto_dispersion/ConvNeXt-RMS-Atto_lr-0.00025_wd-0.1_bs-128_epochs-400_cosine_warmup-0.0_clipgrad-True_pe-encoder-straight_pe-4_mse_metrics.zarr',
    # 'results/pe_encoder_sweep_convnext/ConvNeXt-Atto_lr-0.0008_wd-0.01_bs-128_epochs-200_cosine_warmup-1000_clipgrad-True_pe-encoder-log_metrics.zarr',
    # 'results/pe_encoder_sweep_convnext/ConvNeXt-Atto_lr-0.0008_wd-0.01_bs-128_epochs-200_cosine_warmup-1000_clipgrad-True_pe-encoder-vector_metrics.zarr',
    'results/pe_encoder_sweep_convnext/ConvNeXt-Atto_lr-0.0008_wd-0.01_bs-128_epochs-200_cosine_warmup-1000_clipgrad-True_pe-encoder-straight_metrics.zarr',
    'results/rms_atto_dispersion_no_asinh_test/ConvNeXt-RMS-Atto_lr-0.0005_wd-0.05_bs-128_epochs-200_cosine_warmup-0.0_clipgrad-True_pe-encoder-straight_pe-4_rmse_metrics.zarr',
]

colors = ['C0','C1','C2','C3','C4']
labels = ['wd=0.05','wd=0.005','wd=0.1','old aug','wd=0.05']

# Linestyle per split
split_styles = {
    'train': '--',
    'val': '-'
}

plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
})

fig, axes = plt.subplots(
    1, 1,
    figsize=(6.4, 3.2),
    # sharex=True,
    # sharey=True
)
for i,path in enumerate(paths):
    try:
        root = zarr.open(path, mode='r')
        train_loss = root['train_loss'][:]
        val_loss = root['val_loss'][:]
        train_r2 = root['R2_train'][:]
        val_r2 = root['R2_val'][:]
        try:
            grad_norm = root['grad_norm'][:]
        except KeyError:
            grad_norm = np.full_like(train_loss, np.nan)
    except Exception as e:
        print(f"Skipping {path}: {e}")
        continue

    # Plot R2
    axes.plot(1-train_r2, color=colors[i], linestyle=split_styles['train'], alpha=0.3)
    axes.plot(1-val_r2, color=colors[i], linestyle=split_styles['val'], alpha=1.,linewidth=1.5,label=labels[i])

    # r_axes = axes.twinx()
    # plt.plot(grad_norm, color=colors[i], linestyle='-', alpha=0.5)

plt.title('ConvNeXt-RMS-Atto dispersion')
plt.xlabel('Epochs')
plt.ylabel('Dispersion (1 - R²)')
plt.yscale('log')
# plt.xscale('log')
# plt.xlim(10,450)
# plt.ylim(0.5,1.001)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(f'thesis_plots/dispersion_training.pdf')
plt.close()