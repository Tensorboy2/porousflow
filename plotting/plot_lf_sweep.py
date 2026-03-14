import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
import numpy as np
import zarr
from ploting import figsize


'''
rsync -av   --include='*metrics.zarr/***'   --exclude='*' herbie-jump:/home/sigursv/porousflow/results/lf_sweep_dispersion/  results/lf_sweep_dispersion/
rsync -av   --include='*metrics.zarr/***'   --exclude='*' bigfacet:/home/users/sigursv/porousflow/results/lf_sweep_permeability/  results/lf_sweep_permeability/
'''
models = ['ConvNeXt-Atto', 'ResNet-18', 'ViT-T16', 'Swin-T']
loss_functions = ['mse','rmse', 'L1','huber','log-cosh','rse']

plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
})

cmap = mpl.colormaps['tab10']

n = len(loss_functions)
colors = cmap.resampled(n).colors
split_styles = {
    'validation': '-',
    'train': '--'
}
print('Permeability:')
# Permeability:
fig,axs = plt.subplots(1,4,figsize=(figsize[0],figsize[1]*0.8), sharey=True)

for i,m in enumerate(models):
    higest_r2 = -float('inf')
    best_lf = None
    for j, lf in enumerate(loss_functions):
        path = f'results/lf_sweep_permeability/{m}_lr-0.0005_wd-0.05_bs-128_epochs-200_cosine_warmup-3750.0_clipgrad-True_pe-encoder-None_pe-None_{lf}_metrics.zarr'
        try:
            root = zarr.open(path, mode='r')
            train_r2 = root['R2_train'][:]
            val_r2 = root['R2_val'][:]
        except Exception as e:
            print(f"Skipping {path}: {e}")
            continue
        current_max_r2 = max(val_r2)
        if current_max_r2>higest_r2:
            higest_r2=current_max_r2
            best_lf = lf


        # Train
        axs[i].plot(
            1 - train_r2,
            color=colors[j],
            linestyle=split_styles['train'],
            alpha=0.2,
            lw=0.7,
        )

        # Val
        axs[i].plot(
            1 - val_r2,
            color=colors[j],
            linestyle=split_styles['validation'],
            alpha=0.8,
            lw=1.,
        )
    axs[i].set_yscale('log')
    axs[i].set_xlabel('Epochs')
    axs[i].set_title(m)
    axs[i].grid(alpha=0.3)

    print(f'    Model: {m}, R2: {higest_r2:.5f}, best lf: {best_lf}')

axs[0].set_ylabel(r'$1-R^2$')

# Legend
pe_legend = [
    Line2D([0], [0], color=colors[k], lw=2, label=lf)
    for k,lf in enumerate(loss_functions)
]

fig.legend(
    handles=pe_legend,
    title="Loss function",
    loc="lower center",
    ncol=n,
    frameon=False
)

plt.tight_layout(rect=[0, 0.1, 1.0, 1.0])
plt.savefig('thesis_plots/lf_sweep_permeability.pdf')
print()
print('Dispersion:')
# Dispersion:
fig,axs = plt.subplots(1,4,figsize=(figsize[0],figsize[1]*0.8), sharey=True)
# results/lf_sweep_dispersion/ConvNeXt-Atto_lr-0.0005_wd-0.05_bs-128_epochs-200_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_huber_metrics.zarr
for i,m in enumerate(models):
    higest_r2 = -float('inf')
    best_lf = None
    for j, lf in enumerate(loss_functions):
        path = f'results/lf_sweep_dispersion/{m}_lr-0.0005_wd-0.05_bs-128_epochs-200_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_{lf}_metrics.zarr'
        try:
            root = zarr.open(path, mode='r')
            train_r2 = root['R2_train'][:]
            val_r2 = root['R2_val'][:]
        except Exception as e:
            print(f"Skipping {path}: {e}")
            continue

        current_max_r2 = max(val_r2)
        if current_max_r2>higest_r2:
            higest_r2=current_max_r2
            best_lf = lf

        # Train
        axs[i].plot(
            1 - train_r2,
            color=colors[j],
            linestyle=split_styles['train'],
            alpha=0.2,
            lw=0.7,
        )

        # Val
        axs[i].plot(
            1 - val_r2,
            color=colors[j],
            linestyle=split_styles['validation'],
            alpha=0.8,
            lw=1.,
        )
    axs[i].set_yscale('log')
    axs[i].set_xlabel('Epochs')
    axs[i].set_title(m)
    axs[i].grid(alpha=0.3)

    print(f'    Model: {m}, R2: {higest_r2:.5f}, best lf: {best_lf}')


axs[0].set_ylabel(r'$1-R^2$')

# Legend
pe_legend = [
    Line2D([0], [0], color=colors[k], lw=2, label=lf)
    for k,lf in enumerate(loss_functions)
]

fig.legend(
    handles=pe_legend,
    title="Loss function",
    loc="lower center",
    ncol=n,
    frameon=False
)

plt.tight_layout(rect=[0, 0.1, 1.0, 1.0])
plt.savefig('thesis_plots/lf_sweep_dispersion.pdf')