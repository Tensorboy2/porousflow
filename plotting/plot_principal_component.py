import matplotlib.pyplot as plt
import numpy as np
import zarr
from matplotlib.lines import Line2D

folders = ['results/principal_component_test_dispersion/',
           'results/principal_component_test_dispersion_with_asinh/']


Pes = [0,1,2,3,4]

# Color per length
length_colors = {
    0: 'C4',
    1: 'C3',
    2: 'C2',
    3: 'C1',
    4: 'C0',
}

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

# Create a separate plot for each model family
fig, axes = plt.subplots(
    1, 2,
    figsize=(3.2 * 1.6, 3.2),
    sharex=True,
    sharey=True
)
titles = ['Without asinh', 'With asinh']

for i,folder in enumerate(folders):
#results/principal_component_test_dispersion/ResNet-34_lr-0.0005_wd-0.3_bs-128_epochs-500_cosine_warmup-625.0_clipgrad-True_pe-encoder-None_pe-0_rmse_metrics.zarr
    for p in Pes:
        path = folder + f'ResNet-34_lr-0.0005_wd-0.3_bs-128_epochs-500_cosine_warmup-625.0_clipgrad-True_pe-encoder-None_pe-{p}_rmse_metrics.zarr'
        try:
            root = zarr.open(path, mode='r')
            train_loss = root['train_loss'][:]
            val_loss = root['val_loss'][:]
            train_r2 = root['R2_train'][:]
            val_r2 = root['R2_val'][:]
        except Exception as e:
            print(f"Skipping {path}: {e}")
            continue
        
        # Plot R2
        # axes[i].plot(1 - train_r2, color=length_colors[p], linestyle=split_styles['train'], alpha=0.3)
        axes[i].plot(1 - val_r2, color=length_colors[p], linestyle=split_styles['val'], alpha=1.,linewidth=1.)
    
    axes[i].set_title(titles[i])
    axes[i].set_xlabel('Epoch')
    axes[i].set_yscale('log')
    axes[i].set_xscale('log')
    axes[i].set_xlim(10, 510)
    axes[i].grid(alpha=0.3)

axes[0].set_ylabel(r'$1-R^2$')

# Legends
loss_legend = [
    Line2D([0], [0], color=length_colors[l], lw=2, label=l)
    for l in Pes
]
split_legend = [
    Line2D([0], [0], color='black', linestyle=split_styles[s], lw=2, label=s)
    for s in split_styles
]

leg1 = fig.legend(
    handles=loss_legend,
    title="Training length",
    loc="lower left",
    ncol=len(Pes),
    frameon=False
)
fig.legend(
    handles=split_legend,
    title="Split",
    loc="lower right",
    ncol=len(split_styles),
    frameon=False
)

plt.tight_layout(rect=[0, 0.10, 1.0, 1.0])
plt.savefig(f'thesis_plots/principal_component_dispersion.pdf')
plt.close()