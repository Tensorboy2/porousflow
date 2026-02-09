import matplotlib.pyplot as plt
import numpy as np
import zarr
from matplotlib.lines import Line2D

folder = 'results/vit_long_run/'

models = ['ViT-T16','ViT-S16','ViT-B16','ViT-L16']
length = [2000,1500,1000]

# Color per loss function
length_colors = {
    2000: 'C2',
    1500: 'C1',
    1000: 'C0',
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
fig, axes = plt.subplots(
    1, len(models),
    figsize=(3.5 * 1.6, 3.2),
    sharex=True,
    sharey='row'
)

if len(models) == 1:
    axes = np.array([[axes[0]], [axes[1]]])

for col, m in enumerate(models):
    # ax_loss = axes[0, col]
    ax_r2 = axes[col]

    for l in length:
        path = folder + f'{m}_lr-0.0008_wd-0.1_bs-128_epochs-{l}_cosine_warmup-250_clipgrad-True_pe-encoder-None_pe-None_metrics.zarr'

        try:
            root = zarr.open(path, mode='r')
            train_loss = root['train_loss'][:]
            val_loss = root['val_loss'][:]
            train_r2 = root['R2_train'][:]
            val_r2 = root['R2_val'][:]
        except Exception as e:
            print(f"Skipping {path}: {e}")
            continue

        # Row 1: Loss
        # ax_loss.plot(train_loss, color=length_colors[l], linestyle=split_styles['train'],alpha=0.3)
        # ax_loss.plot(val_loss, color=length_colors[l], linestyle=split_styles['val'],alpha=0.9)

        # Row 2: R2
        ax_r2.plot(1-train_r2, color=length_colors[l], linestyle=split_styles['train'],alpha=0.3)
        ax_r2.plot(1-val_r2, color=length_colors[l], linestyle=split_styles['val'],alpha=0.9)

    ax_r2.set_title(m)
    # ax_loss.set_yscale('log')
    # # ax_loss.set_xscale('log')
    # ax_loss.grid(alpha=0.3)
    # ax_loss.set_xlim(90, 100)

    # ax_r2.set_xlim(100, 2200)
    ax_r2.set_xlabel('Epoch')
    ax_r2.set_yscale('log')
    # ax_r2.set_xscale('log')
    ax_r2.grid(alpha=0.3)

# axes[0, 0].set_ylabel('Loss')
axes[0].set_ylabel(r'$1-R^2$')

# Legends
loss_legend = [
    Line2D([0], [0], color=length_colors[l], lw=2, label=l)
    for l in length
]

split_legend = [
    Line2D([0], [0], color='black', linestyle=split_styles[s], lw=2, label=s)
    for s in split_styles
]

leg1 = fig.legend(
    handles=loss_legend,
    title="Training length",
    loc="lower left",
    ncol=len(length),
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
plt.savefig('thesis_plots/vit_long_run.pdf')
