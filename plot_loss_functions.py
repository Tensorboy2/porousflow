import matplotlib.pyplot as plt
import numpy as np
import zarr
from matplotlib.lines import Line2D

folder = 'results/loss_function_sweep/'

models = ['ConvNeXt-Atto', 'ResNet-18', 'ViT-T16']
loss_functions = ['mse','rmse','huber','log-cosh','rse']

# Color per loss function
loss_colors = {
    'mse': 'C0',
    'rmse': 'C1',
    'huber': 'C2',
    'log-cosh': 'C3',
    'rse': 'C4',
}

# Linestyle per split
split_styles = {
    'train': '--',
    'val': '-'
}

fig, axes = plt.subplots(
    2, len(models),
    figsize=(6.4 * len(models) / 2, 8.0),
    sharex=True,
    sharey='row'
)

if len(models) == 1:
    axes = np.array([[axes[0]], [axes[1]]])

for col, m in enumerate(models):
    ax_loss = axes[0, col]
    ax_r2 = axes[1, col]

    for l in loss_functions:
        path = folder + f'{m}_lr-0.0008_wd-0.1_bs-128_epochs-100_cosine_warmup-250_clipgrad-True_pe-encoder-None_pe-None_{l}_metrics.zarr'

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
        ax_loss.plot(train_loss, color=loss_colors[l], linestyle=split_styles['train'],alpha=0.3)
        ax_loss.plot(val_loss, color=loss_colors[l], linestyle=split_styles['val'])

        # Row 2: R2
        ax_r2.plot(1-train_r2, color=loss_colors[l], linestyle=split_styles['train'],alpha=0.3)
        ax_r2.plot(1-val_r2, color=loss_colors[l], linestyle=split_styles['val'])

    ax_loss.set_title(m)
    ax_loss.set_yscale('log')

    # ax_r2.set_ylim(-0.05, 1.05)
    ax_r2.set_xlabel('Epoch')
    ax_r2.set_yscale('log')

axes[0, 0].set_ylabel('Loss')
axes[1, 0].set_ylabel(r'$1-R^2$')

# Legends
loss_legend = [
    Line2D([0], [0], color=loss_colors[l], lw=2, label=l)
    for l in loss_functions
]

split_legend = [
    Line2D([0], [0], color='black', linestyle=split_styles[s], lw=2, label=s)
    for s in split_styles
]

leg1 = fig.legend(
    handles=loss_legend,
    title="Loss Function",
    loc="upper center",
    ncol=len(loss_functions),
    frameon=False
)

fig.legend(
    handles=split_legend,
    title="Split",
    loc="lower center",
    ncol=len(split_styles),
    frameon=False
)

plt.tight_layout(rect=[0, 0.06, 1, 0.93])
plt.savefig('thesis_plots/loss_functions.pdf')
