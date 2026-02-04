import matplotlib.pyplot as plt
import numpy as np
import zarr
from matplotlib.lines import Line2D

folder = 'results/lf_pe_sweep/'

models = ['ResNet-18']
loss_functions = ['mse','rmse','huber','log-cosh','rse']
pe = [0,1,2,3,4]

# Color per loss function
loss_colors = {
    'mse': 'C0',
    'L1': 'C5',
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
pe_markers = {
    0: 'o',
    1: 's',
    2: '^',
    3: 'D',
    4: 'x',
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
    1, len(pe),
    figsize=(1.6 * len(pe), 3.8),
    sharex=True,
    sharey=True,
)

if len(pe) == 1:
    axes = [axes]

for col, p in enumerate(pe):
    ax = axes[col]

    for l in loss_functions:
        path = folder + f'{models[0]}_lr-0.0003_wd-0.3_bs-128_epochs-200_cosine_warmup-625.0_clipgrad-True_pe-encoder-None_pe-{p}_{l}_metrics.zarr'
        try:
            root = zarr.open(path, mode='r')
            train_r2 = root['R2_train'][:]
            val_r2 = root['R2_val'][:]
        except Exception as e:
            print(f"Skipping {path}: {e}")
            continue

        # Train
        ax.plot(
            1 - train_r2,
            color=loss_colors[l],
            linestyle=split_styles['train'],
            alpha=0.35,
        )

        # Val
        ax.plot(
            1 - val_r2,
            color=loss_colors[l],
            linestyle=split_styles['val'],
        )

    ax.set_title(f'PE = {p}')
    ax.set_yscale('log')
    ax.set_xlabel('Epoch')
    ax.grid(alpha=0.3)

axes[0].set_ylabel(r'$1 - R^2$')


# Legends
loss_legend = [
    Line2D([0], [0], color=loss_colors[l], lw=2, label=l)
    for l in loss_functions
]

split_legend = [
    Line2D([0], [0], color='black', linestyle=split_styles[s], lw=2, label=s)
    for s in split_styles
]
pe_legend = [
    Line2D([0], [0], marker=pe_markers[p], color='black', lw=0, label=f'pe={p}')
    for p in pe
]

loss_legend = [
    Line2D([0], [0], color=loss_colors[l], lw=2, label=l)
    for l in loss_functions
]

split_legend = [
    Line2D([0], [0], color='black', linestyle=split_styles[s], lw=2, label=s)
    for s in split_styles
]

fig.legend(
    handles=loss_legend,
    title="Loss Function",
    loc="upper right",
    ncol=len(loss_functions),
    frameon=False
)

fig.legend(
    handles=split_legend,
    title="Split",
    loc="upper left",
    ncol=len(split_styles),
    frameon=False
)


plt.tight_layout(rect=[0, 0, 1.0, 0.9])
plt.savefig('thesis_plots/lf_pe_sweep.pdf')
