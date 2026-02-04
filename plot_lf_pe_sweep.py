import matplotlib.pyplot as plt
import numpy as np
import zarr
from matplotlib.lines import Line2D

folder = 'results/lf_pe_sweep/'

models = ['ResNet-18']
loss_functions = ['mse','rmse','huber','log-cosh','rse']
pe = [0,1,2,3,4]

# Color per PE
pe_colors = {
    0: 'C0',
    1: 'C1',
    2: 'C2',
    3: 'C3',
    4: 'C4',
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
    1, len(loss_functions),
    figsize=(1.6 * len(loss_functions), 3.8),
    sharex=True,
    sharey=True,
)

if len(loss_functions) == 1:
    axes = [axes]

# Loop over loss function (subplot = loss)
for col, l in enumerate(loss_functions):
    ax = axes[col]

    # Loop over PE (curves = PE)
    for p in pe:
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
            color=pe_colors[p],
            linestyle=split_styles['train'],
            alpha=0.35,
        )

        # Val
        ax.plot(
            1 - val_r2,
            color=pe_colors[p],
            linestyle=split_styles['val'],
        )

    ax.set_title(f'{l}')
    ax.set_yscale('log')
    ax.set_xlabel('Epoch')
    ax.grid(alpha=0.3)

axes[0].set_ylabel(r'$1 - R^2$')

# Legends
pe_legend = [
    Line2D([0], [0], color=pe_colors[p], lw=2, label=f'PE = {p}')
    for p in pe
]

split_legend = [
    Line2D([0], [0], color='black', linestyle=split_styles[s], lw=2, label=s)
    for s in split_styles
]

fig.legend(
    handles=pe_legend,
    title="PE",
    loc="upper right",
    ncol=len(pe),
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
