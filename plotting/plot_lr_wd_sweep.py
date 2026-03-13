import matplotlib.pyplot as plt
import numpy as np
import zarr
from matplotlib.lines import Line2D
from ploting import figsize
from matplotlib.gridspec import GridSpec

'''
rsync -av   --include='*metrics.zarr/***'   --exclude='*' bigfacet:/home/users/sigursv/porousflow/results/permeability_lr_wd_sweep/  results/permeability_lr_wd_sweep/

'''

lrs = [1e-3, 5e-4, 1e-4, 5e-5]
wds = [5e-1, 1e-1, 5e-2, 1e-2]


plt.rcParams.update({
    "font.size":       9,
    "axes.labelsize":  9,
    "axes.titlesize":  9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.8,
    # "pdf.fonttype":    42,
    # "ps.fonttype":     42,
})


models = ['ConvNeXt-Atto', 'ResNet-18', 'ViT-T16', 'Swin-T']

# Permeability

fig = plt.figure(figsize=(figsize[0], figsize[1]))
outer = GridSpec(1, 2, width_ratios=[20, 0.64], wspace=0.05)
inner = outer[0].subgridspec(2, 2, wspace=0.1, hspace=0.35)

axs = []
axs.append(fig.add_subplot(inner[0, 0]))
for row in range(2):
    for col in range(2):
        if row == 0 and col == 0:
            continue
        ax = fig.add_subplot(inner[row, col])
        axs.append(ax)

# Colorbar axis
cax = fig.add_subplot(outer[1])

grids = []
for k in range(4):
    model = models[k]
    grid = np.full((len(lrs), len(wds)), np.nan)
    for j, lr in enumerate(lrs):
        for i, wd in enumerate(wds):
            path = (
                f'results/permeability_lr_wd_sweep/{model}_lr-{lr}_wd-{wd}_bs-128_epochs-200_cosine_warmup-3750.0_clipgrad-True_pe-encoder-None_pe-None_mse_metrics.zarr'
            )
            try:
                root = zarr.open(path, mode='r')
                disp_val = 1 - root['R2_val'][:]
                grid[j, i] = np.min(disp_val)
            except Exception as e:
                print(f"Skipping {path}: {e}")
    grids.append(grid)

vmin = np.nanmin([g.min() for g in grids])
vmax = np.nanmax([g.max() for g in grids])

# Second pass: plot
for k in range(4):
    model = models[k]
    grid = grids[k]
    ax = axs[k]
    row, col = divmod(k, 2)
    im = ax.imshow(grid, aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
    for j in range(len(lrs)):
        for i in range(len(wds)):
            if not np.isnan(grid[j, i]):
                ax.text(i, j, f'{grid[j, i]:.4f}', ha='center', va='center',
                        fontsize=7, color='white' if grid[j, i] < (vmin + vmax) / 2 else 'black')
    ax.set_xticks(range(len(wds)))
    ax.set_yticks(range(len(lrs)))
    ax.set_yticklabels([f'{lr:.0e}' for lr in lrs], fontsize=7)
    if col == 0:
        ax.set_ylabel('Learning rate')
        ax.set_yticklabels([f'{lr:.0e}' for lr in lrs], fontsize=7)
    else:
        plt.setp(ax.get_yticklabels(), visible=False)
    if row == 1:
        ax.set_xticklabels([f'{wd:.0e}' for wd in wds], fontsize=7)
        ax.set_xlabel('Weight decay')
    else:
        ax.set_xticklabels([])
    ax.set_title(model)

fig.colorbar(im, cax=cax, label=r'Lowest $1-R^2$')
plt.savefig('thesis_plots/permeability_lr_wd_sweep.pdf', bbox_inches='tight')
plt.close(fig)

# Dispersion