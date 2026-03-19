import matplotlib.pyplot as plt
import numpy as np
import zarr
from matplotlib.lines import Line2D
from ploting import figsize
from matplotlib.gridspec import GridSpec

'''
rsync -av   --include='*metrics.zarr/***'   --exclude='*' bigfacet:/home/users/sigursv/porousflow/results/permeability_lr_wd_sweep/  results/permeability_lr_wd_sweep/

'''

# lrs = [1e-3, 5e-4, 1e-4, 5e-5]
# wds = [5e-1, 1e-1, 5e-2, 1e-2]
lrs = [5e-3, 1e-3, 5e-4, 1e-4, 5e-5]
wds = [5e-1, 1e-1, 5e-2, 1e-2, 1e-3]


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
def fmt_sci(val):
    exp = int(f'{val:.0e}'.split('e')[1])
    coef = val / 10**exp
    return rf'${coef:.0f}\cdot 10^{{{exp}}}$'

models = ['ConvNeXt-Atto', 'ResNet-18', 'ViT-T16', 'Swin-T']
loss_functions = ['mse', 'mse', 'mse', 'mse']

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
import matplotlib.colors as mcolors
vmin = np.nanmin([g.min() for g in grids])
vmax = np.nanmax([g.max() for g in grids])
norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
# Second pass: plot
for k in range(4):
    model = models[k]
    grid = grids[k]
    ax = axs[k]
    row, col = divmod(k, 2)
    im = ax.imshow(grid, aspect='auto', cmap='viridis', norm=norm)
    # im = ax.imshow(grid, aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
    for j in range(len(lrs)):
        for i in range(len(wds)):
            if not np.isnan(grid[j, i]):
                ax.text(i, j, f'{grid[j, i]:.4f}', ha='center', va='center',
                        fontsize=7, color='white' if grid[j, i] < ((vmin + vmax) / 24) else 'black')
    ax.set_xticks(range(len(wds)))
    ax.set_yticks(range(len(lrs)))
    ax.set_yticklabels([fmt_sci(lr) for lr in lrs], fontsize=7)
    if col == 0:
        ax.set_ylabel('Learning rate')
        ax.set_yticklabels([fmt_sci(lr) for lr in lrs], fontsize=7)
    else:
        plt.setp(ax.get_yticklabels(), visible=False)
    if row == 1:
        ax.set_xticklabels([fmt_sci(wd) for wd in wds], fontsize=7)
        ax.set_xlabel('Weight decay')
    else:
        ax.set_xticklabels([])
    ax.set_title(model)
    # ax.xaxis.set_major_formatter(formatter)
    # ax.yaxis.set_major_formatter(formatter)

fig.colorbar(im, cax=cax, label=r'Lowest $1-R^2$')
plt.savefig('thesis_plots/permeability_lr_wd_sweep.pdf', bbox_inches='tight')
plt.close(fig)

# Dispersion
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
            if model=='Swin-T':
                path = (
                    f'results/dispersion_lr_wd_sweep/{model}_lr-{lr}_wd-{wd}_bs-128_epochs-200_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_rmse_metrics.zarr'
                )
            # elif model=='ViT-T16':
            #     path = (
            #         f'results/dispersion_lr_wd_sweep_2/{model}_lr-{lr}_wd-{wd}_bs-128_epochs-200_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_log-cosh_metrics.zarr'
            #     )
            else:
                path = (
                    f'results/dispersion_lr_wd_sweep/{model}_lr-{lr}_wd-{wd}_bs-128_epochs-200_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_mse_metrics.zarr'
                )
            try:
                root = zarr.open(path, mode='r')
                disp_val = 1 - root['R2_val'][:]
                grid[j, i] = np.min(disp_val)
            except Exception as e:
                print(f"Skipping {path}: {e}")
    grids.append(grid)
import matplotlib.colors as mcolors
vmin = np.nanmin([g.min() for g in grids])
vmax = np.nanmax([g.max() for g in grids])
norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
# Second pass: plot
for k in range(4):
    model = models[k]
    grid = grids[k]
    ax = axs[k]
    row, col = divmod(k, 2)
    # im = ax.imshow(grid, aspect='auto', cmap='viridis', vmin=vmin,vmax=vmax)
    im = ax.imshow(grid, aspect='auto', cmap='viridis', norm=norm)
    for j in range(len(lrs)):
        for i in range(len(wds)):
            if not np.isnan(grid[j, i]):
                ax.text(i, j, f'{grid[j, i]:.4f}', ha='center', va='center',
                        fontsize=7, color='white' if grid[j, i] < (vmin + vmax) / 4 else 'black')
            else:
                ax.text(i, j, 'NaN', ha='center', va='center',
                        fontsize=7, color= 'black')
    ax.set_xticks(range(len(wds)))
    ax.set_yticks(range(len(lrs)))
    ax.set_yticklabels([fmt_sci(lr) for lr in lrs], fontsize=7)
    if col == 0:
        ax.set_ylabel('Learning rate')
        ax.set_yticklabels([fmt_sci(lr) for lr in lrs], fontsize=7)
    else:
        plt.setp(ax.get_yticklabels(), visible=False)
    if row == 1:
        ax.set_xticklabels([fmt_sci(wd) for wd in wds], fontsize=7)
        ax.set_xlabel('Weight decay')
    else:
        ax.set_xticklabels([])
    ax.set_title(model)

cbar = fig.colorbar(im, cax=cax, label=r'Lowest $1-R^2$')
cbar = fig.colorbar(im, cax=cax, label=r'Lowest $1-R^2$')
cbar.set_ticks([0.2, 0.1,0.05])
cbar.set_ticklabels(['0.2', '0.1', '0.05'])
# cbar.set_ticks([0.3,0.2,0.1])
plt.savefig('thesis_plots/dispersion_lr_wd_sweep.pdf', bbox_inches='tight')
plt.close(fig)


fig = plt.figure(figsize=(figsize[0], figsize[1]*0.7))
outer = GridSpec(1, 2, width_ratios=[20, 0.64], wspace=0.05)
inner = outer[0].subgridspec(1, 1, wspace=0.1, hspace=0.35)

axs = []
axs.append(fig.add_subplot(inner[0, 0]))
for row in range(1):
    for col in range(1):
        if row == 0 and col == 0:
            continue
        ax = fig.add_subplot(inner[row, col])
        axs.append(ax)

# Colorbar axis
cax = fig.add_subplot(outer[1])
k=2
model = models[k]
grid = grids[k]
ax = axs
model = models[k]
grid = grids[k]
for j, lr in enumerate(lrs):
    for i, wd in enumerate(wds):
        path = (
            f'results/dispersion_lr_wd_sweep_2/{model}_lr-{lr}_wd-{wd}_bs-128_epochs-200_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_log-cosh_metrics.zarr'
        )
        try:
            root = zarr.open(path, mode='r')
            disp_val = 1 - root['R2_val'][:]
            grid[j, i] = np.min(disp_val)
        except Exception as e:
            print(f"Skipping {path}: {e}")

import matplotlib.colors as mcolors
vmin = np.nanmin([g.min() for g in grids])
vmax = np.nanmax([g.max() for g in grids])
norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
ax = axs[0]
row, col = divmod(k, 2)
# im = ax.imshow(grid, aspect='auto', cmap='viridis', vmin=vmin,vmax=vmax)
im = ax.imshow(grid, aspect='auto', cmap='viridis', norm=norm)
for j in range(len(lrs)):
    for i in range(len(wds)):
        if not np.isnan(grid[j, i]):
            ax.text(i, j, f'{grid[j, i]:.4f}', ha='center', va='center',
                    fontsize=7, color='white' if grid[j, i] < (vmin + vmax) / 4 else 'black')
        else:
            ax.text(i, j, 'NaN', ha='center', va='center',
                    fontsize=7, color= 'black')
ax.set_xticks(range(len(wds)))
ax.set_yticks(range(len(lrs)))
ax.set_yticklabels([fmt_sci(lr) for lr in lrs], fontsize=7)
if col == 0:
    ax.set_ylabel('Learning rate')
    ax.set_yticklabels([fmt_sci(lr) for lr in lrs], fontsize=7)
else:
    plt.setp(ax.get_yticklabels(), visible=False)
if row == 1:
    ax.set_xticklabels([fmt_sci(wd) for wd in wds], fontsize=7)
    ax.set_xlabel('Weight decay')
else:
    ax.set_xticklabels([])
ax.set_title(model+' log-cosh')
cbar = fig.colorbar(im, cax=cax, label=r'Lowest $1-R^2$')
# cbar = fig.colorbar(im, cax=cax, label=r'Lowest $1-R^2$')
cbar.set_ticks([0.2, 0.1,0.05])
cbar.set_ticklabels(['0.2', '0.1', '0.05'])
# cbar.set_ticks([0.3,0.2,0.1])
plt.savefig('thesis_plots/dispersion_lr_wd_sweep_vit_log-cosh.pdf', bbox_inches='tight')
plt.close(fig)