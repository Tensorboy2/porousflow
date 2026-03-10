import matplotlib.pyplot as plt
import numpy as np
import zarr
from matplotlib.lines import Line2D
from ploting import figsize

lrs = [1e-3, 5e-4, 1e-4][::-1]
wds = [3e-1, 1e-1, 5e-2][::-1]

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

colors    = ['C0', 'C1', 'C2'] # one per lr
markers    = ['o', 's', '^']   # one per lr (color)
wd_markers = ['D', 'v', 'P']   # one per wd (replaces linestyles)
MARKER_EVERY = 20        # one per wd

LW_TRAIN    = 0.65
LW_VAL      = 0.65
ALPHA_TRAIN = 0.2
ALPHA_VAL   = 1.0

models = ['ConvNeXt-Atto', 'ResNet-18','ViT-T16']


fig, axs = plt.subplots(1, 3, figsize=figsize, sharey=True)

for k, ax in enumerate(axs.flatten()):
    for j, lr in enumerate(lrs):
        for i, wd in enumerate(wds):
            if models[k]=='ConvNeXt-Atto':
                path = (
                    f'results/dispersion_lr_wd_sweep/'
                    f'ConvNeXt-Atto_lr-{lr}_wd-{wd}_bs-128_epochs-500_cosine'
                    f'_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_rmse_metrics.zarr'
                )
            elif models[k]=='ViT-T16':
                path = (
                    f'results/dispersion_lr_wd_sweep/'
                    f'ViT-T16_lr-{lr}_wd-{wd}_bs-128_epochs-500_cosine'
                    f'_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_rmse_metrics.zarr'
                )
            elif models[k]=='ResNet-18':
                path = (
                    f'results/dispersion_lr_wd_sweep/'
                    f'ResNet-18_lr-{lr}_wd-{wd}_bs-128_epochs-500_cosine'
                    f'_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_rmse_metrics.zarr'
                )

            try:
                root = zarr.open(path, mode='r')
                train_r2 = root['R2_train'][:]
                val_r2   = root['R2_val'][:]
                print(max(val_r2))

                disp_train = 1 - train_r2
                disp_val   = 1 - val_r2

                color  = colors[j]          # j = lr index
                marker = wd_markers[i]      # i = wd index
                epochs = np.arange(len(disp_train))

                ax.plot(disp_train,
                        color=color, linestyle='-', linewidth=LW_TRAIN,
                        alpha=ALPHA_TRAIN,zorder=i)
                ax.plot(disp_val,
                        color=color, linestyle='-', linewidth=LW_VAL,
                        alpha=ALPHA_VAL,
                        marker=marker, markevery=MARKER_EVERY,
                        markersize=3.5, markerfacecolor=color, markeredgewidth=0.5,
                        markeredgecolor='white',zorder=i+10)

            except Exception as e:
                # pass
                print(f"Skipping {path}: {e}")
    ax.set_xlabel('Epoch')
    ax.set_yscale('log')
    # ax.set_xscale('log')
    ax.set_title(models[k])
    ax.grid(alpha=0.3)

axs[0].set_ylabel(r'$1 - R^2$')

lr_entries = [
    Line2D([0], [0], color=colors[i], lw=1.8, label=f'lr = {lr:.0e}')
    for i, lr in enumerate(lrs)
]
wd_entries = [
    Line2D([0], [0], color='k', lw=1.4,
           marker=wd_markers[j], markersize=5,
           markerfacecolor='k', markeredgecolor='white', markeredgewidth=0.5,
           label=f'wd = {wd:.0e}')
    for j, wd in enumerate(wds)
]
# Separator trick: blank entry between the two groups
separator = Line2D([0], [0], color='none', label='')

ax.legend(
    handles=lr_entries + [separator] + wd_entries,
    loc='lower right', frameon=True,
    framealpha=0.92, edgecolor='0.8', ncol=1,
)

plt.tight_layout()
plt.savefig('thesis_plots/dispersion_lr_wd_sweep.pdf', bbox_inches='tight')
plt.close(fig)

fig, axs = plt.subplots(1, 3, figsize=figsize, sharey=True)
colors_model = {'ConvNeXt-Atto': '#e05c2a', 'ResNet-18': '#2a7de0', 'ViT-T16': '#2abf5e'}

for k, ax in enumerate(axs.flatten()):
    model = models[k]
    best_val, best_curve, best_label = np.inf, None, ''

    # First pass: find best run
    for j, lr in enumerate(lrs):
        for i, wd in enumerate(wds):
            path = f'results/dispersion_lr_wd_sweep/{model}_lr-{lr}_wd-{wd}_bs-128_epochs-500_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_rmse_metrics.zarr'
            try:
                root = zarr.open(path, mode='r')
                val_r2 = root['R2_val'][:]
                disp_val = 1 - val_r2
                best_metric = np.min(disp_val)
                if best_metric < best_val:
                    best_val = best_metric
                    best_curve = disp_val
                    best_label = f'lr={lr:.0e}, wd={wd:.0e}'
            except Exception as e:
                print(f"Skipping {path}: {e}")

    # Second pass: plot all in grey
    for j, lr in enumerate(lrs):
        for i, wd in enumerate(wds):
            path = f'results/dispersion_lr_wd_sweep/{model}_lr-{lr}_wd-{wd}_bs-128_epochs-500_cosine_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_rmse_metrics.zarr'
            try:
                root = zarr.open(path, mode='r')
                disp_val = 1 - root['R2_val'][:]
                ax.plot(disp_val, color='0.75', linewidth=0.6, alpha=0.6, zorder=1)
            except:
                pass

    # Plot best on top
    if best_curve is not None:
        ax.plot(best_curve, color=colors_model[model], linewidth=1.,
                alpha=1.0, zorder=3, label=f'best: {best_label}')

    ax.set_xlabel('Epoch')
    ax.set_yscale('log')
    ax.set_title(model)
    ax.grid(alpha=0.3)
    ax.legend(loc='upper right', fontsize=7, framealpha=0.9, edgecolor='0.8')

axs[0].set_ylabel(r'$1 - R^2$')

# lr_entries = [
#     Line2D([0], [0], color=colors[i], lw=1.8, label=f'lr = {lr:.0e}')
#     for i, lr in enumerate(lrs)
# ]
# wd_entries = [
#     Line2D([0], [0], color='k', lw=1.4,
#            marker=wd_markers[j], markersize=5,
#            markerfacecolor='k', markeredgecolor='white', markeredgewidth=0.5,
#            label=f'wd = {wd:.0e}')
#     for j, wd in enumerate(wds)
# ]
# # Separator trick: blank entry between the two groups
# separator = Line2D([0], [0], color='none', label='')

# ax.legend(
#     handles=lr_entries + [separator] + wd_entries,
#     loc='lower right', frameon=True,
#     framealpha=0.92, edgecolor='0.8', ncol=1,
# )

plt.tight_layout()
plt.savefig('thesis_plots/dispersion_lr_wd_sweep_v2.pdf', bbox_inches='tight')
plt.close(fig)


# ── v3: heatmap grid (lr × wd) showing best val 1-R² per model ──────────
from matplotlib.gridspec import GridSpec

fig = plt.figure(figsize=(figsize[0],figsize[1]*0.6))
outer = GridSpec(1,2,width_ratios=[20,0.64],wspace=0.05)

inner = outer[0].subgridspec(1, 3, wspace=0.1)

axs = []
axs.append(fig.add_subplot(inner[0]))

for i in range(1, 3):
    axs.append(fig.add_subplot(inner[i], sharey=(axs[0] if True else None)))
    if True:
        plt.setp(axs[i].get_yticklabels(), visible=False)

# Colorbar axis
cax = fig.add_subplot(outer[1])

grids = []
for k in range(3):
    model = models[k]
    grid = np.full((len(lrs), len(wds)), np.nan)
    for j, lr in enumerate(lrs):
        for i, wd in enumerate(wds):
            path = (
                f'results/dispersion_lr_wd_sweep/{model}_lr-{lr}_wd-{wd}_bs-128_epochs-500_cosine'
                f'_warmup-18750.0_clipgrad-True_pe-encoder-log_pe-4_rmse_metrics.zarr'
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
for k in range(3):
    model = models[k]
    grid = grids[k]
    ax = axs[k]
    im = ax.imshow(grid, aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)

    for j in range(len(lrs)):
        for i in range(len(wds)):
            if not np.isnan(grid[j, i]):
                ax.text(i, j, f'{grid[j, i]:.3f}', ha='center', va='center',
                        fontsize=7, color='white' if grid[j, i] < (vmin + vmax) / 2 else 'black')

    ax.set_xticks(range(len(wds)))
    ax.set_xticklabels([f'{wd:.0e}' for wd in wds], fontsize=7)
    ax.set_yticks(range(len(lrs)))
    ax.set_yticklabels([f'{lr:.0e}' for lr in lrs], fontsize=7)
    ax.set_xlabel('Weight decay')
    ax.set_title(model)

axs[0].set_ylabel('Learning rate')

# Single shared colorbar
fig.colorbar(im, cax=cax, label=r'Lowest $1-R^2$')

plt.savefig('thesis_plots/dispersion_lr_wd_sweep_v3.pdf', bbox_inches='tight')
plt.close(fig)