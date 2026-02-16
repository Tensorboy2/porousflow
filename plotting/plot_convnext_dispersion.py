import matplotlib.pyplot as plt
import numpy as np
import zarr
from matplotlib.lines import Line2D

folder = 'results/convnext_v2_dispersion/'

models = [
    'ConvNeXt-V2-Atto',
    'ConvNeXt-V2-Femto',
    'ConvNeXt-V2-Pico',
    'ConvNeXt-V2-Nano',
    'ConvNeXt-V2-Tiny',
    'ConvNeXt-V2-Small',
    'ConvNeXt-V2-Base',
]

colors = [f'C{i}' for i in range(len(models))]

split_styles = {
    'train': '--',
    'val': '-'
}

plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
})

fig, ax = plt.subplots(figsize=(3.2, 3.2))

for m, color in zip(models, colors):

    path = (
        folder
        + f'{m}_lr-0.0005_wd-0.05_bs-128_epochs-200_'
        + 'cosine_warmup-1250.0_clipgrad-True_'
        + 'pe-encoder-straight_pe-4_rmse_metrics.zarr'
    )

    try:
        root = zarr.open(path, mode='r')
        train_r2 = root['R2_train'][:]
        val_r2 = root['R2_val'][:]
    except Exception as e:
        print(f"Skipping {m}: {e}")
        continue

    ax.plot(1 - train_r2,
            color=color,
            linestyle=split_styles['train'],
            alpha=0.3,
            linewidth=0.7)

    ax.plot(1 - val_r2,
            color=color,
            linestyle=split_styles['val'],
            linewidth=1.0,
            alpha=0.9,)

ax.set_xlabel("Epoch")
ax.set_ylabel(r"$1 - R^2$")
ax.set_yscale("log")
# ax.set_xscale("log")
# ax.set_xlim(10,210)
ax.grid(alpha=0.3)

# -------- Legends --------

model_handles = [
    Line2D([0], [0], color=c, lw=2, label=m.replace("ConvNeXt-V2-", ""))
    for m, c in zip(models, colors)
]
leg1 = ax.legend(handles=model_handles,
                 title="Model",
                 loc="upper right",
                 frameon=False)

ax.add_artist(leg1)

split_handles = [
    Line2D([0], [0], color='black', linestyle=ls, lw=2, label=split)
    for split, ls in split_styles.items()
]


ax.legend(handles=split_handles,
          title="Split",
          loc="lower left",
          frameon=False)

plt.tight_layout()
plt.savefig("thesis_plots/convnext_v2_dispersion.pdf")
plt.close()


params = [3388604, 4849684, 8555204, 14985844, 27871588, 49561444, 87708804, 196443844]
i=0
fig, ax = plt.subplots(figsize=(3.2, 3.2))
for m, color in zip(models, colors):

    path = (
        folder
        + f'{m}_lr-0.0005_wd-0.05_bs-128_epochs-200_'
        + 'cosine_warmup-1250.0_clipgrad-True_'
        + 'pe-encoder-straight_pe-4_rmse_metrics.zarr'
    )

    try:
        root = zarr.open(path, mode='r')
        # train_r2 = root['R2_train'][:]
        val_r2 = np.max(root['R2_val'][:])
    except Exception as e:
        print(f"Skipping {m}: {e}")
        continue

    ax.plot(params[i],1 - val_r2,
            '.',
            # label=m.replace("ConvNeXt-V2-", ""),
            )

    i+=1

ax.set_xlabel("Parameter Count")
ax.set_ylabel(r"$1 - R^2$")
ax.set_yscale("log")
ax.set_xscale("log")
ax.grid(alpha=0.3)
ax.grid(which='minor', alpha=0.15)  # add minor grid lines
ax.minorticks_on()
# plt.legend(title="Model", frameon=True, loc='right')
model_handles = [
    Line2D([0], [0], color=c, lw=2, label=m.replace("ConvNeXt-V2-", ""))
    for m, c in zip(models, colors)
]
leg1 = ax.legend(
    handles=model_handles,
    title="Model",
    loc="upper left",
    bbox_to_anchor=(1, 1),   # places legend to the right of the axes
    frameon=False
)
plt.title(r"ConvNeXt-V2 Max $R^2$ vs. Parameter Count")  # add title for clarity
ax.add_artist(leg1)
plt.tight_layout()
plt.savefig("thesis_plots/convnext_v2_dispersion_params.pdf", bbox_inches='tight')
plt.close()