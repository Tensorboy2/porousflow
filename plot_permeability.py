import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import zarr

data = zarr.open('data/train.zarr', mode='r')
permeabilities = data['lbm_results']['K']
porosities = data['metrics']['metrics']['porosity']

bins = 100

# Extract each component separately
K_xx = permeabilities[:, 0, 0]
K_xy = permeabilities[:, 0, 1]
K_yx = permeabilities[:, 1, 0]
K_yy = permeabilities[:, 1, 1]

panels = [
    (porosities, K_xx, r'$K_{xx}$'),
    (porosities, K_xy, r'$K_{xy}$'),
    (porosities, K_yx, r'$K_{yx}$'),
    (porosities, K_yy, r'$K_{yy}$'),
]

panel_labels = ['(a)', '(b)', '(c)', '(d)']

# Precompute histograms for shared color normalization
H_all = []
for x, y, _ in panels:
    H, _, _ = np.histogram2d(x, y, bins=bins)
    H_all.append(H)

vmax = max(H.max() for H in H_all)
norm = plt.Normalize(vmin=1, vmax=vmax)

plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
})


fig = plt.figure(figsize=(7.2, 4.4))
outer = GridSpec(2, 3, width_ratios=[1, 1, 0.05], hspace=0.35, wspace=0.35)

last_h = None

for idx, (x, y, title) in enumerate(panels):
    row = idx // 2
    col = idx % 2

    inner = outer[row, col].subgridspec(
        2, 2,
        width_ratios=(4, 1),
        height_ratios=(1, 4),
        wspace=0.001,
        hspace=0.001
    )

    ax_main = fig.add_subplot(inner[1, 0])
    ax_x = fig.add_subplot(inner[0, 0], sharex=ax_main)
    ax_y = fig.add_subplot(inner[1, 1], sharey=ax_main)

    last_h = ax_main.hist2d(x, y, bins=bins, cmap='viridis', cmin=1, norm=norm)

    ax_x.hist(x, bins=bins, color='black')
    ax_y.hist(y, bins=bins, orientation='horizontal', color='black')

    ax_x.tick_params(labelbottom=False)
    ax_y.tick_params(labelleft=False)

    ax_main.set_xlabel('Porosity')
    ax_main.set_ylabel(title + r' ($10^{-10}$ $m^2$)')

    ax_x.spines['bottom'].set_visible(False)
    ax_y.spines['left'].set_visible(False)

    ax_main.text(
        0.05, 0.95, panel_labels[idx],
        transform=ax_main.transAxes,
        fontsize=14,
        fontweight='bold',
        va='top',
        ha='left'
    )

# Shared colorbar
cax = fig.add_subplot(outer[:, 2])
cbar = fig.colorbar(last_h[3], cax=cax)
cbar.set_label('Counts')
# plt.tight_layout(pad=0.3)

plt.savefig('permeability_vs_porosity_unmerged.pdf', bbox_inches='tight')
plt.show()
