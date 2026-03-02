import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import zarr

bins = 100
paths = ['train', 'validation', 'test']

# -------------------------------------------------
# Merge datasets
# -------------------------------------------------
all_porosity = []
all_K = []

for path in paths:
    data = zarr.open(f'data/{path}.zarr', mode='r')
    K = data['lbm_results']['K'][:]                 # (N,2,2)
    phi = data['metrics']['metrics']['porosity'][:] # (N,)
    
    all_porosity.append(phi)
    all_K.append(K)

porosity = np.concatenate(all_porosity, axis=0)
K = np.concatenate(all_K, axis=0)

# Components
K_xx = K[:, 0, 0]
K_xy = K[:, 0, 1]
K_yx = K[:, 1, 0]
K_yy = K[:, 1, 1]

# Example grouping: diagonal vs off-diagonal
panels = [
    (np.concatenate([porosity, porosity]),
     np.concatenate([K_xx, K_yy]),
     r'$K_{xx},K_{yy}$'),

    (np.concatenate([porosity, porosity]),
     np.concatenate([K_xy, K_yx]),
     r'$K_{xy},K_{yx}$'),
]

# -------------------------------------------------
# Precompute global normalization (density-based)
# -------------------------------------------------
H_all = []
for x, y, _ in panels:
    H, _, _ = np.histogram2d(x, y, bins=bins, density=False)
    H_all.append(H)

vmax = max(H.max() for H in H_all)
norm = plt.Normalize(vmin=1, vmax=vmax)

# -------------------------------------------------
# Figure layout
# -------------------------------------------------
from ploting import figsize
plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
})
fig = plt.figure(figsize=figsize)
first = GridSpec(1, 2, width_ratios=[20, 1], wspace=0.075)
outer = first[0].subgridspec(1,2,width_ratios=[10,10],wspace=0.185)

cax = fig.add_subplot(first[1])

for i, (x, y, title) in enumerate(panels):
    
    base = outer[i].subgridspec(
        2, 2,
        width_ratios=(4, 1),
        height_ratios=(1, 4),
        wspace=0.001,
        hspace=0.001
    )

    ax_main = fig.add_subplot(base[1, 0])
    ax_x = fig.add_subplot(base[0, 0], sharex=ax_main)
    ax_y = fig.add_subplot(base[1, 1], sharey=ax_main)

    # 2D histogram with normalized density
    h = ax_main.hist2d(
        x, y,
        bins=bins,
        cmap='viridis',
        density=False,
        norm=norm,
        cmin=1,
    )

    # Marginals
    ax_x.hist(x, bins=bins, color='black', density=False)
    ax_y.hist(y, bins=bins, orientation='horizontal',
              color='black', density=False)

    ax_x.tick_params(labelbottom=False)
    ax_y.tick_params(labelleft=False)

    ax_main.set_xlabel('Porosity')
    # ax_main.set_title(title)
    
    ax_main.annotate(title,(0.1,0.9),xycoords='axes fraction')
    # Force scientific notation for the y-axis
    # ax_main.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    # To move the "10^a" text (it often overlaps titles)
    # ax_main.yaxis.get_offset_text().set_fontsize(8)
    # ax_main.yaxis.get_offset_text().set_position((-0.1, 1.05))
    ax_main.yaxis.get_major_formatter().set_useOffset(False)
    ax_main.yaxis.get_major_formatter().set_scientific(True)
  
    # ax_main.set_ylabel('Permeability ($10^{-11}$ $m^2$)')    # Horizontal alignment: 'left', 'right', 'center'
    if i==0:
        ax_main.set_ylabel(r'Permeability ($10^{-10}$ $m^2$)')
    if i==1:
        ax_main.set_ylim(-2.9e-10,2.9e-10)
        # plt.draw() # Force the renderer to create the labels first

        # labels = ax_main.get_yticklabels()
        # print(labels)
        # ax_main.set_yticklabels(labels[:-2])


# Shared colorbar
cb = fig.colorbar(h[3], cax=cax)
cb.set_label('Bin count')
# plt.tight_layout()
plt.savefig('thesis_plots/permeability_dist.pdf', bbox_inches='tight')
# plt.show()

# last_h = None

# for idx, (x, y, title) in enumerate(panels):
#     row = idx // 2
#     col = idx % 2

#     inner = outer[row, col].subgridspec(
#         2, 2,
#         width_ratios=(4, 1),
#         height_ratios=(1, 4),
#         wspace=0.001,
#         hspace=0.001
#     )

#     ax_main = fig.add_subplot(inner[1, 0])
#     ax_x = fig.add_subplot(inner[0, 0], sharex=ax_main)
#     ax_y = fig.add_subplot(inner[1, 1], sharey=ax_main)

#     last_h = ax_main.hist2d(x, y, bins=bins, cmap='viridis', cmin=1, norm=norm)

#     ax_x.hist(x, bins=bins, color='black')
#     ax_y.hist(y, bins=bins, orientation='horizontal', color='black')

#     ax_x.tick_params(labelbottom=False)
#     ax_y.tick_params(labelleft=False)

#     ax_main.set_xlabel('Porosity')
#     ax_main.set_ylabel(title + r' ($10^{-10}$ $m^2$)')

#     ax_x.spines['bottom'].set_visible(False)
#     ax_y.spines['left'].set_visible(False)

#     ax_main.text(
#         0.05, 0.95, panel_labels[idx],
#         transform=ax_main.transAxes,
#         fontsize=14,
#         fontweight='bold',
#         va='top',
#         ha='left'
#     )

# # Shared colorbar
# cax = fig.add_subplot(outer[:, 2])
# cbar = fig.colorbar(last_h[3], cax=cax)
# cbar.set_label('Counts')
# # plt.tight_layout(pad=0.3)

# plt.savefig('thesis_plots/permeability_dist.pdf', bbox_inches='tight')
# plt.show()
