import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LogNorm
import zarr


data = zarr.open('data/train.zarr', mode='r')
permeabilities = data['lbm_results']['K']
porosities = data['metrics']['metrics']['porosity']

# Merege K
K_diag = np.concatenate([
    permeabilities[:, 0, 0],
    permeabilities[:, 1, 1],
])

K_off = np.concatenate([
    permeabilities[:, 0, 1],
    permeabilities[:, 1, 0],
])

por_diag = np.concatenate([porosities, porosities])
por_off  = np.concatenate([porosities, porosities])
bins = 100

H_diag, _, _ = np.histogram2d(por_diag, K_diag, bins=bins)
H_off,  _, _ = np.histogram2d(por_off,  K_off,  bins=bins)

vmax = max(H_diag.max(), H_off.max())
norm = plt.Normalize(vmin=1, vmax=vmax)
# norm = LogNorm(vmin=1, vmax=vmax)
# Matplotlib parameters for fonts and sizes
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16
})

fig = plt.figure(figsize=(8, 4))
outer = GridSpec(1, 3, width_ratios=[1,1,0.06], figure=fig, wspace=0.5)

panels = [
    (por_diag, K_diag, r'$K_{xx}, K_{yy}$'),
    (por_off,  K_off,  r'$K_{xy}, K_{yx}$'),
]

panel_labels = ['(a)', '(b)']

for idx, (x, y, title) in enumerate(panels):
    inner = outer[idx].subgridspec(
        2, 2,
        width_ratios=(4, 1),
        height_ratios=(1, 4),
        wspace=0.001,
        hspace=0.001
    )

    ax_main = fig.add_subplot(inner[1, 0])
    ax_x = fig.add_subplot(inner[0, 0], sharex=ax_main)
    ax_y = fig.add_subplot(inner[1, 1], sharey=ax_main)

    h = ax_main.hist2d(x, y, bins=bins, cmap='viridis', cmin=1, norm=norm)
    ax_main.yaxis.get_offset_text().set_visible(False)

    ax_x.hist(x, bins=bins, color='black')
    ax_y.hist(y, bins=bins, orientation='horizontal', color='black')

    ax_x.tick_params(labelbottom=False)
    ax_y.tick_params(labelleft=False)

    ax_main.set_xlabel('Porosity')
    ax_main.set_ylabel(f'{title}'+r' ($10^{-10}$ $m^2$)')

    # Top histogram (x-marginal)
    ax_x.spines['bottom'].set_visible(False)
    ax_x.tick_params(bottom=False)

    # Right histogram (y-marginal)
    ax_y.spines['left'].set_visible(False)
    ax_y.tick_params(left=False)


    ax_main.text(
        0.15, 0.90, panel_labels[idx],
        transform=ax_main.transAxes,
        fontsize=14,
        fontweight='bold',
        va='top',
        ha='left'
    )

    # if idx == 1:
    #     ax_main.set_yticks([0, ])
    #     ax_main.set_yticklabels(['0', '1', '2', ''])



        

cax = fig.add_subplot(outer[2])
cbar = fig.colorbar(h[3], cax=cax)
cbar.set_label('Counts')


plt.savefig('permeability_vs_porosity.pdf', bbox_inches='tight')
