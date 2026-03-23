'''
Docstring for plot_dispersion.py
Script for showing the distribution of the dispersion dataset.
'''
import os 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import zarr
from ploting import figsize
from plottools import shared_cbar, NormedCmap

Pe_values = np.array([0.1, 10, 50, 100, 500])

plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
})

# fig, ax = shared_cbar(2,figsize=figsize,cbar_width=1)
fig = plt.figure(figsize=(figsize[0], figsize[1] * 0.7))
outer = GridSpec(1, 2, width_ratios=[20, 0.4], wspace=0.05)
inner = outer[0].subgridspec(1, 2, wspace=0.225)
axs = [fig.add_subplot(inner[i]) for i in range(2)]
cax = fig.add_subplot(outer[1])
unique_pe = np.array([0.1, 10, 50, 100, 500])
n = len(unique_pe)
# cmap = NormedCmap('hsv', [0.1, 500], norm='log', cutoff=(0.07,1))
cmap = matplotlib.colormaps.get_cmap('viridis')
norm = mcolors.BoundaryNorm(np.arange(-0.5, n), cmap.N)
sm = cm.ScalarMappable(norm=norm, cmap=cmap)
paths = ['train', 'validation', 'test']
num_samples = -1
pe_to_idx = {pe: i for i, pe in enumerate(unique_pe)}
for path in paths:

    data_path = f'data/{path}.zarr'
    data = zarr.open(data_path, mode='r')

    Dx = data['dispersion_results']['Dx'][:num_samples]
    Dy = data['dispersion_results']['Dy'][:num_samples]
    porosities = data['metrics']['metrics']['porosity'][:num_samples]
    N = porosities.shape[0]

    for i in range(2):
        for j, Pe in enumerate(Pe_values[::-1]):  # reverse cleanly

            # color = cmap(norm(Pe))
            color = cmap(norm(pe_to_idx[Pe]))
            axs[i].plot(
                porosities,
                Dx[:, -(j+1), i, i],
                # np.arcsinh(Dx[:, -(j+1), i, i]),
                linestyle='',
                marker='o',
                markerfacecolor='none',
                markeredgecolor=color,
                markersize=1.0,
                markeredgewidth=0.5,
                alpha=0.3,
            )

            axs[i].plot(
                porosities,
                # np.arcsinh(Dy[:, -(j+1), (i+1)%2, (i+1)%2]),
                Dy[:, -(j+1), (i+1)%2, (i+1)%2],
                linestyle='',
                marker='o',
                markerfacecolor='none',
                markeredgecolor=color,
                markersize=1.0,
                markeredgewidth=0.5,
                alpha=0.3,
            )

for i in range(2):
    axs[i].set_xlabel('Porosity')
    axs[i].set_yscale('log')

axs[0].set_ylabel(r'$D/D_m$')
axs[0].annotate(r'$D_\parallel$',(0.85,0.9),xycoords='axes fraction')
axs[1].annotate(r'$D_\bot$',(0.85,0.9),xycoords='axes fraction')
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cax)
cbar.set_label('Péclet number (Pe)')
cbar.set_ticks(np.arange(n))
cbar.set_ticklabels([str(v) for v in unique_pe])
# cbar = cmap.lined_colorbar(Pe_values, cax=ax[-1], label='Péclet number (Pe)')
# cbar.set_ticks(Pe_values)
# cbar.set_ticklabels([str(p) for p in Pe_values])
plt.savefig('thesis_plots/dispersion_dist_2.png',dpi=300, bbox_inches='tight')
plt.close()


# fig, axes = plt.subplots(len(pe_indices), 4, figsize=(7.2, 7.2))
# # fig.suptitle('Dx components vs Porosity', fontsize=16, fontweight='bold')

# y_labels = {0:r'$D_{xx}$', 1:r'$D_{xy}$', 2:r'$D_{yx}$', 3:r'$D_{yy}$'}
# labels = {0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)}

# for i in range(len(pe_indices)):
#     for j in range(4):
#         ax = axes[i, j]

#         # Merge all k into one vector
#         # Dx_merged = Dx[:, 1, i, j]
#         # por_merged = np.repeat(porosities, K)

#         ax.hist2d(
#             porosities,
#             Dx[:, i, labels[j][0], labels[j][1]],
#             bins=100,
#             cmap='viridis',
#             cmin=1
#         )

#         ax.set_title(f'Pe: {Pe_values[i]}')
#         ax.set_xlabel('Porosity')
#         # ax.set_yscale('log')
#         ax.set_ylabel(y_labels[j])

# plt.tight_layout()
# plt.savefig('thesis_plots/Dx_vs_porosity.png', bbox_inches='tight')

# fig, axes = plt.subplots(len(pe_indices), 4, figsize=(7.2, 7.2))
# # fig.suptitle('Dx components vs Porosity', fontsize=16, fontweight='bold')

# y_labels = {0:r'$D_{xx}$', 1:r'$D_{xy}$', 2:r'$D_{yx}$', 3:r'$D_{yy}$'}
# labels = {0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)}

# for i in range(len(pe_indices)):
#     for j in range(4):
#         ax = axes[i, j]

#         # Merge all k into one vector
#         # Dx_merged = Dx[:, 1, i, j]
#         # por_merged = np.repeat(porosities, K)

#         ax.hist2d(
#             porosities,
#             Dy[:, i, labels[j][0], labels[j][1]],
#             bins=100,
#             cmap='viridis',
#             cmin=1
#         )

#         ax.set_title(f'Pe: {Pe_values[i]}')
#         ax.set_xlabel('Porosity')
#         # ax.set_yscale('log')
#         ax.set_ylabel(y_labels[j])

# plt.tight_layout()
# plt.savefig('thesis_plots/Dy_vs_porosity.png', bbox_inches='tight')

# fig, axes = plt.subplots(len(pe_indices), 4, figsize=(7.2, 7.2))
# # fig.suptitle('Dx components vs Porosity', fontsize=16, fontweight='bold')

# y_labels = {0:r'$D_{xx}$', 1:r'$D_{xy}$', 2:r'$D_{yx}$', 3:r'$D_{yy}$'}
# labels = {0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)}

# for i in range(len(pe_indices)):
#     for j in range(4):
#         ax = axes[i, j]

#         # Merge all k into one vector
#         # Dx_merged = Dx[:, 1, i, j]
#         # por_merged = np.repeat(porosities, K)

#         ax.hist2d(
#             porosities,
#             np.asinh(np.asinh(Dx[:, i, labels[j][0], labels[j][1]])),
#             bins=100,
#             cmap='viridis',
#             cmin=1
#         )

#         ax.set_title(f'Pe: {Pe_values[i]}')
#         ax.set_xlabel('Porosity')
#         # ax.set_yscale('log')
#         ax.set_ylabel(y_labels[j])

# plt.tight_layout()
# plt.savefig('thesis_plots/Dx_vs_porosity_asinh.png', bbox_inches='tight')

# fig, axes = plt.subplots(len(pe_indices), 4, figsize=(7.2, 7.2))
# # fig.suptitle('Dx components vs Porosity', fontsize=16, fontweight='bold')

# y_labels = {0:r'$D_{xx}$', 1:r'$D_{xy}$', 2:r'$D_{yx}$', 3:r'$D_{yy}$'}
# labels = {0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)}

# for i in range(len(pe_indices)):
#     for j in range(4):
#         ax = axes[i, j]

#         # Merge all k into one vector
#         # Dx_merged = Dx[:, 1, i, j]
#         # por_merged = np.repeat(porosities, K)

#         ax.hist2d(
#             porosities,
#             np.asinh(Dy[:, i, labels[j][0], labels[j][1]]),
#             bins=100,
#             cmap='viridis',
#             cmin=1
#         )

#         ax.set_title(f'Pe: {Pe_values[i]}')
#         ax.set_xlabel('Porosity')
#         # ax.set_yscale('log')
#         ax.set_ylabel(y_labels[j])

# plt.tight_layout()
# plt.savefig('thesis_plots/Dy_vs_porosity_asinh.png', bbox_inches='tight')


