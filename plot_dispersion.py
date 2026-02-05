'''
Docstring for plot_dispersion.py
Script for showing the distribution of the dispersion dataset.
'''
import os 
import numpy as np
import matplotlib.pyplot as plt
import zarr

data_path = 'data/train.zarr'
data = zarr.open(data_path, mode='r')
Dx = data['dispersion_results']['Dx']
Dy = data['dispersion_results']['Dy']
porosities = data['metrics']['metrics']['porosity']


Pe_values = [0.1, 10, 50, 100, 500]
pe_indices = [0, 1, 2, 3, 4]   # K
N = porosities.shape[0]

plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
})

fig, axes = plt.subplots(len(pe_indices), 4, figsize=(7.2, 7.2))
# fig.suptitle('Dx components vs Porosity', fontsize=16, fontweight='bold')

y_labels = {0:r'$D_{xx}$', 1:r'$D_{xy}$', 2:r'$D_{yx}$', 3:r'$D_{yy}$'}
labels = {0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)}

for i in range(len(pe_indices)):
    for j in range(4):
        ax = axes[i, j]

        # Merge all k into one vector
        # Dx_merged = Dx[:, 1, i, j]
        # por_merged = np.repeat(porosities, K)

        ax.hist2d(
            porosities,
            Dx[:, i, labels[j][0], labels[j][1]],
            bins=100,
            cmap='viridis',
            cmin=1
        )

        ax.set_title(f'Pe: {Pe_values[i]}')
        ax.set_xlabel('Porosity')
        # ax.set_yscale('log')
        ax.set_ylabel(y_labels[j])

plt.tight_layout()
plt.savefig('thesis_plots/Dx_vs_porosity.png', bbox_inches='tight')

fig, axes = plt.subplots(len(pe_indices), 4, figsize=(7.2, 7.2))
# fig.suptitle('Dx components vs Porosity', fontsize=16, fontweight='bold')

y_labels = {0:r'$D_{xx}$', 1:r'$D_{xy}$', 2:r'$D_{yx}$', 3:r'$D_{yy}$'}
labels = {0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)}

for i in range(len(pe_indices)):
    for j in range(4):
        ax = axes[i, j]

        # Merge all k into one vector
        # Dx_merged = Dx[:, 1, i, j]
        # por_merged = np.repeat(porosities, K)

        ax.hist2d(
            porosities,
            Dy[:, i, labels[j][0], labels[j][1]],
            bins=100,
            cmap='viridis',
            cmin=1
        )

        ax.set_title(f'Pe: {Pe_values[i]}')
        ax.set_xlabel('Porosity')
        # ax.set_yscale('log')
        ax.set_ylabel(y_labels[j])

plt.tight_layout()
plt.savefig('thesis_plots/Dy_vs_porosity.png', bbox_inches='tight')

fig, axes = plt.subplots(len(pe_indices), 4, figsize=(7.2, 7.2))
# fig.suptitle('Dx components vs Porosity', fontsize=16, fontweight='bold')

y_labels = {0:r'$D_{xx}$', 1:r'$D_{xy}$', 2:r'$D_{yx}$', 3:r'$D_{yy}$'}
labels = {0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)}

for i in range(len(pe_indices)):
    for j in range(4):
        ax = axes[i, j]

        # Merge all k into one vector
        # Dx_merged = Dx[:, 1, i, j]
        # por_merged = np.repeat(porosities, K)

        ax.hist2d(
            porosities,
            np.asinh(np.asinh(Dx[:, i, labels[j][0], labels[j][1]])),
            bins=100,
            cmap='viridis',
            cmin=1
        )

        ax.set_title(f'Pe: {Pe_values[i]}')
        ax.set_xlabel('Porosity')
        # ax.set_yscale('log')
        ax.set_ylabel(y_labels[j])

plt.tight_layout()
plt.savefig('thesis_plots/Dx_vs_porosity_asinh.png', bbox_inches='tight')

fig, axes = plt.subplots(len(pe_indices), 4, figsize=(7.2, 7.2))
# fig.suptitle('Dx components vs Porosity', fontsize=16, fontweight='bold')

y_labels = {0:r'$D_{xx}$', 1:r'$D_{xy}$', 2:r'$D_{yx}$', 3:r'$D_{yy}$'}
labels = {0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)}

for i in range(len(pe_indices)):
    for j in range(4):
        ax = axes[i, j]

        # Merge all k into one vector
        # Dx_merged = Dx[:, 1, i, j]
        # por_merged = np.repeat(porosities, K)

        ax.hist2d(
            porosities,
            np.asinh(Dy[:, i, labels[j][0], labels[j][1]]),
            bins=100,
            cmap='viridis',
            cmin=1
        )

        ax.set_title(f'Pe: {Pe_values[i]}')
        ax.set_xlabel('Porosity')
        # ax.set_yscale('log')
        ax.set_ylabel(y_labels[j])

plt.tight_layout()
plt.savefig('thesis_plots/Dy_vs_porosity_asinh.png', bbox_inches='tight')
