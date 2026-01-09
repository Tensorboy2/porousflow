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
Dx = np.arcsinh(data['dispersion_results']['Dx'])
Dy = np.arcsinh(data['dispersion_results']['Dy'])

print(f'Dx shape: {Dx.shape}')
print(f'Dy shape: {Dy.shape}')

Pe_values = [0.1, 10, 50, 100, 500]

# Create a separate figure for each Pe value
for k, Pe in enumerate(Pe_values):
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    fig.suptitle(f'Dispersion Coefficients Distribution at Pe={Pe}', fontsize=16, fontweight='bold')
    
    for i in range(2):
        for j in range(2):
            ax = axes[i, j]
            
            # Extract data for this Pe value and grid position
            dx_data = Dx[:, k, i, j]
            dy_data = Dy[:, k, i, j]
            
            # Create histograms
            ax.hist(dx_data, bins=30, alpha=0.6, label='Dx', color='blue', edgecolor='black')
            ax.hist(dy_data, bins=30, alpha=0.6, label='Dy', color='orange', edgecolor='black')
            
            # Add statistics to the plot
            # ax.axvline(np.mean(dx_data), color='blue', linestyle='--', linewidth=2, label=f'Dx mean: {np.mean(dx_data):.3f}')
            # ax.axvline(np.mean(dy_data), color='orange', linestyle='--', linewidth=2, label=f'Dy mean: {np.mean(dy_data):.3f}')
            
            # ax.set_title(f'Grid Position ({i}, {j})', fontsize=12)
            ax.set_xlabel('Dispersion Coefficient', fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Optional: Save figure
    plt.savefig(f'dispersion_Pe_{Pe}.pdf', dpi=300, bbox_inches='tight')

# plt.show()

# Optional: Create an overview plot showing all Pe values for one grid position
fig, axes = plt.subplots(len(Pe_values), 2, figsize=(6, 8))
# fig.suptitle('Dispersion Coefficients for All Pe Values (Grid Position 0,0)', fontsize=16, fontweight='bold')

for k, Pe in enumerate(Pe_values):
    # Dx distribution
    axes[k, 0].hist(Dx[:, k, 0, 0], bins=30, alpha=0.7, color='blue', edgecolor='black')
    axes[k, 0].set_title(f'Dx at Pe={Pe}')
    axes[k, 0].set_xlabel('Dx')
    axes[k, 0].set_ylabel('Frequency')
    axes[k, 0].axvline(np.mean(Dx[:, k, 0, 0]), color='red', linestyle='--', linewidth=2)
    axes[k, 0].grid(True, alpha=0.3)
    
    # Dy distribution
    axes[k, 1].hist(Dy[:, k, 1, 1], bins=30, alpha=0.7, color='orange', edgecolor='black')
    axes[k, 1].set_title(f'Dy at Pe={Pe}')
    axes[k, 1].set_xlabel('Dy')
    axes[k, 1].set_ylabel('Frequency')
    axes[k, 1].axvline(np.mean(Dy[:, k, 1, 1]), color='red', linestyle='--', linewidth=2)
    axes[k, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('diagonal_components.pdf', dpi=300, bbox_inches='tight')
plt.show()


print("\nSummary Statistics:")
print("=" * 60)
for k, Pe in enumerate(Pe_values):
    print(f"\nPe = {Pe}:")
    print(f"  Dx: mean={np.mean(Dx[:, k, :, :]):.4f}, std={np.std(Dx[:, k, :, :]):.4f}")
    print(f"  Dy: mean={np.mean(Dy[:, k, :, :]):.4f}, std={np.std(Dy[:, k, :, :]):.4f}")