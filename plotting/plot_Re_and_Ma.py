import numpy as np
import matplotlib.pyplot as plt
import zarr
from ploting import figsize
from matplotlib.ticker import ScalarFormatter

root = zarr.open('data/train.zarr',mode='r')
lbm_results = root['lbm_results']

# histogram of Re and Ma
Re_values = []
Ma_values = []
for i in range(16000):
    Re_x = lbm_results['Re_phys_x'][i]
    Re_y = lbm_results['Re_phys_y'][i]
    Ma_x = lbm_results['Ma_x'][i]
    Ma_y = lbm_results['Ma_y'][i]
    Re_values.append(Re_x)
    Re_values.append(Re_y)
    Ma_values.append(Ma_x)
    Ma_values.append(Ma_y)

fig, ax = plt.subplots(1,2,figsize=(figsize[0],figsize[1]*0.8),sharey=True)
ax[0].hist(Re_values, bins=100, color='C9')
ax[0].set_xlabel('Reynolds Number')
ax[0].set_ylabel('Count')

ax[1].hist(Ma_values, bins=100, color='C6')
ax[1].set_xlabel('Mach Number')

ax[1].set_xticks([0, 0.00005, 0.0001, 0.00015])

formatter = ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((0, 0))  # force scientific for all magnitudes
ax[1].xaxis.set_major_formatter(formatter)

plt.tight_layout()
plt.savefig('Re_and_Ma_histogram.pdf', bbox_inches='tight',dpi=300)