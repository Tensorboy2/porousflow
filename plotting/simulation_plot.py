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
porosities = data['metrics']['metrics']['porosity']
iteration_x = data['lbm_results']['iteration_x']
iteration_y = data['lbm_results']['iteration_y']

fig,ax = plt.subplots(1,2,figsize=(6.4,3.2))
ax[0].hist2d(porosities,iteration_x,bins=100,cmin=1)
ax[0].set_ylabel('Iterations-x')
ax[0].set_xlabel('Porosity')

ax[1].hist2d(porosities,iteration_y,bins=100,cmin=1)
ax[1].set_ylabel('Iterations-y')
ax[1].set_xlabel('Porosity')
plt.tight_layout()
plt.savefig('thesis_plots/iterations.pdf', bbox_inches='tight')
plt.show()