'''
Docstring for plot_dispersion.py
Script for showing the distribution of the dispersion dataset.
'''
import os 
import numpy as np
import matplotlib.pyplot as plt
import zarr
from ploting import figsize

data_path = 'data/train.zarr'
data = zarr.open(data_path, mode='r')
porosities = data['metrics']['metrics']['porosity']
iteration_x = data['lbm_results']['iteration_x']
iteration_y = data['lbm_results']['iteration_y']

fig,ax = plt.subplots(1,2,figsize=figsize)

ax[0].hist(iteration_x,bins=100)
ax[0].set_ylabel('Count')
ax[0].set_xlabel('Iteration-x')

ax[1].hist(iteration_y,bins=100)
ax[1].set_ylabel('Count')
ax[1].set_xlabel('Iteration-y')
plt.tight_layout()
plt.savefig('thesis_plots/iterations.pdf', bbox_inches='tight')
plt.show()