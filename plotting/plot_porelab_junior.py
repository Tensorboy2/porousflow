'''
plotting png for porelab junior forum presentation
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import zarr

''' Example image'''
data = zarr.open('data/train.zarr', mode='r')
images = data['filled_images']['filled_images'][0]
fig = plt.figure(figsize=(25, 14))
plt.imshow(images,cmap='gray')
plt.axis('off')
plt.savefig('porelab_junior_plots/example_media_porelab_junior_image.png', bbox_inches='tight', dpi=300)

'''Percolation plot'''
from scipy.ndimage import  binary_dilation, label

STRUCTURE_8 = np.ones((3, 3), dtype=int) # 8-connectivity structure

def detect_percolation(binary_img):
    """Check if fluid clusters percolate in x or y (periodic)."""

    dilated = binary_dilation(binary_img,iterations=2)
    labeled, _ = label(~dilated,STRUCTURE_8)
    h, w = labeled.shape

    # Boundary check:
    x_perc = any(labeled[0, j] != 0 and labeled[0, j] == labeled[-1, j] for j in range(w))
    y_perc = any(labeled[i, 0] != 0 and labeled[i, 0] == labeled[i, -1] for i in range(h))

    return x_perc, y_perc, labeled

x,y,labeled = detect_percolation(images)
fig = plt.figure(figsize=(25, 14))
plt.imshow(labeled,cmap='nipy_spectral')
plt.axis('off')
plt.savefig('porelab_junior_plots/example_media_porelab_junior_percolation.png', bbox_inches='tight', dpi=300)

'''Filled image and diff'''
filled_images = data['filled_images']['filled_images'][0]
diff = filled_images.astype(int) - images.astype(int)
fig,ax = plt.subplots(1,3,figsize=(25, 14))
ax[0].imshow(images,cmap='gray')
ax[0].axis('off')
ax[1].imshow(filled_images,cmap='gray')
ax[1].axis('off')
ax[2].imshow(diff,cmap='gray')
ax[2].axis('off')
plt.savefig('porelab_junior_plots/example_media_porelab_junior_filled_diff.png', bbox_inches='tight', dpi=300)