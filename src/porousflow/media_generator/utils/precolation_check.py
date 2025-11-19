'''
precolation_check.py

Module for checking percolation in porous media structures.
'''
import numpy as np
from scipy.ndimage import  binary_dilation, label

STRUCTURE_8 = np.ones((3, 3), dtype=int) # 8-connectivity structure

def detect_percolation(binary_img):
    """Check if fluid clusters percolate in x or y (periodic)."""

    dilated = binary_dilation(binary_img)
    labeled, _ = label(~dilated,STRUCTURE_8)
    h, w = labeled.shape

    # Boundary check:
    x_perc = any(labeled[0, j] != 0 and labeled[0, j] == labeled[-1, j] for j in range(w))
    y_perc = any(labeled[i, 0] != 0 and labeled[i, 0] == labeled[i, -1] for i in range(h))

    return x_perc, y_perc, labeled


def fill_non_percolating_fluid(binary_img):
    """
    # Not in use!
    Turns non-percolating fluid regions into solid.
    """
    labels, _ = label(~binary_img)
    top, bottom = set(labels[0, :]) - {0}, set(labels[-1, :]) - {0}
    left, right = set(labels[:, 0]) - {0}, set(labels[:, -1]) - {0}

    percolating = (top & bottom) | (left & right)
    non_perc_mask = (labels > 0) & ~np.isin(labels, list(percolating))

    filled_img = binary_img.copy()
    filled_img[non_perc_mask] = True  # fill in fluid to solid
    return filled_img

def fill_non_percolating_fluid_periodic(binary_img):
    """Turns non-percolating fluid regions into solid with periodic BCs."""
    h, w = binary_img.shape

    tiled = np.tile(binary_img, (3, 3))
    labels, _ = label(tiled,STRUCTURE_8)

    center_labels = labels[h:2*h, w:2*w]

    percolating_labels = set()

    for label_id in np.unique(center_labels):
        if label_id == 0:
            continue
        mask = labels == label_id

        left_edge = mask[h:2*h, w] | mask[h:2*h, 2*w-1]
        top_edge  = mask[h, w:2*w] | mask[2*h-1, w:2*w]

        if left_edge.any() or top_edge.any():
            percolating_labels.add(label_id)

    filled_img = binary_img.copy()
    non_perc_mask = (center_labels > 0) & ~np.isin(center_labels, list(percolating_labels))
    filled_img[non_perc_mask] = True  # Fill holes to solid

    return filled_img

if __name__ == "__main__":
    # Example usage
    from binary_blobs import periodic_binary_blobs
    import matplotlib.pyplot as plt

    test_img = periodic_binary_blobs(n_dim=2, length=128, volume_fraction=0.4, blob_size_fraction=0.1, 
                                     seed=0)
    plt.subplot(1, 2, 1)
    plt.imshow(test_img, cmap='gray')
    plt.title("Original Image")

    x_perc, y_perc, labels = detect_percolation(test_img)
    print("Percolation in x:", x_perc)
    print("Percolation in y:", y_perc)
    # print("Labeled clusters:\n", labels)

    filled_img = fill_non_percolating_fluid_periodic(test_img)
    # print("Filled image:\n", filled_img)
    plt.subplot(1, 2, 2)
    plt.imshow(filled_img, cmap='gray')
    plt.title("Filled Non-Percolating Regions")

    plt.show()