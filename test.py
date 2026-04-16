import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Generate data
np.random.seed(0)
n = 20000
x = np.random.rand(n)
y = x**2 + np.random.normal(0, 0.05, n)

x2 = np.random.rand(n)
y2 = x2**2 + np.random.normal(0, 0.05, n)

# KDE densities
xy = np.vstack([x, y])
z = gaussian_kde(xy)(xy)

xy2 = np.vstack([x2, y2])
z2 = gaussian_kde(xy2)(xy2)

# Normalize densities
z = (z - z.min()) / (z.max() - z.min())
z2 = (z2 - z2.min()) / (z2.max() - z2.min())

# Plot
plt.figure()

plt.scatter(x, y, c=z, cmap='Blues', alpha=0.5, s=20)
plt.scatter(x2, y2, c=z2, cmap='Oranges', alpha=0.5, s=20, edgecolors='black', linewidths=0.2)

plt.title("Overlapping Distributions with KDE Coloring")
plt.xlabel("x")
plt.ylabel("y")

plt.show()