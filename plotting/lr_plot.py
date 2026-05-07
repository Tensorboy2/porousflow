import numpy as np
import matplotlib.pyplot as plt
from ploting import figsize
plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
})
x = np.linspace(0,1,1000)
y = np.zeros_like(x)
# plot linear warmup and cosine decay
for i in range(len(x)):
    if x[i] < 0.1:
        y[i] = x[i]/0.1
    else:
        y[i] = 0.5 * (1 + np.cos(np.pi * (x[i]-0.1) / 0.9))


fig, ax = plt.subplots(1,1,figsize=(figsize[0],figsize[1]*0.6))
ax.plot(x,y,label='LR Multiplier')
ax.set_xlabel('Normalized Training Step')
ax.set_ylabel('Learning Rate Multiplier')
ax.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('lr_plot.pdf')
plt.show()