import numpy as np
import matplotlib.pyplot as plt
from ploting import figsize
from plottools import MarkerCycler

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

x = np.linspace(-4, 4, 1000)
plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
})
fig, ax = plt.subplots(figsize=(figsize[0],figsize[1]*0.8))
ax.plot(x, sigmoid(x), label='Sigmoid', color='C0',linestyle='-')
ax.plot(x, relu(x), label='ReLU', color='C1',linestyle='--')
ax.plot(x, gelu(x), label='GELU', color='C2',linestyle='-.')
ax.plot(x, np.tanh(x), label='Tanh', color='C3',linestyle=':')
# ax.set_title('Activation Functions')
ax.set_xlabel('Input')
ax.set_ylabel('Output')
ax.legend(loc='upper left')
ax.set_ylim(-1.2,2.5)
ax.set_xlim(-4,4)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('thesis_plots/activation_functions.pdf')
plt.close()