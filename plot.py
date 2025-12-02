import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

num_samples = 64
D_m = 1
L = 128
t = L**2 / D_m

sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(2, 2, figsize=(8,6))

start = -10000
path_to_data = 'data/self_diffusivity_results_validation'
for i in range(num_samples):
    print(f"loading sample {i}...")
    M = np.load(os.path.join(path_to_data, f"simulation_result_{i}.npz"))
    Mx = M["Mx"]      # shape: (T, 2, 2)
    # My is never used — skip loading if unnecessary

    t_ = np.arange(Mx.shape[0]) * 1e-3

    print(f"plotting sample {i}...")
    ax[0,0].plot(Mx[start:,0,0] / (2*t_[start:]))
    ax[0,1].plot(Mx[start:,0,1] / (2*t_[start:]))
    ax[1,0].plot(Mx[start:,1,0] / (2*t_[start:]))
    ax[1,1].plot(Mx[start:,1,1] / (2*t_[start:]))
path_to_data = 'data/self_diffusivity_results_test'
for i in range(num_samples):
    print(f"loading sample {i}...")
    M = np.load(os.path.join(path_to_data, f"simulation_result_{i}.npz"))
    Mx = M["Mx"]      # shape: (T, 2, 2)
    # My is never used — skip loading if unnecessary

    t_ = np.arange(Mx.shape[0]) * 1e-3

    print(f"plotting sample {i}...")
    ax[0,0].plot(Mx[start:,0,0] / (2*t_[start:]))
    ax[0,1].plot(Mx[start:,0,1] / (2*t_[start:]))
    ax[1,0].plot(Mx[start:,1,0] / (2*t_[start:]))
    ax[1,1].plot(Mx[start:,1,1] / (2*t_[start:]))

plt.tight_layout()
plt.savefig("example.pdf")
