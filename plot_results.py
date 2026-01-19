'''
plotting of metrics.zarr files from training runs
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import zarr
import pandas as pd
from pathlib import Path
import zarr

# metrics = {}

# for p in Path("results/convnext-atto-lr-bs-wd-sweep-gpu-dispersion").rglob("*metrics.zarr"):
#     run_name = p.parent.name
#     runs[run_name] = zarr.open(p, mode="r")
#     # print(f"Loaded run: {p}")
#     print(runs)

import re
from pathlib import Path

pattern = re.compile(
    r"""
    (?P<model>[^_]+)
    _lr-(?P<lr>[\d.e-]+)
    _wd-(?P<wd>[\d.e-]+)
    _epochs-(?P<epochs>\d+)
    _(?P<scheduler>[^_]+)
    _warmup-(?P<warmup>\d+)
    _metrics\.zarr
    """,
    re.VERBOSE,
)

def parse_run_name(path: Path):
    m = pattern.match(path.name)
    if m is None:
        raise ValueError(f"Could not parse: {path.name}")
    d = m.groupdict()
    d["lr"] = float(d["lr"])
    d["wd"] = float(d["wd"])
    d["epochs"] = int(d["epochs"])
    d["warmup"] = int(d["warmup"])
    return d

import zarr
import pandas as pd

rows = []

base = Path("results/convnext-atto-lr-bs-wd-sweep-gpu-dispersion")

for p in base.rglob("*metrics.zarr"):
    meta = parse_run_name(p)
    z = zarr.open(p, mode="r")

    # example metric paths – adjust if needed
    train_loss = z["train_loss"][:]
    val_loss   = z["val_loss"][:]
    R2_train = z["R2_train"][:]
    R2_val   = z["R2_val"][:]

    for step, (tl, vl, tr, vr) in enumerate(zip(train_loss, val_loss, R2_train, R2_val)):
        rows.append({
            **meta,
            "step": step,
            "train_loss": tl,
            "val_loss": vl,
            "R2_train": tr,
            "R2_val": vr,
        })

df = pd.DataFrame(rows)
print(df)

# Plotting:
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")
import matplotlib.cm as cm

# Get unique learning rates and weight decays
learning_rates = sorted(df["lr"].unique())
weight_decays = sorted(df["wd"].unique())
n_subplots = len(learning_rates)

# Create color map
colors = cm.get_cmap('tab10')(range(len(weight_decays)))

fig, axes = plt.subplots(1, n_subplots, figsize=(12, 5), sharey=True)

# Handle case where there's only one learning rate
if n_subplots == 1:
    axes = [axes]

for idx, lr in enumerate(learning_rates):
    ax = axes[idx]
    lr_data = df[df["lr"] == lr]
    
    for wd_idx, wd in enumerate(weight_decays):
        group = lr_data[lr_data["wd"] == wd]
        if len(group) == 0:
            continue
        
        label = f"wd={wd}"
        color = colors[wd_idx]
        ax.plot(group["step"], group["R2_train"], label=label, 
                alpha=0.4, linestyle='--', color=color)
        ax.plot(group["step"], group["R2_val"], label=label, 
                linestyle='-', color=color)
    
    ax.set_yscale("log")
    ax.set_xscale("log")

    ax.set_ylim(0.5, 1.1)
    ax.set_xlabel("Training Step")
    ax.set_title(f"lr={lr}")
    
    # Create deduplicated legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels[1::2], handles[1::2]))
    ax.legend(by_label.values(), by_label.keys(), 
             fontsize=8, loc='best')

# Only add y-label to the leftmost subplot
axes[0].set_ylabel("R² Score")

# Add overall title and linestyle note
fig.suptitle("R² Score over Training Steps", fontsize=14, y=1.02)
fig.text(0.5, 0.01, 'Solid: Validation | Dashed: Train', 
         ha='center', fontsize=9, 
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
plt.show()

# grid plot of final R2_val for different lr and wd
final = df.loc[
    df.groupby(["lr", "wd"])["step"].idxmax()
]

pivot = final.pivot(
    index="lr",
    columns="wd",
    values="R2_val",
)

plt.figure(figsize=(8, 6))
sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis")
plt.xlabel("Weight Decay")
plt.ylabel("Learning Rate")
plt.title("Final R2_val for different Learning Rates and Weight Decays")
# plt.show()