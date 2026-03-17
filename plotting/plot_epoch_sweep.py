import matplotlib.pyplot as plt
import numpy as np
import zarr
from matplotlib.lines import Line2D
from ploting import figsize

folder = 'results/epoch_sweep_all_models/'
models = {
    'resnet': ['ResNet-18', 'ResNet-34', 'ResNet-50', 'ResNet-101', 'ResNet-152'],
    'swin': ['Swin-T', 'Swin-S', 'Swin-B', 'Swin-L'],
    'vit': ['ViT-T16', 'ViT-S16', 'ViT-B16', 'ViT-L16'],
    'convnext': ['ConvNeXt-Atto', 'ConvNeXt-Femto', 'ConvNeXt-Pico', 'ConvNeXt-Nano', 
                 'ConvNeXt-Tiny', 'ConvNeXt-Small', 'ConvNeXt-Base', 'ConvNeXt-Large'],
    'convnext-v2': ['ConvNeXt-V2-Atto', 'ConvNeXt-V2-Femto', 'ConvNeXt-V2-Pico', 'ConvNeXt-V2-Nano', 
                 'ConvNeXt-V2-Tiny', 'ConvNeXt-V2-Small', 'ConvNeXt-V2-Base', 'ConvNeXt-V2-Large'],
    'convnext-rms': ['ConvNeXt-RMS-Atto', 'ConvNeXt-RMS-Femto', 'ConvNeXt-RMS-Pico', 'ConvNeXt-RMS-Nano', 
                 'ConvNeXt-RMS-Tiny', 'ConvNeXt-RMS-Small', 'ConvNeXt-RMS-Base', 'ConvNeXt-RMS-Large'],
}
length = [1500, 1000, 700, 500, 300, 100]

# Color per length
length_colors = {
    1500: 'C5',
    1000: 'C4',
    700: 'C3',
    500: 'C2',
    300: 'C1',
    100: 'C0',
}

# Linestyle per split
split_styles = {
    'train': '--',
    'val': '-'
}

plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
})

# Create a separate plot for each model family
for model_family, model_list in models.items():
    if model_family != 'convnext' and model_family != 'convnext-v2' and model_family != 'convnext-rms':  # Only plot for convnext family
        fig, axes = plt.subplots(
            1, len(model_list),
            figsize=(figsize[0],figsize[1]*0.7),
            sharex=True,
            sharey=True
        )
    else:
        fig, axes = plt.subplots(
            2, 4,
            figsize=(figsize[0], figsize[1]*0.7*2),
            sharex=True,
            sharey=True
        )
        axes = axes.flatten()
    
    # Handle case where there's only one model
    if len(model_list) == 1:
        axes = [axes]
    
    for col, m in enumerate(model_list):
        ax_r2 = axes[col]
        #results/permeability_epoch_sweep/ConvNeXt-Atto_lr-0.0008_wd-0.1_bs-128_epochs-100_cosine_warmup-1250_clipgrad-True_pe-encoder-None_pe-None_rmse_metrics.zarr
        for l in length:
            if l ==1500:
                path = (
                    folder
                    + f'{m}_lr-0.0005_wd-0.1_bs-128_epochs-{l}_cosine_warmup-3750.0_'
                    + f'clipgrad-True_pe-encoder-None_pe-None_mse_metrics.zarr'
                )
            else: 
                path = (
                    folder
                    + f'{m}_lr-0.0005_wd-0.1_bs-128_epochs-{l}_cosine_warmup-0_'
                    + f'clipgrad-True_pe-encoder-None_pe-None_mse_metrics.zarr'
                )
            try:
                root = zarr.open(path, mode='r')
                train_loss = root['train_loss'][:]
                val_loss = root['val_loss'][:]
                train_r2 = root['R2_train'][:]
                val_r2 = root['R2_val'][:]
            except Exception as e:
                print(f"Skipping {path}: {e}")
                continue
            
            # Plot R2
            ax_r2.plot(1 - train_r2, color=length_colors[l], linestyle=split_styles['train'], alpha=0.3)
            ax_r2.plot(1 - val_r2, color=length_colors[l], linestyle=split_styles['val'], alpha=1.,linewidth=1.)

        
        ax_r2.set_title(m)
        ax_r2.set_xlabel('Epoch')
        ax_r2.set_yscale('log')
        # ax_r2.set_xscale('log')
        # ax_r2.set_xlim(10, 1100)
        ax_r2.grid(alpha=0.3)
    
    axes[0].set_ylabel(r'$1-R^2$')
    if model_family == 'convnext' or model_family == 'convnext-v2' or model_family == 'convnext-rms':
        axes[4].set_ylabel(r'$1-R^2$')
    
    # Legends
    loss_legend = [
        Line2D([0], [0], color=length_colors[l], lw=2, label=l)
        for l in length
    ]
    split_legend = [
        Line2D([0], [0], color='black', linestyle=split_styles[s], lw=2, label=s)
        for s in split_styles
    ]
    
    leg1 = fig.legend(
        handles=loss_legend,
        title="Training length:",
        loc="lower center",
        ncol=len(length),
        frameon=False
    )
    fig.legend(
        handles=split_legend,
        title="Split:",
        loc="upper center",
        ncol=len(split_styles),
        frameon=False
    )
    
    plt.tight_layout(rect=[0, 0.10, 1.0, 0.9])
    plt.savefig(f'thesis_plots/{model_family}_epoch_sweep_permeability.pdf')
    plt.close()


# Best 1-R2 for each model combined plot
# Family markers
family_markers = {
    'resnet': 'o',
    'swin': 's',
    'vit': 'D',
    'convnext': '^',
    'convnext-v2': 'v',
    'convnext-rms': 'P',
}

# Family colors: assign each family a distinct colormap for its models
import matplotlib.cm as cm
family_cmaps = {
    'resnet':       cm.cool,
    'swin':         cm.cool,
    'vit':          cm.cool,
    'convnext':     cm.cool,
    'convnext-v2':  cm.cool,
    'convnext-rms': cm.cool,
}

plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
})


legend_family_handles = []
legend_model_handles = []
# pt_to_inch = 1.0 / 72.27

# LaTeX width in points (example value, replace with your actual value)
# latex_width_pt = 418.25368 

# # Calculate figure width in inches
# fig_width_inches = latex_width_pt * 0.01389

# # Optional: Set height, e.g., using golden ratio (height = width / 1.618)
# golden_ratio = 1.618
# fig_height_inches = fig_width_inches / golden_ratio


for model_family, model_list in models.items():

    fig, ax = plt.subplots(figsize=(figsize[0],figsize[1]*0.7))

    cmap = family_cmaps[model_family]
    marker = family_markers[model_family]
    n = len(model_list)
    colors = [cmap(0.35 + 0.6 * i / max(n - 1, 1)) for i in range(n)]

    legend_model_handles = []

    for i, m in enumerate(model_list):
        color = colors[i]
        xs, ys = [], []

        for l in length:
            if l == 1500:
                path = (
                    folder
                    + f'{m}_lr-0.0005_wd-0.1_bs-128_epochs-{l}_cosine_warmup-3750.0_'
                    + f'clipgrad-True_pe-encoder-None_pe-None_mse_metrics.zarr'
                )
            else:
                path = (
                    folder
                    + f'{m}_lr-0.0005_wd-0.1_bs-128_epochs-{l}_cosine_warmup-0_'
                    + f'clipgrad-True_pe-encoder-None_pe-None_mse_metrics.zarr'
                )

            try:
                root = zarr.open(path, mode='r')
                val_r2 = root['R2_val'][:]
                best = 1 - np.max(val_r2)
                xs.append(l)
                ys.append(best)
            except Exception:
                continue

        if not xs:
            continue

        ax.plot(xs, ys, color=color, linewidth=0.9, alpha=0.5)
        ax.scatter(xs, ys, color=color, marker=marker, s=45,
                   edgecolors='white', linewidths=0.4)

        legend_model_handles.append(
            Line2D([0], [0],
                   color=color,
                   marker=marker,
                   linestyle='-',
                   linewidth=1.2,
                   markersize=5,
                   label=m)
        )

    # ----- Axis formatting -----
    ax.set_yscale('log')
    ax.set_ylabel(r'Lowest $1 - R^2$')
    ax.set_xlabel('Training epochs')
    ax.set_xticks(length)
    ax.grid(alpha=0.3)
    ax.grid(which='minor', alpha=0.15)
    ax.minorticks_on()

    # ----- Legend inside plot -----
    ax.legend(
        handles=legend_model_handles,
        title='Model',
        # loc='upper right',   # fixed position = consistent layout
        frameon=True,
        framealpha=0.3,
        edgecolor='#cccccc',
        fontsize=7,
        labelspacing=0.3,
        handlelength=1.5,
        handletextpad=0.4,
    )

    plt.tight_layout()
    plt.savefig(
        f'thesis_plots/best_r2_epoch_sweep_permeability_{model_family}.pdf'
    )
    plt.close()

print("Saved.")


families = {
    "ConvNeXt-V2": ["Atto", "Femto", "Pico", "Nano", "Tiny", "Small", "Base", "Large"],
    "ConvNeXt-RMS": ["Atto", "Femto", "Pico", "Nano", "Tiny", "Small", "Base", "Large"],
    "ConvNeXt": ["Atto", "Femto", "Pico", "Nano", "Tiny", "Small", "Base", "Large"],
    "ResNet": ["18", "34", "50", "101", "152"],
    "ViT": ["T16", "S16", "B16", "L16"],
}

models = {
    'resnet': ['ResNet-18', 'ResNet-34', 'ResNet-50', 'ResNet-101', 'ResNet-152'],
    'swin': ['Swin-T', 'Swin-S', 'Swin-B', 'Swin-L'],
    'vit': ['ViT-T16', 'ViT-S16', 'ViT-B16', 'ViT-L16'],
    'convnext': ['ConvNeXt-Atto', 'ConvNeXt-Femto', 'ConvNeXt-Pico', 'ConvNeXt-Nano', 
                 'ConvNeXt-Tiny', 'ConvNeXt-Small', 'ConvNeXt-Base', 'ConvNeXt-Large'],
    'convnext-v2': ['ConvNeXt-V2-Atto', 'ConvNeXt-V2-Femto', 'ConvNeXt-V2-Pico', 'ConvNeXt-V2-Nano', 
                 'ConvNeXt-V2-Tiny', 'ConvNeXt-V2-Small', 'ConvNeXt-V2-Base', 'ConvNeXt-V2-Large'],
    'convnext-rms': ['ConvNeXt-RMS-Atto', 'ConvNeXt-RMS-Femto', 'ConvNeXt-RMS-Pico', 'ConvNeXt-RMS-Nano', 
                 'ConvNeXt-RMS-Tiny', 'ConvNeXt-RMS-Small', 'ConvNeXt-RMS-Base', 'ConvNeXt-RMS-Large'],
}

# Added Swin sizes and standardized keys
sizes = {
    "ConvNeXt-V2": [3388604, 4849684, 8555204, 14985844, 27871588, 49561444, 87708804, 196443844],
    "ConvNeXt-RMS": [3371724, 4829428, 8528196, 14946324, 27811204, 49438852, 87545348, 196198660],
    "ConvNeXt": [3373884, 4832020, 8531652, 14951284, 27818596, 49453156, 87564420, 196227268],
    "ResNet": [11172292, 21280452, 23509956, 42502084, 58145732],
    "ViT": [5401156, 21419140, 85305604, 302644228],
    "Swin": [27504334, 48804958, 86700156, 194930872]
}

# Mapping from models dict key -> run_model_test.py --model arg
FAMILY_TO_MODEL_ARG = {
    'resnet':       'resnet',
    'vit':          'vit',
    'swin':         'swin',
    'convnext':     'convnext',
    'convnext-v2':  'convnext',
    'convnext-rms': 'convnext',
}

# --version arg per family
FAMILY_TO_VERSION = {
    'resnet':       'None',
    'vit':          'None',
    'swin':         'None',
    'convnext':     'v1',
    'convnext-v2':  'v2',
    'convnext-rms': 'rms',
}

def build_test_cmd(local_path: str, model_name: str, model_family: str, task: str = 'permeability') -> list[str]:
    """Build the run_model_test.py command from loop variables."""
    # Extract size: last hyphen-separated token, e.g. "Large", "18", "B16"
    size = model_name.split('-')[-1].lower()

    return [
        "python3", "run_model_test.py",
        "--pretrained_path", local_path,
        "--model",           FAMILY_TO_MODEL_ARG[model_family],
        "--model_name",      model_name,
        "--size",            size,
        "--version",         FAMILY_TO_VERSION[model_family],
        "--task",            task,
        "--loss_function",   "mse",
    ]

# Mapping to handle the lowercase keys in 'models'
family_map = {
    'resnet': 'ResNet', 'swin': 'Swin', 'vit': 'ViT',
    'convnext': 'ConvNeXt', 'convnext-v2': 'ConvNeXt-V2', 'convnext-rms': 'ConvNeXt-RMS'
}
family_cmaps = {
    'resnet': 'C0',
    'swin': 'C1',
    'vit': 'C2',
    'convnext': 'C3',
    'convnext-v2': 'C4',
    'convnext-rms': 'C9'
}
import json
CACHE_FILE = "thesis_plots/all_family_data_cache.json"
family_markers = {
    'resnet': 'o', 'swin': 's', 'vit': '^', 
    'convnext': 'D', 'convnext-v2': 'v', 'convnext-rms': 'p'
}
import subprocess
import os
import re
if os.path.exists(CACHE_FILE):
    print("Loading cached family data...")
    with open(CACHE_FILE, 'r') as f:
        all_family_data = json.load(f)
    # json doesn't preserve tuples, convert lists back
    all_family_data = {
        fam: ([(p, v, c, name, t) for p, v, c, name, t in entries], marker)
        for fam, (entries, marker) in all_family_data.items()
    }
else:
    length = [1500, 1000, 700, 500, 300, 100]
    all_family_data = {}

    for model_family, model_list in models.items():
        size_key = family_map.get(model_family)
        parameter_counts = sizes.get(size_key, [])
        color  = family_cmaps.get(model_family, 'C0')
        marker = family_markers.get(model_family, 'o')
        n_models   = len(model_list)
        family_data = []

        for i, (m, p_size) in enumerate(zip(model_list, parameter_counts)):
            current_best_path = None
            best_r2_for_model = -np.inf
            test_r2_for_model = None

            for l in length:
                warmup = "3750.0" if l == 1500 else "0"
                path = (f"{folder}{m}_lr-0.0005_wd-0.1_bs-128_epochs-{l}_"
                        f"cosine_warmup-{warmup}_clipgrad-True_pe-encoder-None_"
                        f"pe-None_mse_metrics.zarr")
                try:
                    root = zarr.open(path, mode='r')
                    val_r2 = root['R2_val'][:]
                    max_in_file = np.max(val_r2)
                    if max_in_file > best_r2_for_model:
                        best_r2_for_model = max_in_file
                        current_best_path = path
                except:
                    continue

            if current_best_path:
                current_best_path = current_best_path.strip('metrics.zarr')[:-1] + '.pth'
                local = f"res{current_best_path}"
                servers = [
                    ("bigfacet", "bigfacet:/home/users/sigursv/porousflow"),
                    ("herbie",   "herbie-jump:/home/sigursv/porousflow"),
                ]

                if not os.path.exists(local):
                    for name, base in servers:
                        try:
                            subprocess.run(["rsync", "-av", f"{base}/res{current_best_path}", local], check=True)
                            print(f"Fetched from {name}")
                            break
                        except subprocess.CalledProcessError:
                            print(f"Not found on {name}, trying next...")
                    else:
                        print("File does not exist on any familiar cluster.")

                if os.path.exists(local):
                    cmd = build_test_cmd(local, m, model_family)
                    print("Running:", " ".join(cmd))
                    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                    match = re.search(r"Test R2:\s*([\d.]+)", result.stdout)
                    if match:
                        test_r2_for_model = float(match.group(1))
                        print(f"Test R2:       {test_r2_for_model:.5f}")
                    print(f"Validation R2: {best_r2_for_model:.5f}")

            if best_r2_for_model != -np.inf:
                test_error_val = (float(1 - test_r2_for_model)) if test_r2_for_model is not None else None
                family_data.append((
                    int(p_size),
                    float(1 - best_r2_for_model),
                    color,
                    m,
                    test_error_val
                ))

        all_family_data[model_family] = (family_data, marker)

    os.makedirs("thesis_plots", exist_ok=True)
    with open(CACHE_FILE, 'w') as f:
        json.dump(all_family_data, f, indent=2)
    print(f"Saved family data to {CACHE_FILE}")


def make_scaling_plot(all_family_data, path: str):
    fig, axes = plt.subplots(2, 1, figsize=(figsize[0], figsize[1]*1.45),
                             sharex=True, sharey=True)

    for ax, use_test in zip(axes, [False, True]):
        for model_family, (family_data, marker) in all_family_data.items():
            color = family_cmaps.get(model_family, 'C0')

            if use_test:
                plot_data = [(x, te, name) for x, ve, c, name, te in family_data if te is not None]
            else:
                plot_data = [(x, ve, name) for x, ve, c, name, te in family_data]

            if not plot_data:
                continue

            plot_data.sort(key=lambda x: x[0])
            xs, ys, names = zip(*plot_data)

            ax.plot(xs, ys, color=color, alpha=0.3, zorder=1)
            for x, y, name in plot_data:
                ax.plot(x, y, color=color, linestyle='', marker=marker,
                        markersize=7, fillstyle='none', zorder=2)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylabel(r'Test $1 - R^2$' if use_test else r'Validation $1 - R^2$')
        ax.grid(True, which="both", ls="-", alpha=0.15)

    axes[1].set_xlabel('Total Parameters')

    legend_elements = [
        Line2D([0], [0], marker=family_markers[f], color=family_cmaps[f],
               label=family_map[f], markersize=8, fillstyle='none', linestyle='-')
        for f in models.keys()
    ]
    axes[0].legend(handles=legend_elements, title="Architectures",
                   loc='best', fontsize=8)

    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"Saved {path}")

make_scaling_plot(all_family_data, 'thesis_plots/scaling_laws_r2_vs_params.pdf')