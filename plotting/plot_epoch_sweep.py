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
            figsize=figsize,
            sharex=True,
            sharey=True
        )
    else:
        fig, axes = plt.subplots(
            2, 4,
            figsize=(figsize[0], figsize[1]*2),
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
        title="Training length",
        loc="lower left",
        ncol=len(length),
        frameon=False
    )
    fig.legend(
        handles=split_legend,
        title="Split",
        loc="lower right",
        ncol=len(split_styles),
        frameon=False
    )
    
    plt.tight_layout(rect=[0, 0.10, 1.0, 1.0])
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

# # LaTeX width in points (example value, replace with your actual value)
# latex_width_pt = 418.25368 

# # Calculate figure width in inches
# fig_width_inches = latex_width_pt * 0.01389

# # Optional: Set height, e.g., using golden ratio (height = width / 1.618)
# golden_ratio = 1.618
# fig_height_inches = fig_width_inches / golden_ratio




for model_family, model_list in models.items():
    fig, ax = plt.subplots(figsize=figsize)
    cmap = family_cmaps[model_family]
    marker = family_markers[model_family]
    n = len(model_list)
    colors = [cmap(0.35 + 0.6 * i / max(n - 1, 1)) for i in range(n)]
    
    legend_model_handles = []  # Reset per family

    for i, m in enumerate(model_list):
        color = colors[i]
        xs, ys = [], []

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
                val_r2 = root['R2_val'][:]
                best = 1 - np.max(val_r2)
                xs.append(l)
                ys.append(best)
            except Exception as e:
                print(f"Skipping {path}: {e}")
                continue

        if not xs:
            continue

        ax.plot(xs, ys, color=color, linestyle='-', linewidth=0.9, alpha=0.5, zorder=2)
        ax.scatter(xs, ys, color=color, marker=marker, s=45, zorder=3,
                   edgecolors='white', linewidths=0.4)
        legend_model_handles.append(
            Line2D([0], [0], color=color, marker=marker, linestyle='-',
                   linewidth=1.2, markersize=5, label=m)
        )

    ax.set_yscale('log')
    ax.set_ylabel(r'Best $1 - R^2$')
    ax.set_xlabel('Training epochs')
    ax.set_xticks(length)
    ax.set_xticklabels(length)
    ax.grid(alpha=0.3)
    ax.grid(which='minor', alpha=0.15)
    ax.minorticks_on()
    ax.set_title(f'{model_family} — best validation $1-R^2$ across training lengths')

    leg_models = fig.legend(
        handles=legend_model_handles,
        title='Model',
        loc='center left',
        bbox_to_anchor=(0.9, 0.5),
        frameon=True,
        framealpha=0.9,
        edgecolor='#cccccc',
        ncol=1,
        fontsize=7,
    )

    plt.tight_layout(rect=[0, 0, 0.9, 1.0])
    plt.savefig(f'thesis_plots/best_r2_epoch_sweep_permeability_{model_family}.pdf', bbox_inches='tight')
    plt.close()

print("Saved.")