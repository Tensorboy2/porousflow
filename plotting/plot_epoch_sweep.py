import matplotlib.pyplot as plt
import numpy as np
import zarr
from matplotlib.lines import Line2D

folder = 'results/permeability_epoch_sweep/'
models = {
    'resnet': ['ResNet-18', 'ResNet-34', 'ResNet-50', 'ResNet-101', 'ResNet-152'],
    'swin': ['Swin-T', 'Swin-S', 'Swin-B', 'Swin-L'],
    'convnext': ['ConvNeXt-Atto', 'ConvNeXt-Femto', 'ConvNeXt-Pico', 'ConvNeXt-Nano', 
                 'ConvNeXt-Tiny', 'ConvNeXt-Small', 'ConvNeXt-Base'],
}
length = [1000, 700, 500, 300, 100]

# Color per length
length_colors = {
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
    fig, axes = plt.subplots(
        1, len(model_list),
        figsize=(3.2 * len(model_list) / 2, 3.2),
        sharex=True,
        sharey=True
    )
    
    # Handle case where there's only one model
    if len(model_list) == 1:
        axes = [axes]
    
    for col, m in enumerate(model_list):
        ax_r2 = axes[col]
        #results/permeability_epoch_sweep/ConvNeXt-Atto_lr-0.0008_wd-0.1_bs-128_epochs-100_cosine_warmup-1250_clipgrad-True_pe-encoder-None_pe-None_rmse_metrics.zarr
        for l in length:
            path = folder + f'{m}_lr-0.0008_wd-0.1_bs-128_epochs-{l}_cosine_warmup-1250_clipgrad-True_pe-encoder-None_pe-None_rmse_metrics.zarr'
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
        ax_r2.set_xscale('log')
        ax_r2.set_xlim(10, 1100)
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
    plt.savefig(f'thesis_plots/{model_family}_long_run.pdf')
    plt.close()