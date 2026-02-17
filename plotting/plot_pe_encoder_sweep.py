import matplotlib.pyplot as plt
import numpy as np
import zarr
from matplotlib.lines import Line2D

folder = 'results/small_sweep_dispersion_epoch_pe_encoder_sweep/'
models = {
    'resnet': ['ResNet-18','ResNet-50'],
    'swin': ['Swin-T', 'Swin-S'],
    'vit': ['ViT-T16', 'ViT-S16'],
    'convnext': ['ConvNeXt-Atto','ConvNeXt-Small'],
}
length = [200, 100, 50]
pe_encoders = ['straight', 'log', 'vector']

# Color per length
length_colors = {
    50: 'C2',
    100: 'C3',
    200: 'C9',
}
# Color per PE encoder
pe_encoder_colors = {
    'straight': 'C0',
    'log': 'C1',
    'vector': 'C2',
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

for model_family, model_list in models.items():
    fig, axes = plt.subplots(
        len(model_list), len(pe_encoders),          # rows=models, cols=pe_encoders
        figsize=(3.2 * len(pe_encoders) / 2, 3.2 * len(model_list) / 2),
        sharex=True,
        sharey=True
    )
    
    axes = np.atleast_2d(axes)

    for row, m in enumerate(model_list):            # rows = models
        for col, pe_encoder in enumerate(pe_encoders):  # cols = pe_encoders
            ax = axes[row, col]

            for l in length:
                path = (
                    folder
                    + f'{m}_lr-0.0005_wd-0.05_bs-128_epochs-{l}_'
                    + f'cosine_warmup-1250.0_clipgrad-True_'
                    + f'pe-encoder-{pe_encoder}_pe-4_rmse_metrics.zarr'
                )
                try:
                    root = zarr.open(path, mode='r')
                    train_r2 = root['R2_train'][:]
                    val_r2 = root['R2_val'][:]
                except Exception as e:
                    print(f"Skipping {path}: {e}")
                    continue

                ax.plot(1 - train_r2, color=length_colors[l], linestyle=split_styles['train'], alpha=0.3)
                ax.plot(1 - val_r2,   color=length_colors[l], linestyle=split_styles['val'],   alpha=1., linewidth=1.)

            # Column titles only on top row
            if row == 0:
                ax.set_title(pe_encoder)
            
            # Row labels only on left column
            if col == 0:
                ax.set_ylabel(f'{m}\n' + r'$1-R^2$')

            # x-axis label only on bottom row
            if row == len(model_list) - 1:
                ax.set_xlabel('Epoch')

            ax.set_yscale('log')
            ax.grid(alpha=0.3)
            ax.grid(which='minor', alpha=0.15)
            ax.minorticks_on()

    # Legends (unchanged)
    length_legend = [
        Line2D([0], [0], color=length_colors[l], lw=2, label=l)
        for l in length
    ]
    split_legend = [
        Line2D([0], [0], color='black', linestyle=split_styles[s], lw=2, label=s)
        for s in split_styles
    ]

    fig.legend(handles=length_legend, title="Training length", loc="lower left",
               ncol=len(length), frameon=False)
    fig.legend(handles=split_legend, title="Split", loc="lower right",
               ncol=len(split_styles), frameon=False)

    plt.tight_layout(rect=[0, 0.08, 1.0, 1.0])
    plt.savefig(f'thesis_plots/{model_family}_epoch_pe_encoder_sweep.pdf', bbox_inches='tight')
    plt.close()