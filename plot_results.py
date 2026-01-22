'''
plotting of metrics.zarr files from training runs
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import zarr
import pandas as pd
from pathlib import Path
import re
import os
import seaborn as sns
import matplotlib.cm as cm

pattern = re.compile(
    r"""
    (?P<model>[^_]+)
    _lr-(?P<lr>[\d.e-]+)
    _wd-(?P<wd>[\d.e-]+)
    _bs-(?P<bs>\d+)
    _epochs-(?P<epochs>\d+)
    (?:_(?P<scheduler>[^_]+))?
    _cosine
    _warmup-(?P<warmup>\d+)
    _metrics\.zarr
    """,
    re.VERBOSE,
)

def parse_run_name(path: Path):
    '''
    Function for parsing run names from file paths.
    
    :param path: Path to metrics file
    :type path: Path
    '''
    m = pattern.match(path.name)
    if m is None:
        raise ValueError(f"Could not parse: {path.name}")
    d = m.groupdict()
    d["lr"] = float(d["lr"])
    d["wd"] = float(d["wd"])
    d["bs"] = int(d["bs"])
    d["epochs"] = int(d["epochs"])
    d["warmup"] = int(d["warmup"])
    return d

def FetchMetrics(path: Path):
    '''
    Function for fetching metrics from zarr files in a given path.
    
    :param path: Base path to search for metrics
    :type path: Path
    '''
    rows = []
    base = Path(path)
    for p in base.rglob("*metrics.zarr"):
        meta = parse_run_name(p)
        z = zarr.open(p, mode="r")
        train_loss = z["train_loss"][:]
        val_loss = z["val_loss"][:]
        R2_train = z["R2_train"][:]
        R2_val = z["R2_val"][:]
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
    return df

# Model families for colormap assignment
MODEL_FAMILIES = {
    "ConvNeXt": ["Atto", "Femto", "Pico", "Nano", "Tiny", "Small", "Base", "Large"],
    "ConvNeXt-V2": ["Atto", "Femto", "Pico", "Nano", "Tiny", "Small", "Base", "Large"],
    "ConvNeXt-RMS": ["Atto", "Femto", "Pico", "Nano", "Tiny", "Small", "Base", "Large"],
    "ResNet": ["18", "34", "50", "101", "152"],
    "ViT": ["T16", "S16", "B16", "L16"],
}

# Assign distinct colormaps for each model family
MODEL_CMAPS = {
    "ConvNeXt": "viridis",
    "ConvNeXt-V2": "plasma",
    "ConvNeXt-RMS": "cividis",
    "ResNet": "autumn",
    "ViT": "winter",
}

def get_model_family(model_name):
    '''
    Determine model family from model name.
    
    :param model_name: Full model name
    :return: Model family name or None
    '''
    for family in MODEL_FAMILIES.keys():
        if model_name.startswith(family):
            return family
    return None

def get_color_for_model(model_name, model_family=None):
    '''
    Get a unique color for a model based on its family.
    
    :param model_name: Full model name
    :param model_family: Optional pre-computed family name
    :return: RGB color tuple
    '''
    if model_family is None:
        model_family = get_model_family(model_name)
    
    if model_family is None:
        return (0.5, 0.5, 0.5)  # default gray
    
    # Get the variant list and colormap
    variants = MODEL_FAMILIES.get(model_family, [])
    cmap_name = MODEL_CMAPS.get(model_family, "viridis")
    cmap = cm.get_cmap(cmap_name)
    
    # Find the variant in the model name
    variant_idx = 0
    for idx, variant in enumerate(variants):
        if variant in model_name:
            variant_idx = idx
            break
    
    # Map variant index to color
    if len(variants) > 1:
        color_pos = variant_idx / (len(variants) - 1)
    else:
        color_pos = 0.5
    
    return cmap(color_pos)

def plot_metric_on_axis(ax, data, metric, model_colors, title=None):
    '''
    Plot a single metric on a given axis.
    
    :param ax: Matplotlib axis
    :param data: DataFrame with model data
    :param metric: Metric name (e.g., 'R2')
    :param model_colors: Dictionary mapping model names to colors
    :param title: Optional title for the subplot
    '''
    unique_models = sorted(data['model'].unique())
    
    for model in unique_models:
        model_data = data[data['model'] == model]
        color = model_colors[model]
        
        # Plot train
        # ax.plot(
        #     model_data['step'],
        #     model_data[f'{metric}_train'],
        #     color=color,
        #     linestyle='--',
        #     alpha=0.6,
        #     linewidth=1.5,
        #     label=f'{model} (train)'
        # )
        
        # Plot validation
        ax.plot(
            model_data['step'],
            model_data[f'{metric}_val'],
            color=color,
            linestyle='-',
            alpha=0.7,
            linewidth=2,
            label=f'{model} (val)'
        )
    
    ax.set_xlabel("Training Steps", fontsize=11)
    ax.set_ylabel(metric, fontsize=11)
    ax.set_xscale('log')
    if title:
        ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
             frameon=True, fontsize=8)

def plot_training(runs, metrics_to_plot=None, ylim=None, 
                 subplot_by='none', models_to_include=None):
    '''
    Function for plotting metrics over epochs with flexible subplot organization.
    
    :param runs: Dictionary of run names to paths
    :param metrics_to_plot: List of metric names to plot (e.g., ['R2', 'loss'])
    :param ylim: Optional y-axis limits as tuple (min, max) or dict {metric: (min, max)}
    :param subplot_by: How to organize subplots - 'none', 'family', 'model', or list of model groups
    :param models_to_include: Optional list of model names to include (filters others out)
    '''
    if metrics_to_plot is None:
        metrics_to_plot = ['R2']
    
    # Fetch all data
    all_dfs = []
    for run_name, run_path in runs.items():
        print(f"Processing run: {run_name}")
        df = FetchMetrics(Path(run_path))
        df = df.assign(run=run_name)
        all_dfs.append(df)
    
    if not all_dfs:
        print("No runs found.")
        return
    
    combined = pd.concat(all_dfs, ignore_index=True)
    
    # Filter models if specified
    if models_to_include:
        combined = combined[combined['model'].isin(models_to_include)]
    
    # Get unique models and assign colors
    unique_models = combined['model'].unique()
    model_colors = {}
    for model in unique_models:
        model_family = get_model_family(model)
        model_colors[model] = get_color_for_model(model, model_family)
    
    # Organize subplots based on subplot_by parameter
    if subplot_by == 'none':
        # Single plot with all models
        n_metrics = len(metrics_to_plot)
        fig, axes = plt.subplots(n_metrics, 1, figsize=(14, 6 * n_metrics))
        if n_metrics == 1:
            axes = [axes]
        
        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx]
            plot_metric_on_axis(ax, combined, metric, model_colors, 
                              f"{metric} over Training Steps")
            
            # Apply ylim
            if ylim:
                if isinstance(ylim, dict):
                    if metric in ylim:
                        ax.set_ylim(ylim[metric])
                else:
                    ax.set_ylim(ylim)
    
    elif subplot_by == 'family':
        # Separate subplot for each model family
        model_families = {}
        for model in unique_models:
            family = get_model_family(model)
            if family not in model_families:
                model_families[family] = []
            model_families[family].append(model)
        
        n_families = len(model_families)
        n_metrics = len(metrics_to_plot)
        
        fig, axes = plt.subplots(n_families, n_metrics, 
                                figsize=(6 * n_metrics, 4 * n_families),
                                squeeze=False)
        
        for f_idx, (family, models) in enumerate(sorted(model_families.items())):
            family_data = combined[combined['model'].isin(models)]
            
            for m_idx, metric in enumerate(metrics_to_plot):
                ax = axes[f_idx, m_idx]
                plot_metric_on_axis(ax, family_data, metric, model_colors,
                                  f"{family} - {metric}")
                
                # Apply ylim
                if ylim:
                    if isinstance(ylim, dict):
                        if metric in ylim:
                            ax.set_ylim(ylim[metric])
                    else:
                        ax.set_ylim(ylim)
    
    elif subplot_by == 'model':
        # Separate subplot for each model
        n_models = len(unique_models)
        n_metrics = len(metrics_to_plot)
        
        # Determine grid layout
        ncols = min(3, n_models)  # Max 3 columns
        nrows = (n_models + ncols - 1) // ncols
        
        fig, axes = plt.subplots(nrows, ncols, 
                                figsize=(7 * ncols, 5 * nrows),
                                squeeze=False)
        
        for idx, model in enumerate(sorted(unique_models)):
            row = idx // ncols
            col = idx % ncols
            ax = axes[row, col]
            
            model_data = combined[combined['model'] == model]
            
            # Plot all metrics on same axis for this model
            for metric in metrics_to_plot:
                color = model_colors[model]
                
                # Plot train
                # ax.plot(
                #     model_data['step'],
                #     model_data[f'{metric}_train'],
                #     color=color,
                #     linestyle='--',
                #     alpha=0.6,
                #     linewidth=1.5,
                #     label=f'{metric} (train)'
                # )
                
                # Plot validation
                ax.plot(
                    model_data['step'],
                    model_data[f'{metric}_val'],
                    color=color,
                    linestyle='-',
                    alpha=0.7,
                    linewidth=2,
                    label=f'{metric} (val)'
                )
            
            ax.set_title(model, fontsize=12, fontweight='bold')
            ax.set_xlabel("Training Steps", fontsize=10)
            ax.set_ylabel("Metric Value", fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            
            # Apply ylim
            if ylim and not isinstance(ylim, dict):
                ax.set_ylim(ylim)
        
        # Hide empty subplots
        for idx in range(n_models, nrows * ncols):
            row = idx // ncols
            col = idx % ncols
            axes[row, col].axis('off')
    
    elif isinstance(subplot_by, list):
        # Custom grouping - subplot_by is list of lists of model names
        n_groups = len(subplot_by)
        n_metrics = len(metrics_to_plot)
        
        fig, axes = plt.subplots(n_groups, n_metrics, 
                                figsize=(6 * n_metrics, 4 * n_groups),
                                squeeze=False)
        
        for g_idx, group in enumerate(subplot_by):
            group_data = combined[combined['model'].isin(group)]
            group_name = " vs ".join(group)
            
            for m_idx, metric in enumerate(metrics_to_plot):
                ax = axes[g_idx, m_idx]
                plot_metric_on_axis(ax, group_data, metric, model_colors,
                                  f"{group_name} - {metric}")
                
                # Apply ylim
                if ylim:
                    if isinstance(ylim, dict):
                        if metric in ylim:
                            ax.set_ylim(ylim[metric])
                    else:
                        ax.set_ylim(ylim)
    
    plt.tight_layout()
    # plt.show()
    plt.savefig("training_metrics.png", dpi=300)

# Usage examples
if __name__ == "__main__":
    results = "results"
    runs = {
        "all_models_permeability": os.path.join(results, "all_models_permeability")
    }
    
    # Example 1: All models in one plot
    # plot_training(runs, metrics_to_plot=['R2'], ylim=(0.5, 1.1))
    
    # Example 2: Separate by model family
    # plot_training(runs, metrics_to_plot=['R2'], ylim=(0.5, 1.1), subplot_by='family')
    
    # Example 3: Each model gets its own subplot
    # plot_training(runs, metrics_to_plot=['R2'],ylim=(0.5, 1.1), subplot_by='model')
    
    # Example 4: Custom grouping - compare specific models
    # plot_training(runs, 
    #              metrics_to_plot=['R2'], 
    #              ylim=(0.5, 1.1),
    #              subplot_by=[
    #                 #  ['ConvNeXt-Tiny', 'ResNet-50', 'ViT-S16'],
    #                 #  ['ConvNeXt-Base', 'ResNet-101', 'ViT-B16']
    #                 ["ConvNeXt-Atto", "ConvNeXt-Femto", "ConvNeXt-Pico", "ConvNeXt-Nano", "ConvNeXt-Tiny", "ConvNeXt-Small", "ConvNeXt-Base", "ConvNeXt-Large"],
    #                 ["ConvNeXt-V2-Atto", "ConvNeXt-V2-Femto", "ConvNeXt-V2-Pico", "ConvNeXt-V2-Nano", "ConvNeXt-V2-Tiny", "ConvNeXt-V2-Small", "ConvNeXt-V2-Base", "ConvNeXt-V2-Large"],
    #                 ["ConvNeXt-RMS-Atto", "ConvNeXt-RMS-Femto", "ConvNeXt-RMS-Pico", "ConvNeXt-RMS-Nano", "ConvNeXt-RMS-Tiny", "ConvNeXt-RMS-Small", "ConvNeXt-RMS-Base", "ConvNeXt-RMS-Large"]
    #              ])
    plot_training(runs, 
                 metrics_to_plot=['R2'], 
                 ylim=(0.8, 1.01),
                 subplot_by=[
                    #  ['ConvNeXt-Tiny', 'ResNet-50', 'ViT-S16'],
                    #  ['ConvNeXt-Base', 'ResNet-101', 'ViT-B16']
                    # ["ConvNeXt-Atto", "ConvNeXt-Femto", "ConvNeXt-Pico", "ConvNeXt-Nano", "ConvNeXt-Tiny", "ConvNeXt-Small", "ConvNeXt-Base", "ConvNeXt-Large"],
                    # ["ConvNeXt-V2-Atto", "ConvNeXt-V2-Femto", "ConvNeXt-V2-Pico", "ConvNeXt-V2-Nano", "ConvNeXt-V2-Tiny", "ConvNeXt-V2-Small", "ConvNeXt-V2-Base", "ConvNeXt-V2-Large"],
                    # ["ConvNeXt-RMS-Atto", "ConvNeXt-RMS-Femto", "ConvNeXt-RMS-Pico", "ConvNeXt-RMS-Nano", "ConvNeXt-RMS-Tiny", "ConvNeXt-RMS-Small", "ConvNeXt-RMS-Base", "ConvNeXt-RMS-Large"]
                    ['ViT-T16'],['ViT-S16'],['ViT-B16'],['ViT-L16']
                 ])
    
    # Example 5: Filter to only include specific models
    # plot_training(runs, 
    #              metrics_to_plot=['R2'], 
    #              ylim=(0.5, 1.1),
    #              models_to_include=['ViT-T16','ViT-S16''ViT-B16','ViT-L16'])