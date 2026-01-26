"""
Refactored plotting of metrics.zarr files from training runs.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import zarr
import pandas as pd
from pathlib import Path
import re
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass


# Configuration
@dataclass
class ModelConfig:
    """Configuration for model families and their visual properties."""
    families: Dict[str, List[str]] = None
    colormaps: Dict[str, str] = None
    
    def __post_init__(self):
        if self.families is None:
            self.families = {
                "ConvNeXt-V2": ["Atto", "Femto", "Pico", "Nano", "Tiny", "Small", "Base", "Large"],
                "ConvNeXt-RMS": ["Atto", "Femto", "Pico", "Nano", "Tiny", "Small", "Base", "Large"],
                "ConvNeXt": ["Atto", "Femto", "Pico", "Nano", "Tiny", "Small", "Base", "Large"],
                "ResNet": ["18", "34", "50", "101", "152"],
                "ViT": ["T16", "S16", "B16", "L16"],
            }
        
        if self.colormaps is None:
            self.colormaps = {
                "ConvNeXt-V2": "plasma",
                "ConvNeXt-RMS": "cividis",
                "ConvNeXt": "viridis",
                "ResNet": "autumn",
                "ViT": "winter",
            }


# Pattern for parsing run names
FILENAME_PATTERN = re.compile(
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


class MetricsLoader:
    """Handles loading and parsing of metrics files."""
    
    @staticmethod
    def parse_run_name(path: Path) -> Dict:
        """Parse run metadata from filename."""
        match = FILENAME_PATTERN.match(path.name)
        if not match:
            raise ValueError(f"Could not parse: {path.name}")
        
        metadata = match.groupdict()
        metadata["lr"] = float(metadata["lr"])
        metadata["wd"] = float(metadata["wd"])
        metadata["bs"] = int(metadata["bs"])
        metadata["epochs"] = int(metadata["epochs"])
        metadata["warmup"] = int(metadata["warmup"])
        return metadata
    
    @staticmethod
    def load_zarr_metrics(path: Path) -> Dict[str, np.ndarray]:
        """Load metrics from a zarr file."""
        z = zarr.open(path, mode="r")
        return {
            "train_loss": z["train_loss"][:],
            "val_loss": z["val_loss"][:],
            "R2_train": z["R2_train"][:],
            "R2_val": z["R2_val"][:],
        }
    
    def fetch_metrics(self, base_path: Path) -> pd.DataFrame:
        """Fetch all metrics from zarr files in path."""
        rows = []
        
        for metrics_file in Path(base_path).rglob("*metrics.zarr"):
            try:
                metadata = self.parse_run_name(metrics_file)
                metrics = self.load_zarr_metrics(metrics_file)
                
                # Combine metrics into rows
                n_steps = len(metrics["train_loss"])
                for step in range(n_steps):
                    row = {
                        **metadata,
                        "step": step,
                        "train_loss": metrics["train_loss"][step],
                        "val_loss": metrics["val_loss"][step],
                        "R2_train": metrics["R2_train"][step],
                        "R2_val": metrics["R2_val"][step],
                    }
                    rows.append(row)
            except Exception as e:
                print(f"Warning: Failed to process {metrics_file}: {e}")
        
        return pd.DataFrame(rows)


class ColorManager:
    """Manages color assignment for models."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
    
    def get_model_family(self, model_name: str) -> Optional[str]:
        """Determine model family from model name."""
        # Check longer names first to avoid ConvNeXt matching ConvNeXt-V2
        sorted_families = sorted(self.config.families.keys(), key=len, reverse=True)
        for family in sorted_families:
            if model_name.startswith(family):
                return family
        return None
    
    def get_color(self, model_name: str) -> Tuple[float, float, float]:
        """Get color for a model based on its family and variant."""
        family = self.get_model_family(model_name)
        if family is None:
            return (0.5, 0.5, 0.5)
        
        variants = self.config.families[family]
        cmap_name = self.config.colormaps.get(family, "viridis")
        cmap = cm.get_cmap(cmap_name)
        
        # Find variant index
        variant_idx = 0
        for idx, variant in enumerate(variants):
            if variant in model_name:
                variant_idx = idx
                break
        
        # Map to color
        color_pos = variant_idx / (len(variants) - 1) if len(variants) > 1 else 0.5
        return cmap(color_pos)
    
    def get_colors_for_models(self, models: List[str]) -> Dict[str, Tuple]:
        """Get color mapping for multiple models."""
        return {model: self.get_color(model) for model in models}


class MetricsPlotter:
    """Handles plotting of metrics."""
    
    def __init__(self, color_manager: ColorManager):
        self.color_manager = color_manager
    
    def plot_single_metric(
        self,
        ax: plt.Axes,
        data: pd.DataFrame,
        metric: str,
        model_colors: Dict[str, Tuple],
        title: Optional[str] = None
    ):
        """Plot a single metric on an axis."""
        # Track if we've added the legend entries for train/val
        legend_added = False
        
        for model in sorted(data['model'].unique()):
            model_data = data[data['model'] == model]
            color = model_colors[model]
            
            # Validation curve (no label, we'll add it separately)
            ax.plot(
                model_data['step'],
                1-model_data[f'{metric}_val'],
                color=color,
                linestyle='-',
                alpha=0.9,
                linewidth=1,
            )
            # Training curve
            ax.plot(
                model_data['step'],
                1-model_data[f'{metric}_train'],
                color=color,
                linestyle='--',
                alpha=0.6,
                linewidth=0.9,
                label=f'{model}'
            )
            
        
        # Add custom legend entries for line styles
        from matplotlib.lines import Line2D
        model_handles, model_labels = ax.get_legend_handles_labels()
        
        # Add line style indicators
        style_handles = [
            Line2D([0], [0], color='gray', linestyle='-', linewidth=1, label='Val'),
            Line2D([0], [0], color='gray', linestyle='--', linewidth=0.9, label='Train'),
        ]
        
        # Combine: models first, then line styles
        all_handles = model_handles + style_handles
        all_labels = model_labels + ['Train', 'Val']
        
        ax.set_xlabel("Epochs", fontsize=11)
        ax.set_ylabel(r"$1-R^2$", fontsize=11)
        ax.set_yscale('log')
        # ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(all_handles, all_labels, bbox_to_anchor=(1.05, 1), 
                 loc='upper left', frameon=True, fontsize=8)
        
        if title:
            ax.set_title(title, fontsize=12, fontweight='bold')
    
    def apply_ylim(self, ax: plt.Axes, ylim: Union[Tuple, Dict], metric: str):
        """Apply y-axis limits to an axis."""
        if ylim is None:
            return
        
        if isinstance(ylim, dict):
            if metric in ylim:
                ax.set_ylim(ylim[metric])
        else:
            ax.set_ylim(ylim)


class SubplotOrganizer:
    """Organizes subplots based on different grouping strategies."""
    
    @staticmethod
    def organize_by_none(data: pd.DataFrame) -> List[Tuple[str, pd.DataFrame]]:
        """All models in one group."""
        return [("All Models", data)]
    
    @staticmethod
    def organize_by_family(data: pd.DataFrame, color_manager: ColorManager) -> List[Tuple[str, pd.DataFrame]]:
        """Group by model family."""
        groups = []
        model_families = {}
        
        for model in data['model'].unique():
            family = color_manager.get_model_family(model)
            if family not in model_families:
                model_families[family] = []
            model_families[family].append(model)
        
        for family, models in sorted(model_families.items()):
            family_data = data[data['model'].isin(models)]
            groups.append((family, family_data))
        
        return groups
    
    @staticmethod
    def organize_by_model(data: pd.DataFrame) -> List[Tuple[str, pd.DataFrame]]:
        """Each model gets its own group."""
        groups = []
        for model in sorted(data['model'].unique()):
            model_data = data[data['model'] == model]
            groups.append((model, model_data))
        return groups
    
    @staticmethod
    def organize_by_custom(data: pd.DataFrame, model_groups: List[List[str]]) -> List[Tuple[str, pd.DataFrame]]:
        """Custom grouping based on provided model lists."""
        groups = []
        for group in model_groups:
            group_data = data[data['model'].isin(group)]
            group_name = " vs ".join(group)
            groups.append((group_name, group_data))
        return groups


class TrainingPlotter:
    """Main class for plotting training metrics."""
    
    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.loader = MetricsLoader()
        self.color_manager = ColorManager(self.config)
        self.plotter = MetricsPlotter(self.color_manager)
        self.organizer = SubplotOrganizer()
    
    def load_runs(self, runs: Dict[str, str], models_to_include: Optional[List[str]] = None) -> pd.DataFrame:
        """Load and combine metrics from multiple runs."""
        all_dfs = []
        
        for run_name, run_path in runs.items():
            print(f"Processing run: {run_name}")
            df = self.loader.fetch_metrics(Path(run_path))
            df['run'] = run_name
            all_dfs.append(df)
        
        if not all_dfs:
            raise ValueError("No runs found.")
        
        combined = pd.concat(all_dfs, ignore_index=True)
        
        if models_to_include:
            combined = combined[combined['model'].isin(models_to_include)]
        
        return combined
    
    def plot(
        self,
        runs: Dict[str, str],
        metrics: Union[str, List[str]] = 'R2',
        ylim: Optional[Union[Tuple, Dict]] = None,
        subplot_by: Union[str, List[List[str]]] = 'none',
        models_to_include: Optional[List[str]] = None,
        figsize: Optional[Tuple[int, int]] = None,
        output_path: str = "training_metrics.png",
        dpi: int = 300
    ):
        """
        Plot training metrics with flexible organization.
        
        Args:
            runs: Dictionary mapping run names to paths
            metrics: Metric(s) to plot (e.g., 'R2' or ['R2', 'loss'])
            ylim: Y-axis limits as tuple (min, max) or dict {metric: (min, max)}
            subplot_by: How to organize: 'none', 'family', 'model', or list of model groups
            models_to_include: Optional list of models to filter
            figsize: Optional figure size override
            output_path: Where to save the figure
            dpi: Resolution for saved figure
        """
        # Normalize metrics to list
        if isinstance(metrics, str):
            metrics = [metrics]
        
        # Load data
        data = self.load_runs(runs, models_to_include)
        model_colors = self.color_manager.get_colors_for_models(data['model'].unique())
        
        # Organize subplots
        groups = self._organize_data(data, subplot_by)
        
        # Create figure
        fig, axes = self._create_figure(len(groups), len(metrics), figsize)
        
        # Plot each group
        for group_idx, (group_name, group_data) in enumerate(groups):
            for metric_idx, metric in enumerate(metrics):
                ax = self._get_axis(axes, group_idx, metric_idx, len(groups), len(metrics))
                
                title = f"{group_name} - {metric}" if len(groups) > 1 else f"{metric} over Training Steps"
                self.plotter.plot_single_metric(ax, group_data, metric, model_colors, title)
                self.plotter.apply_ylim(ax, ylim, metric)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved figure to {output_path}")
        plt.close()
    
    def _organize_data(self, data: pd.DataFrame, subplot_by: Union[str, List]) -> List[Tuple[str, pd.DataFrame]]:
        """Organize data into groups based on subplot_by parameter."""
        if subplot_by == 'none':
            return self.organizer.organize_by_none(data)
        elif subplot_by == 'family':
            return self.organizer.organize_by_family(data, self.color_manager)
        elif subplot_by == 'model':
            return self.organizer.organize_by_model(data)
        elif isinstance(subplot_by, list):
            return self.organizer.organize_by_custom(data, subplot_by)
        else:
            raise ValueError(f"Invalid subplot_by: {subplot_by}")
    
    def _create_figure(self, n_groups: int, n_metrics: int, figsize: Optional[Tuple]) -> Tuple:
        """Create figure with appropriate layout."""
        if figsize is None:
            figsize = (6 * n_metrics, 4 * n_groups)
        
        fig, axes = plt.subplots(n_groups, n_metrics, figsize=figsize, squeeze=False)
        return fig, axes
    
    def _get_axis(self, axes, group_idx: int, metric_idx: int, n_groups: int, n_metrics: int) -> plt.Axes:
        """Get the appropriate axis from the axes array."""
        if n_groups == 1 and n_metrics == 1:
            return axes[0, 0]
        elif n_groups == 1:
            return axes[0, metric_idx]
        elif n_metrics == 1:
            return axes[group_idx, 0]
        else:
            return axes[group_idx, metric_idx]
    
    def plot_convnext_summary(
        self,
        runs: Dict[str, str],
        metric: str = 'R2_val',
        output_path: str = "convnext_summary.png",
        dpi: int = 300
    ):
        """Plot summary comparing ConvNeXt versions across model sizes."""
        data = self.load_runs(runs)
        
        families = [f for f in self.config.families.keys() if 'ConvNeXt' in f]
        sizes = self.config.families.get('ConvNeXt', [])
        x = np.arange(len(sizes))
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for family in families:
            means, stds = [], []
            
            for size in sizes:
                model_name = f"{family}-{size}"
                model_data = data[data['model'] == model_name]
                
                if model_data.empty:
                    means.append(np.nan)
                    stds.append(np.nan)
                    continue
                
                # Get max value for each run
                max_vals = model_data.groupby('run')[metric].max().values
                means.append(np.nanmean(max_vals))
                stds.append(np.nanstd(max_vals))
            
            # Plot with family color
            cmap_name = self.config.colormaps.get(family, 'viridis')
            color = cm.get_cmap(cmap_name)(0.6)
            
            ax.plot(x, means, marker='o', label=family, color=color, linewidth=2)
            ax.fill_between(
                x,
                np.array(means) - np.array(stds),
                np.array(means) + np.array(stds),
                color=color,
                alpha=0.2
            )
        
        ax.set_xticks(x)
        ax.set_xticklabels(sizes)
        ax.set_xlabel('Model Size', fontsize=12)
        ax.set_ylabel(metric, fontsize=12)
        ax.set_title(f'ConvNeXt max validation {metric} by size and version', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved summary to {output_path}")
        plt.close()


# Convenience function
def plot_training_metrics(
    runs: Dict[str, str],
    metrics: Union[str, List[str]] = 'R2',
    ylim: Optional[Union[Tuple, Dict]] = None,
    subplot_by: Union[str, List[List[str]]] = 'none',
    models_to_include: Optional[List[str]] = None,
    output_path: str = "training_metrics.png"
):
    """Convenience function for plotting training metrics."""
    plotter = TrainingPlotter()
    plotter.plot(runs, metrics, ylim, subplot_by, models_to_include, output_path=output_path)


# Usage example
if __name__ == "__main__":
    # Initialize plotter
    plotter = TrainingPlotter()
    
    # Define runs
    runs = {
        # "all_models_permeability": "results/all_models_permeability",
        "zero_pecle_all_models": "results/zero_pecle_all_models",
    }
    
    # Example 1: Simple plot with all models
    # plotter.plot(runs, metrics='R2')
    
    # Example 2: Group by family
    # plotter.plot(runs, metrics='R2',
    #             #   ylim=(0.8, 1.001), 
    #               subplot_by='family', output_path="by_family.pdf")
    
    # # Example 3: Custom grouping
    plotter.plot(
        runs,
        metrics='R2',
        # ylim=(0.995, 1.001),
        subplot_by=[
            ["ConvNeXt-Atto","ConvNeXt-V2-Atto","ConvNeXt-RMS-Atto"],
            ["ConvNeXt-Femto","ConvNeXt-V2-Femto","ConvNeXt-RMS-Femto"],
        ],
        output_path="custom_comparison.png"
    )
    
    # Example 4: ConvNeXt summary
    # plotter.plot_convnext_summary(runs, metric='R2_val', output_path="convnext_pecle_best_r2.pdf")