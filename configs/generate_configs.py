"""
Training configuration generator for PyTorch experiments.
Generates YAML configs and optional SLURM batch scripts.
Uses a JSON registry for model configurations.
"""
import yaml
import json
import argparse
import os
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict


"""
Run plans:
    - The first run is a run over different batch sizes for all models.
    - The second run is a run over different learning rates for all models.
    - The third run is a run over different weight decays for all models.
    - The fourth run is a run over different number of data samples for all models.
    - The fifth run is a run over different number of epochs for all models.
    - The sixth run is a run over different decay schedules for all models.
"""


# ============================================================================
# CONFIGURATION PRESETS - Edit these to change experiment settings
# ============================================================================

# Model presets for different experiment scales
MODEL_PRESETS = {
    "quick_test": [
        "ConvNeXt-Atto",
    ],
    "small_sweep": [
        "ViT-T16",
        "ViT-S16",
        "ConvNeXt-Atto",
        "ConvNeXt-Femto",
        "ConvNeXt-Pico",
        "ResNet-18",
        "ResNet-34",
    ],
    "full_vit": [
        "ViT-T16",
        "ViT-S16",
        "ViT-B16",
        "ViT-L16",
    ],
    "full_convnext": [
        "ConvNeXt-Atto",
        "ConvNeXt-Femto",
        "ConvNeXt-Pico",
        "ConvNeXt-Nano",
        "ConvNeXt-Tiny",
        "ConvNeXt-Small",
        "ConvNeXt-Base",
    ],
    "full_resnet": [
        "ResNet-18",
        "ResNet-34",
        "ResNet-50",
        "ResNet-101",
        "ResNet-152",
    ],
    "all_models": [
        # All ViT
        "ViT-T16", "ViT-S16", "ViT-B16", "ViT-L16",
        # All ConvNeXt
        "ConvNeXt-Atto", "ConvNeXt-Femto", "ConvNeXt-Pico", "ConvNeXt-Nano",
        "ConvNeXt-Tiny", "ConvNeXt-Small", "ConvNeXt-Base", "ConvNeXt-Large",
        # All ConvNeXt V2
        "ConvNeXt-V2-Atto", "ConvNeXt-V2-Femto", "ConvNeXt-V2-Pico", "ConvNeXt-V2-Nano",
        "ConvNeXt-V2-Tiny", "ConvNeXt-V2-Small", "ConvNeXt-V2-Base", "ConvNeXt-V2-Large",
        # All ConvNeXt RMS
        "ConvNeXt-RMS-Atto", "ConvNeXt-RMS-Femto", "ConvNeXt-RMS-Pico", "ConvNeXt-RMS-Nano",
        "ConvNeXt-RMS-Tiny", "ConvNeXt-RMS-Small", "ConvNeXt-RMS-Base", "ConvNeXt-RMS-Large",
        # All ResNet
        "ResNet-18", "ResNet-34", "ResNet-50", "ResNet-101", "ResNet-152",
    ],
}

# Default model preset to use
DEFAULT_MODEL_PRESET = "quick_test"

# Hyperparameter sweep configurations
# Each list defines the values to sweep over for that hyperparameter
HYPERPARAM_SWEEPS = {
    "learning_rate": {
        "single": [8e-4],  # Default single value
        "sweep": [1e-5, 5e-5, 1e-4, 5e-4],
    },
    "batch_size": {
        "single": [128],
        "sweep": [64, 128, 256, 512],
    },
    "weight_decay": {
        "single": [1e-1],
        "sweep": [5e-2, 1e-1, 0.3],
    },
    "num_training_samples": {
        "single": [None],  # None means use all available data
        "scaling": [100, 500, 1000, 5000, 10000, None],
    },
    "num_epochs": {
        "single": [30],
        "sweep": [100,300,500,700,1000],    
    },
    "decay": {
        "single": ["cosine"],
        "sweep": ["cosine", "linear", "exponential", "step"],
        "common": ["cosine", "linear"],
    },
    'pe_encoder': {
        'single': ['log'],
        'sweep': ['straight', 'log', 'vector'],
    },
    'Pe': {
        'single': [0],
        'sweep': [0,1,2,3,4],
    },
}

# Sweep presets - predefined combinations of sweeps
SWEEP_PRESETS = {
    "none": {
        # No sweeps, use single values for everything
        "learning_rate": "single",
        "batch_size": "single",
        "weight_decay": "single",
        "num_training_samples": "single",
        "num_epochs": "single",
        "decay": "single",
        'pe_encoder': 'single',
        'Pe': 'single',
    },
    "pe_encoder_sweep": {
        # Sweep only learning rate
        "learning_rate": "single",
        "batch_size": "single",
        "weight_decay": "sweep",
        "num_training_samples": "single",
        "num_epochs": "single",
        "decay": "single",
        'pe_encoder': 'sweep',
        'Pe': 'single',
    },
    "Pe_sweep": {
        # Sweep only learning rate
        "learning_rate": "single",
        "batch_size": "single",
        "weight_decay": "single",
        # "num_training_samples": "single",
        "num_epochs": "single",
        "decay": "single",
        'pe_encoder': 'single',
        "Pe": "sweep",
    },
    "lr_sweep": {
        # Sweep only learning rate
        "learning_rate": "sweep",
        "batch_size": "single",
        "weight_decay": "single",
        "num_training_samples": "single",
        "num_epochs": "single",
        "decay": "single",
    },
    "bs_sweep": {
        # Sweep only batch size
        "learning_rate": "single",
        "batch_size": "sweep",
        "weight_decay": "single",
        "num_training_samples": "single",
        "num_epochs": "single",
        "decay": "single",
    },
    "data_scaling": {
        # Sweep number of data samples (data scaling experiment)
        "learning_rate": "single",
        "batch_size": "single",
        "weight_decay": "single",
        "num_training_samples": "scaling",
        "num_epochs": "single",
        "decay": "single",
    },
    "lr_bs_sweep": {
        # Sweep learning rate and batch size together
        "learning_rate": "sweep",
        "batch_size": "sweep",
        "weight_decay": "single",
        "num_training_samples": "single",
        "num_epochs": "single",
        "decay": "single",
    },
    "lr_bs_wd_sweep": {
        # Sweep learning rate, batch size and weight decay together
        "learning_rate": "sweep",
        "batch_size": "sweep",
        "weight_decay": "sweep",
        "num_training_samples": "single",
        "num_epochs": "single",
        "decay": "single",
    },
    "optimizer_sweep": {
        # Sweep LR, weight decay, and decay schedule
        "learning_rate": "fine",
        "batch_size": "single",
        "weight_decay": "sweep",
        "num_training_samples": "single",
        "num_epochs": "single",
        "decay": "common",
    },
    "full_sweep": {
        # Sweep everything (WARNING: will generate many configs!)
        "learning_rate": "coarse",
        "batch_size": "small",
        "weight_decay": "light",
        "num_training_samples": "small",
        "num_epochs": "single",
        "decay": "common",
    },
}

DEFAULT_SWEEP_PRESET = "none"

# Base training configurations by task (used as defaults)
TASK_CONFIGS = {
    "permeability": {
        "learning_rate": 5e-5,
        "weight_decay": 1e-1,
        "batch_size": 128,
        "num_epochs": 100,
        "decay": "cosine",
        "warmup_steps": 125*2, # steps per epoch * warmup epochs
        "num_training_samples": None,  # None = use all data
        "num_validation_samples": None,
        "prefetch_factor": 4,
        "pin_memory": True,
    },
    "dispersion": {
        "learning_rate": 5e-3,
        "weight_decay": 1e-5,
        "batch_size": 64,
        "num_epochs": 100,
        "decay": "cosine",
        "warmup_steps": 5*16000/64, # steps per epoch * warmup epochs
        "num_training_samples": None,
        "num_validation_samples": None,
        "prefetch_factor": 4,
        "pin_memory": True,
        "pe": {
            "pe_encoder": False,
            "pe": 4,
            "include_direction": False,
        },
        "pe_encoder": None,
        "Pe": 4,
    },
}

# Device-specific configurations
DEVICE_CONFIGS = {
    "gpu": {
        "slurm": {
            "partition": "normal",
            "cpus_per_task": 2,
            "gres": "gpu:1",
            "time": None,
            "mem": None,
        },
    },
    "cpu": {
        "slurm": {
            "partition": "normal",
            "cpus_per_task": 4,
            "gres": None,
            "time": "01:00:00",
            "mem": "16G",
        },
        "training_overrides": {
            "batch_size": 8,
            "num_epochs": 1000,
            "warmup_steps": int(0.1*10*32*10/8),
            "weight_decay": 1e-3,
            "decay": "cosine",
            "learning_rate": 8e-4,
            "num_training_samples": 32,
            "num_validation_samples": 4,
            "num_test_samples": 4,
            "pe_encoder": None,
            "prefetch_factor": None,
            "pin_memory": False,
            "pin_memory_device": "",
            "use_amp": False,
        },
    },
}

# Experiment naming template
EXP_NAME_TEMPLATE = "{task}_{device}_run"

# Output directories
OUTPUT_BASE_DIR = "./experiments"
RESULTS_BASE_DIR = "./results"

# Main training script name
MAIN_SCRIPT = "run_model_training.py"


# ============================================================================
# Core classes (unchanged from original)
# ============================================================================
import os
@dataclass
class SlurmConfig:
    """SLURM job configuration."""
    partition: str = "normal"
    ntasks: int = 1
    cpus_per_task: int = 2
    gres: str = "gpu:1"
    time: str = None
    mem: str = None
    slurm_out_dir: str = "slurm_outputs"
    
    
    def to_header(self, job_name: str) -> str:
        """Generate SLURM header."""
        # os.makedirs(os.path.join(self.slurm_out_dir, job_name), exist_ok=True)
        lines = [
            "#!/bin/bash",
            f"#SBATCH --output={self.slurm_out_dir}/%x_%j.out",
            # f"#SBATCH --error={self.slurm_out_dir}/%x_%j.err",
            f"#SBATCH --job-name={job_name}",
            f"#SBATCH --partition={self.partition}",
            f"#SBATCH --ntasks={self.ntasks}",
            f"#SBATCH --cpus-per-task={self.cpus_per_task}",
        ]
        if self.gres:
            lines.append(f"#SBATCH --gres={self.gres}")
        if self.time:
            lines.append(f"#SBATCH --time={self.time}")
        if self.mem:
            lines.append(f"#SBATCH --mem={self.mem}")
        return "\n".join(lines)


class ModelRegistry:
    """Registry for available model configurations."""
    
    DEFAULT_MODELS = {
        # Vision Transformers
        "ViT-T16": {
            "type": "vit",
            "size": "T16",
            "clip_grad": True,
            "description": "Vision Transformer Tiny with 16x16 patches"
        },
        "ViT-S16": {
            "type": "vit",
            "size": "S16",
            "clip_grad": True,
            "description": "Vision Transformer Small with 16x16 patches"
        },
        "ViT-B16": {
            "type": "vit",
            "size": "B16",
            "clip_grad": True,
            "description": "Vision Transformer Base with 16x16 patches"
        },
        "ViT-L16": {
            "type": "vit",
            "size": "L16",
            "clip_grad": True,
            "description": "Vision Transformer Large with 16x16 patches"
        },
        
        # ConvNeXt variants
        "ConvNeXt-Atto": {
            "type": "convnext",
            "size": "atto",
            "clip_grad": True,
            "description": "ConvNeXt Atto (smallest)",
            # "training_overrides": {"batch_size": 1024}
        },
        "ConvNeXt-Femto": {
            "type": "convnext",
            "size": "femto",
            "clip_grad": True,
            "description": "ConvNeXt Femto",
            # "training_overrides": {"batch_size": 1024}
        },
        "ConvNeXt-Pico": {
            "type": "convnext",
            "size": "pico",
            "clip_grad": True,
            "description": "ConvNeXt Pico",
            # "training_overrides": {"batch_size": 512}
        },
        "ConvNeXt-Nano": {
            "type": "convnext",
            "size": "nano",
            "clip_grad": True,
            "description": "ConvNeXt Nano"
        },
        "ConvNeXt-Tiny": {
            "type": "convnext",
            "size": "tiny",
            "clip_grad": True,
            "description": "ConvNeXt Tiny"
        },
        "ConvNeXt-Small": {
            "type": "convnext",
            "size": "small",
            "clip_grad": True,
            "description": "ConvNeXt Small"
        },
        "ConvNeXt-Base": {
            "type": "convnext",
            "size": "base",
            "clip_grad": True,
            "description": "ConvNeXt Base"
        },
        "ConvNeXt-Large": {
            "type": "convnext",
            "size": "large",
            "clip_grad": True,
            "description": "ConvNeXt Large"
        },
        
        # ConvNeXt V2 variants
        "ConvNeXt-V2-Atto": {
            "type": "convnext",
            "size": "atto",
            "version": "v2",
            "clip_grad": True,
            "description": "ConvNeXt V2 Atto"
        },
        "ConvNeXt-V2-Femto": {
            "type": "convnext",
            "size": "femto",
            "version": "v2",
            "clip_grad": True,
            "description": "ConvNeXt V2 Femto"
        },
        "ConvNeXt-V2-Pico": {
            "type": "convnext",
            "size": "pico",
            "version": "v2",
            "clip_grad": True,
            "description": "ConvNeXt V2 Pico"
        },
        "ConvNeXt-V2-Nano": {
            "type": "convnext",
            "size": "nano",
            "version": "v2",
            "clip_grad": True,
            "description": "ConvNeXt V2 Nano"
        },
        "ConvNeXt-V2-Tiny": {
            "type": "convnext",
            "size": "tiny",
            "version": "v2",
            "clip_grad": True,
            "description": "ConvNeXt V2 Tiny"
        },
        "ConvNeXt-V2-Small": {
            "type": "convnext",
            "size": "small",
            "version": "v2",
            "clip_grad": True,
            "description": "ConvNeXt V2 Small"
        },
        "ConvNeXt-V2-Base": {
            "type": "convnext",
            "size": "base",
            "version": "v2",
            "clip_grad": True,
            "description": "ConvNeXt V2 Base"
        },
        "ConvNeXt-V2-Large": {
            "type": "convnext",
            "size": "large",
            "version": "v2",
            "clip_grad": True,
            "description": "ConvNeXt V2 Large"
        },

        # ConvNeXt RMS variants
        "ConvNeXt-RMS-Atto": {
            "type": "convnext",
            "size": "atto",
            "version": "rms",
            "clip_grad": True,
            "description": "ConvNeXt RMS Atto"
        },
        "ConvNeXt-RMS-Femto": {
            "type": "convnext",
            "size": "femto",
            "version": "rms",
            "clip_grad": True,
            "description": "ConvNeXt RMS Femto"
        },
        "ConvNeXt-RMS-Pico": {
            "type": "convnext",
            "size": "pico",
            "version": "rms",
            "clip_grad": True,
            "description": "ConvNeXt RMS Pico"
        },
        "ConvNeXt-RMS-Nano": {
            "type": "convnext",
            "size": "nano",
            "version": "rms",
            "clip_grad": True,
            "description": "ConvNeXt RMS Nano"
        },
        "ConvNeXt-RMS-Tiny": {
            "type": "convnext",
            "size": "tiny",
            "version": "rms",
            "clip_grad": True,
            "description": "ConvNeXt RMS Tiny"
        },
        "ConvNeXt-RMS-Small": {
            "type": "convnext",
            "size": "small",
            "version": "rms",
            "clip_grad": True,
            "description": "ConvNeXt RMS Small"
        },
        "ConvNeXt-RMS-Base": {
            "type": "convnext",
            "size": "base",
            "version": "rms",
            "clip_grad": True,
            "description": "ConvNeXt RMS Base"
        },
        "ConvNeXt-RMS-Large": {
            "type": "convnext",
            "size": "large",
            "version": "rms",
            "clip_grad": True,
            "description": "ConvNeXt RMS Large"
        },
        
        # ResNet variants
        "ResNet-18": {
            "type": "resnet",
            "size": "18",
            "clip_grad": True,
            "description": "ResNet 18 layers",
            # "training_overrides": {"batch_size": 1024}
        },
        "ResNet-34": {
            "type": "resnet",
            "size": "34",
            "clip_grad": True,
            "description": "ResNet 34 layers",
            # "training_overrides": {"batch_size": 1024}
        },
        "ResNet-50": {
            "type": "resnet",
            "size": "50",
            "clip_grad": True,
            "description": "ResNet 50 layers"
        },
        "ResNet-101": {
            "type": "resnet",
            "size": "101",
            "clip_grad": True,
            "description": "ResNet 101 layers"
        },
        "ResNet-152": {
            "type": "resnet",
            "size": "152",
            "clip_grad": True,
            "description": "ResNet 152 layers"
        },
    }
    
    def __init__(self, registry_path: Optional[Path] = None):
        """Initialize registry from file or use defaults."""
        if registry_path and registry_path.exists():
            with open(registry_path, 'r') as f:
                self.models = json.load(f)
            print(f"Loaded model registry from {registry_path}")
        else:
            self.models = self.DEFAULT_MODELS.copy()
    
    def save(self, path: Path):
        """Save registry to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.models, f, indent=2)
        print(f"Saved model registry to {path}")
    
    def get(self, name: str) -> Dict[str, Any]:
        """Get model config by name."""
        if name not in self.models:
            raise ValueError(
                f"Model '{name}' not found in registry. "
                f"Available models: {', '.join(self.models.keys())}"
            )
        return self.models[name].copy()
    
    def list_models(self, model_type: Optional[str] = None) -> List[str]:
        """List available models, optionally filtered by type."""
        if model_type:
            return [
                name for name, cfg in self.models.items() 
                if cfg.get("type") == model_type
            ]
        return list(self.models.keys())
    
    def print_catalog(self):
        """Print formatted catalog of available models."""
        print("\n" + "="*70)
        print("AVAILABLE MODELS")
        print("="*70)
        
        by_type = {}
        for name, cfg in self.models.items():
            model_type = cfg.get("type", "unknown")
            if model_type not in by_type:
                by_type[model_type] = []
            by_type[model_type].append((name, cfg))
        
        for model_type in sorted(by_type.keys()):
            print(f"\n{model_type.upper()}:")
            print("-" * 70)
            for name, cfg in sorted(by_type[model_type]):
                desc = cfg.get("description", "")
                clip = "✓" if cfg.get("clip_grad") else "✗"
                print(f"  {name:30s} | Clip: {clip} | {desc}")
        
        print("\n" + "="*70 + "\n")


class ConfigGenerator:
    """Generate training configs and SLURM scripts."""
    
    def __init__(
        self,
        exp_name: str,
        output_dir: Path,
        base_config: Dict[str, Any],
        model_registry: ModelRegistry,
        slurm_config: SlurmConfig = None,
        is_cpu: bool = False,
        sweep_configs: Dict[str, List[Any]] = None,
    ):
        self.exp_name = exp_name
        self.output_dir = Path(output_dir)
        self.base_config = base_config
        self.registry = model_registry
        self.slurm_config = slurm_config or SlurmConfig()
        self.is_cpu = is_cpu
        self.sweep_configs = sweep_configs or {}
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.yaml_dir = self.output_dir / exp_name
        self.yaml_dir.mkdir(exist_ok=True)
    
    def _generate_sweep_combinations(self) -> List[Dict[str, Any]]:
        """Generate all combinations of hyperparameters to sweep."""
        if not self.sweep_configs:
            return [{}]
        
        import itertools
        
        # Get all parameter names and their values
        param_names = list(self.sweep_configs.keys())
        param_values = [self.sweep_configs[name] for name in param_names]
        
        # Generate all combinations
        combinations = []
        for values in itertools.product(*param_values):
            combo = dict(zip(param_names, values))
            combinations.append(combo)
        
        return combinations
    
    def _format_param_value(self, value: Any) -> str:
        """Format parameter value for use in filenames."""
        if value is None:
            return "all"
        elif isinstance(value, float):
            return f"{value:.0e}".replace("-", "m").replace("+", "p")
        else:
            return str(value)
    
    def _build_experiment(self, model_name: str, sweep_params: Dict[str, Any] = None) -> tuple[Dict[str, Any], str]:
        """Build experiment configuration for a model with optional sweep parameters.
        
        Returns:
            tuple: (experiment_config, unique_suffix)
        """
        model_cfg = self.registry.get(model_name)
        
        exp = {
            "model": {
                "type": model_cfg["type"],
                "name": model_name,
                "size": model_cfg["size"],
                "in_channels": 1,
                "pretrained_path": None,
            },
        }
        
        if "version" in model_cfg:
            exp["model"]["version"] = model_cfg["version"]
        
        # Start with base config
        exp.update(self.base_config)
        # Apply model-specific training overrides if present in the registry.
        # Support `training_overrides` dict which may include a numeric
        # `batch_multiplier` (applied against the already-computed
        # `exp['batch_size']`). An explicit `batch_size` in overrides or at
        # model top-level will overwrite the value.
        overrides = model_cfg.get("training_overrides") or {}
        if isinstance(overrides, dict):
            # Handle multiplier first so explicit batch_size can still override
            mult = overrides.pop("batch_multiplier", None)
            if mult is not None:
                try:
                    exp["batch_size"] = int(exp.get("batch_size", 1) * float(mult))
                except Exception:
                    pass
            # Apply remaining explicit overrides
            exp.update(overrides)

        # Backwards-compatible top-level keys on the model entry
        if "batch_multiplier" in model_cfg and "training_overrides" not in model_cfg:
            try:
                exp["batch_size"] = int(exp.get("batch_size", 1) * float(model_cfg["batch_multiplier"]))
            except Exception:
                pass
        elif "batch_size" in model_cfg:
            exp["batch_size"] = model_cfg["batch_size"]
        exp["clip_grad"] = model_cfg.get("clip_grad", True)
        
        # Apply sweep parameters and build suffix
        suffix_parts = []
        if sweep_params:
            for param_name, param_value in sweep_params.items():
                exp[param_name] = param_value
                # Add to suffix for unique naming
                short_name = {
                    "learning_rate": "lr",
                    "batch_size": "bs",
                    "weight_decay": "wd",
                    "num_training_samples": "n",
                    "num_validation_samples": "nv",
                    "num_epochs": "ep",
                    "decay": "decay",
                    "pe_encoder": "pe_encoder",
                    "Pe": "Pe",
                }.get(param_name, param_name[:4])
                suffix_parts.append(f"{short_name}{self._format_param_value(param_value)}")
        
        suffix = "_".join(suffix_parts) if suffix_parts else ""
        
        if self.is_cpu:
            exp["device"] = "cpu"
        
        return exp, suffix
    
    def generate_herbie_mode(self, model_names: List[str], main_script: str = MAIN_SCRIPT, conda_env: Optional[str] = None) -> Path:
        """Generate single YAML with all experiments (including sweeps).

        Also create a simple launcher script that starts the job inside a
        detached `screen` session (useful for mocking SLURM on a single node).
        """
        experiments = []
        
        # Generate sweep combinations
        sweep_combos = self._generate_sweep_combinations()
        
        # Generate experiment for each model x sweep combination
        for model_name in model_names:
            for sweep_params in sweep_combos:
                exp, _ = self._build_experiment(model_name, sweep_params)
                experiments.append(exp)
        
        mode_suffix = "_cpu" if self.is_cpu else ""
        yaml_path = self.yaml_dir / f"{self.exp_name}_all{mode_suffix}.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump({"experiments": experiments}, f, default_flow_style=False)
        
        print(f"✓ Generated herbie mode YAML with {len(experiments)} experiments")
        print(f"  ({len(model_names)} models × {len(sweep_combos)} sweep combinations)")
        if self.is_cpu:
            print(f"  [CPU MODE] - device set to 'cpu'")
        print(f"  {yaml_path}")
        # Create a simple launcher that runs the main script inside a detached
        # `screen` session so users can mimic SLURM submission locally.
        launcher_path = self.output_dir / f"run_herbie_{self.exp_name}.sh"
        session_name = f"{self.exp_name}_herbie"
        # Build launcher lines differently for CPU vs GPU to avoid surprising
        # behavior when users expect SLURM to allocate devices.
        if self.is_cpu:
            launcher_lines = [
                "#!/bin/bash",
                f"# Launch herbie-mode experiment in a detached screen session: {session_name}",
                "# Requires `screen` to be installed on the system.",
                "",
                # Export CONDA_ENV if requested
                *( [f"CONDA_ENV={conda_env}", "export CONDA_ENV", ""] if conda_env else [] ),
                "",
                f"echo 'Starting screen session: {session_name}'",
                # Use bash -lc so that quotes and envs behave similar to interactive runs
                f"screen -dmS {session_name} bash -lc '" + \
                ("if [ -n \"$CONDA_ENV\" ]; then if command -v conda >/dev/null 2>&1; then eval \"$(conda shell.bash hook)\" && conda activate \"$CONDA_ENV\"; elif [ -f \"$HOME/miniconda3/etc/profile.d/conda.sh\" ]; then . \"$HOME/miniconda3/etc/profile.d/conda.sh\" && conda activate \"$CONDA_ENV\"; else echo \"conda not found; continuing\"; fi; fi; ")
                + f"python3 {main_script} --config \"{yaml_path}\"; exec bash'",
                "",
                "echo 'To attach: screen -r '" + session_name
            ]
        else:
            # For GPU runs, don't assume SLURM will allocate GPUs. Provide a
            # conservative default and a simple nvidia-smi check. The script
            # prefers an existing CUDA_VISIBLE_DEVICES environment variable,
            # otherwise it defaults to 0.
            launcher_lines = [
                "#!/bin/bash",
                f"# Launch herbie-mode experiment in a detached screen session: {session_name}",
                "# Requires `screen` and `nvidia-smi` (optional) to be installed.",
                "",
                # Export CONDA_ENV if requested
                *( [f"CONDA_ENV={conda_env}", "export CONDA_ENV", ""] if conda_env else [] ),
                "",
                "# If CUDA_VISIBLE_DEVICES is not set, default to GPU 0 (change if needed)",
                ": ${CUDA_VISIBLE_DEVICES:=0}",
                "export CUDA_VISIBLE_DEVICES",
                "",
                "# Quick sanity check (non-fatal): print selected GPU(s)",
                "if command -v nvidia-smi >/dev/null 2>&1; then",
                "  echo 'nvidia-smi available; showing selected devices:'",
                "  nvidia-smi --query-gpu=index,name,uuid,memory.total --format=csv -i $CUDA_VISIBLE_DEVICES || true",
                "else",
                "  echo 'nvidia-smi not found; ensure CUDA drivers are available on this machine'",
                "fi",
                "",
                f"echo 'Starting screen session: {session_name} (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)'",
                # Use bash -lc so that the exported env is visible inside the session
                f"screen -dmS {session_name} bash -lc '" + \
                ("if [ -n \"$CONDA_ENV\" ]; then if command -v conda >/dev/null 2>&1; then eval \"$(conda shell.bash hook)\" && conda activate \"$CONDA_ENV\"; elif [ -f \"$HOME/miniconda3/etc/profile.d/conda.sh\" ]; then . \"$HOME/miniconda3/etc/profile.d/conda.sh\" && conda activate \"$CONDA_ENV\"; else echo \"conda not found; continuing\"; fi; fi; ")
                + f"python3 {main_script} --config \"{yaml_path}\"; exec bash'",
                "",
                "echo 'To attach: screen -r '" + session_name
            ]

        with open(launcher_path, "w") as f:
            f.write("\n".join(launcher_lines) + "\n")
        launcher_path.chmod(0o755)

        print(f"\n✓ Launcher script: {launcher_path}")
        print(f"  Start with: bash {launcher_path}")

        return yaml_path
    
    def generate_individual_mode(
        self,
        model_names: List[str],
        main_script: str = MAIN_SCRIPT
    ) -> tuple[List[Path], Path]:
        """Generate individual YAMLs and execution scripts (including sweeps)."""
        yaml_paths = []
        script_paths = []
        
        use_slurm = not self.is_cpu
        sweep_combos = self._generate_sweep_combinations()
        
        for model_name in model_names:
            for sweep_params in sweep_combos:
                exp, suffix = self._build_experiment(model_name, sweep_params)
                
                safe_name = model_name.replace('/', '-').replace(' ', '_')
                job_name = f"{safe_name}_{self.exp_name}"
                if suffix:
                    job_name = f"{job_name}_{suffix}"
                
                # Generate YAML
                yaml_path = self.yaml_dir / f"{job_name}.yaml"
                
                with open(yaml_path, "w") as f:
                    yaml.dump({"experiments": [exp]}, f, default_flow_style=False)
                yaml_paths.append(yaml_path)
                
                # Generate script
                if use_slurm:
                    script_content = f"""{self.slurm_config.to_header(job_name)}

python3 {main_script} --config "{yaml_path}"
"""
                else:
                    script_content = f"""#!/bin/bash
# CPU test for {job_name}

python3 {main_script} --config "{yaml_path}"
"""
                
                script_path = self.yaml_dir / f"{job_name}.sh"
                with open(script_path, "w") as f:
                    f.write(script_content)
                script_path.chmod(0o755)
                script_paths.append(script_path)
        
        print(f"✓ Generated {len(yaml_paths)} individual configs")
        print(f"  ({len(model_names)} models × {len(sweep_combos)} sweep combinations)")
        
        # Create launcher script
        launcher_name = f"{'submit' if use_slurm else 'run'}_all_{self.exp_name}.sh"
        launcher_path = self.output_dir / launcher_name
        
        with open(launcher_path, "w") as f:
            f.write("#!/bin/bash\n")
            f.write(f"# {'Submit' if use_slurm else 'Run'} all jobs for experiment: {self.exp_name}\n")
            f.write(f"# Total jobs: {len(script_paths)}\n\n")
            for script in script_paths:
                cmd = "sbatch" if use_slurm else "bash"
                f.write(f"{cmd} {script}\n")
        
        launcher_path.chmod(0o755)
        print(f"\n✓ Launcher script: {launcher_path}")
        print(f"  Run with: bash {launcher_path}")
        
        return script_paths, launcher_path

    def generate_bigfacet_mode(self, model_names: List[str], main_script: str = MAIN_SCRIPT, conda_env: Optional[str] = None) -> Path:
        """Generate individual YAMLs and a launcher that distributes jobs across GPUs.

        The launcher will detect the number of GPUs (via `nvidia-smi` if
        available) and start each job in a detached `screen` session, assigning
        GPUs in round-robin by setting `CUDA_VISIBLE_DEVICES` per session.
        """
        yaml_paths = []
        script_paths = []

        sweep_combos = self._generate_sweep_combinations()

        for model_name in model_names:
            for sweep_params in sweep_combos:
                exp, suffix = self._build_experiment(model_name, sweep_params)

                safe_name = model_name.replace('/', '-').replace(' ', '_')
                job_name = f"{safe_name}_{self.exp_name}"
                if suffix:
                    job_name = f"{job_name}_{suffix}"

                # Generate YAML per job
                yaml_path = self.yaml_dir / f"{job_name}.yaml"
                with open(yaml_path, "w") as f:
                    yaml.dump({"experiments": [exp]}, f, default_flow_style=False)
                yaml_paths.append(yaml_path)

        # Detect GPU count if possible
        ngpus = 0
        if shutil.which('nvidia-smi'):
            try:
                import subprocess
                out = subprocess.check_output(['nvidia-smi', '--query-gpu=index', '--format=csv,noheader'])
                ngpus = len([l for l in out.decode().splitlines() if l.strip()])
            except Exception:
                ngpus = 0

        if ngpus <= 0:
            ngpus = 1

        # Create per-job runner scripts (embed fixed GPU assignment) to avoid
        # complex quoting when starting screen sessions. Then create a simple
        # launcher that starts each runner in its own detached screen session.
        launcher_path = self.output_dir / f"bigfacet_submit_{self.exp_name}.sh"
        session_prefix = f"{self.exp_name}_bf"

        job_scripts: List[Path] = []
        logs_dir = self.output_dir / "slurm_out"
        logs_dir.mkdir(parents=True, exist_ok=True)

        for idx, p in enumerate(yaml_paths):
            gpu = idx % ngpus
            job_name = p.stem
            job_script = self.yaml_dir / f"{job_name}_run.sh"
            script_lines = ["#!/bin/bash"]
            if conda_env:
                script_lines += [f"CONDA_ENV={conda_env}", "export CONDA_ENV", "",
                                 "# try to activate conda env if available"]
            script_lines += [f"export CUDA_VISIBLE_DEVICES={gpu}", "",
                             "# optional conda activation (non-fatal)"]
            if conda_env:
                script_lines += [
                    "if command -v conda >/dev/null 2>&1; then",
                    "  eval \"$(conda shell.bash hook)\" && conda activate \"$CONDA_ENV\"",
                    "elif [ -f \"$HOME/miniconda3/etc/profile.d/conda.sh\" ]; then",
                    "  . \"$HOME/miniconda3/etc/profile.d/conda.sh\" && conda activate \"$CONDA_ENV\"",
                    "else",
                    "  echo 'conda not found; continuing without activation'",
                    "fi",
                    ""
                ]
            out_file = logs_dir / f"{job_name}.out"
            script_lines += [f"echo 'Starting job: python3 {main_script} --config \"{p}\" (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)' >> \"{out_file}\" 2>&1",
                             f"python3 {main_script} --config \"{p}\" >> \"{out_file}\" 2>&1",
                             "exec bash"]

            with open(job_script, "w") as f:
                f.write("\n".join(script_lines) + "\n")
            job_script.chmod(0o755)
            job_scripts.append(job_script)

        # Build launcher that starts each job script in a detached screen session
        launcher_lines = [
            "#!/bin/bash",
            f"# BigFacet launcher for {self.exp_name}",
            f"# Detected GPUs: {ngpus}",
            "# Requires `screen` to be installed.",
            "",
        ]

        for idx, js in enumerate(job_scripts):
            session_name = f"{session_prefix}_{idx}"
            launcher_lines += [
                f"echo 'Starting session: {session_name} -> {js}'",
                f"screen -dmS {session_name} bash -c '\"{js}\"'",
                "",
            ]

        with open(launcher_path, "w") as f:
            f.write("\n".join(launcher_lines) + "\n")
        launcher_path.chmod(0o755)

        print(f"\n✓ Generated {len(yaml_paths)} job YAMLs, {len(job_scripts)} runner scripts and BigFacet launcher: {launcher_path}")
        print(f"  Detected GPUs: {ngpus}")

        return launcher_path


# ============================================================================
# Main entry point
# ============================================================================

def print_presets():
    """Print available model presets."""
    print("\n" + "="*70)
    print("AVAILABLE MODEL PRESETS")
    print("="*70)
    for preset_name, models in MODEL_PRESETS.items():
        print(f"\n{preset_name}:")
        print(f"  Models: {', '.join(models)}")
        print(f"  Count: {len(models)}")
    print("\n" + "="*70 + "\n")


def print_sweep_presets():
    """Print available sweep presets with details."""
    print("\n" + "="*70)
    print("AVAILABLE SWEEP PRESETS")
    print("="*70)
    
    for preset_name, preset_config in SWEEP_PRESETS.items():
        print(f"\n{preset_name}:")
        total_combinations = 1
        for param, sweep_type in preset_config.items():
            values = HYPERPARAM_SWEEPS[param][sweep_type]
            n_values = len(values)
            total_combinations *= n_values
            print(f"  {param:20s}: {sweep_type:15s} ({n_values} values)")
        print(f"  → Total combinations: {total_combinations}")
    
    print("\n" + "="*70 + "\n")


def print_hyperparam_sweeps():
    """Print all available hyperparameter sweep configurations."""
    print("\n" + "="*70)
    print("HYPERPARAMETER SWEEP OPTIONS")
    print("="*70)
    
    for param_name, sweep_types in HYPERPARAM_SWEEPS.items():
        print(f"\n{param_name}:")
        for sweep_name, values in sweep_types.items():
            print(f"  {sweep_name:15s}: {values}")
    
    print("\n" + "="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate training configs and execution scripts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate configs for permeability task on GPU
  python configs/generate_configs.py --task permeability --device gpu
  
  # Generate configs for dispersion task on CPU
  python configs/generate_configs.py --task dispersion --device cpu
  
  # Use herbie mode (single YAML)
  python configs/generate_configs.py --task permeability --device gpu --herbie
  
  # List available models or presets
  python configs/generate_configs.py --list-models
  python configs/generate_configs.py --list-presets
        """
    )
    
    # Main arguments
    parser.add_argument(
        "--task",
        choices=list(TASK_CONFIGS.keys()),
        default="permeability",
        help="Task type for experiments"
    )
    parser.add_argument(
        "--device",
        choices=["gpu", "cpu"],
        default="gpu",
        help="Device type (affects batch size, epochs, SLURM config)"
    )
    
    # Mode selection
    parser.add_argument(
        "--herbie",
        action="store_true",
        help="Generate single YAML with all experiments (herbie mode)"
    )
    parser.add_argument(
        "--bigfacet",
        action="store_true",
        help="Generate YAMLs and launcher that distribute jobs across GPUs (bigfacet mode)"
    )
    parser.add_argument(
        "--conda-env",
        type=str,
        help="Optional conda environment name to activate in launcher scripts"
    )
    parser.add_argument(
        "--experiment",
        choices=list(SWEEP_PRESETS.keys()),
        help="Use a named sweep preset (e.g. lr_sweep, bs_sweep) to generate experiments"
    )
    
    # Optional overrides
    parser.add_argument(
        "--preset",
        choices=list(MODEL_PRESETS.keys()),
        default=DEFAULT_MODEL_PRESET,
        help=f"Model preset to use (default: {DEFAULT_MODEL_PRESET})"
    )
    parser.add_argument(
        "--exp-name",
        help="Custom experiment name (default: auto-generated)"
    )
    parser.add_argument(
        "--output-dir",
        help=f"Output directory (default: {OUTPUT_BASE_DIR})"
    )
    
    # Utility commands
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List all available models and exit"
    )
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="List all model presets and exit"
    )
    parser.add_argument(
        "--save-registry",
        type=str,
        help="Save default model registry to JSON file"
    )
    parser.add_argument(
        "--registry",
        type=str,
        help="Load model registry from JSON file"
    )
    
    args = parser.parse_args()
    
    # -------------------------
    # Handle utility commands
    # -------------------------
    
    registry_path = Path(args.registry) if args.registry else None
    registry = ModelRegistry(registry_path)
    
    if args.save_registry:
        registry.save(Path(args.save_registry))
        return
    
    if args.list_models:
        registry.print_catalog()
        return
    
    if args.list_presets:
        print_presets()
        return
    
    # -------------------------
    # Build configuration
    # -------------------------
    
    # Select models from preset
    model_names = MODEL_PRESETS[args.preset]
    
    # Generate experiment name
    if args.exp_name:
        exp_name = args.exp_name
    else:
        exp_name = EXP_NAME_TEMPLATE.format(task=args.task, device=args.device)
    
    # Get task configuration
    common_config = TASK_CONFIGS[args.task].copy()
    common_config["task"] = args.task
    
    # Apply device-specific overrides
    is_cpu = args.device == "cpu"
    if is_cpu and "training_overrides" in DEVICE_CONFIGS["cpu"]:
        common_config.update(DEVICE_CONFIGS["cpu"]["training_overrides"])
    
    # Set save path
    common_config["save_model_path"] = f"{RESULTS_BASE_DIR}/{exp_name}"
    
    # Get SLURM configuration
    slurm_dict = DEVICE_CONFIGS[args.device]["slurm"]
    slurm_config = SlurmConfig(**slurm_dict)
    
    # Get output directory
    output_dir = args.output_dir or OUTPUT_BASE_DIR
    
    # -------------------------
    # Generate configs
    # -------------------------
    
    print("\n" + "="*70)
    print("CONFIGURATION SUMMARY")
    print("="*70)
    print(f"Task: {args.task}")
    print(f"Device: {args.device}")
    print(f"Preset: {args.preset} ({len(model_names)} models)")
    print(f"Experiment: {exp_name}")
    print(f"Mode: {'herbie (single YAML)' if args.herbie else 'individual (separate jobs)'}")
    print(f"Output: {output_dir}")
    print("="*70 + "\n")
    
    # If the user requested a named sweep preset, expand it and generate jobs.
    if args.experiment:
        preset = SWEEP_PRESETS[args.experiment]

        # Map preset sweep types to actual lists in HYPERPARAM_SWEEPS.
        # Allow some common aliases (coarse/fine/small/light -> sweep).
        SWEEP_ALIAS = {
            "coarse": "sweep",
            "fine": "sweep",
            "small": "sweep",
            "light": "sweep",
            "common": "common",
            "scaling": "scaling",
            "single": "single",
            "sweep": "sweep",
        }

        sweep_configs: Dict[str, List[Any]] = {}
        for param, sweep_type in preset.items():
            # Resolve alias to a key present in HYPERPARAM_SWEEPS[param]
            mapped = SWEEP_ALIAS.get(sweep_type, sweep_type)
            candidates = HYPERPARAM_SWEEPS.get(param, {})
            values = None
            if mapped in candidates:
                values = candidates[mapped]
            else:
                # Fallback: prefer 'sweep', then 'scaling', then 'single'
                for fallback in ("sweep", "scaling", "single"):
                    if fallback in candidates:
                        values = candidates[fallback]
                        break
            if values is None:
                raise ValueError(f"No sweep values found for param '{param}' using preset '{args.experiment}'")

            sweep_configs[param] = values

        print(f"\n--- Generating experiment using preset: {args.experiment} ---")
        gen = ConfigGenerator(
            exp_name=exp_name,
            output_dir=output_dir,
            base_config=common_config,
            model_registry=registry,
            slurm_config=slurm_config,
            is_cpu=is_cpu,
            sweep_configs=sweep_configs,
        )

        # Default to individual mode unless herbie/bigfacet specified
        if args.herbie:
            gen.generate_herbie_mode(model_names, main_script=MAIN_SCRIPT, conda_env=args.conda_env)
        elif args.bigfacet:
            gen.generate_bigfacet_mode(model_names, main_script=MAIN_SCRIPT, conda_env=args.conda_env)
        else:
            gen.generate_individual_mode(model_names)

        return

    # Default behaviour: generate as before
    generator = ConfigGenerator(
        exp_name=exp_name,
        output_dir=output_dir,
        base_config=common_config,
        model_registry=registry,
        slurm_config=slurm_config,
        is_cpu=is_cpu,
    )
    
    if args.herbie:
        generator.generate_herbie_mode(model_names, main_script=MAIN_SCRIPT, conda_env=args.conda_env)
    elif args.bigfacet:
        generator.generate_bigfacet_mode(model_names, main_script=MAIN_SCRIPT, conda_env=args.conda_env)
    else:
        generator.generate_individual_mode(model_names)


if __name__ == "__main__":
    main()