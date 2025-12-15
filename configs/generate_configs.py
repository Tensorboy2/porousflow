"""
Training configuration generator for PyTorch experiments.
Generates YAML configs and optional SLURM batch scripts.
Uses a JSON registry for model configurations.
"""
import yaml
import json
import argparse
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class SlurmConfig:
    """SLURM job configuration."""
    partition: str = "normal"
    ntasks: int = 1
    cpus_per_task: int = 2
    gres: str = "gpu:1"
    time: str = None
    mem: str = None
    
    def to_header(self, job_name: str) -> str:
        """Generate SLURM header."""
        lines = [
            "#!/bin/bash",
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
            "clip_grad": False,
            "description": "Vision Transformer Tiny with 16x16 patches"
        },
        "ViT-S16": {
            "type": "vit",
            "size": "S16",
            "clip_grad": False,
            "description": "Vision Transformer Small with 16x16 patches"
        },
        "ViT-B16": {
            "type": "vit",
            "size": "B16",
            "clip_grad": False,
            "description": "Vision Transformer Base with 16x16 patches"
        },
        "ViT-L16": {
            "type": "vit",
            "size": "L16",
            "clip_grad": False,
            "description": "Vision Transformer Large with 16x16 patches"
        },
        
        # ConvNeXt variants
        "ConvNeXt-Atto": {
            "type": "convnext",
            "size": "atto",
            "clip_grad": True,
            "description": "ConvNeXt Atto (smallest)"
        },
        "ConvNeXt-Femto": {
            "type": "convnext",
            "size": "femto",
            "clip_grad": True,
            "description": "ConvNeXt Femto"
        },
        "ConvNeXt-Pico": {
            "type": "convnext",
            "size": "pico",
            "clip_grad": True,
            "description": "ConvNeXt Pico"
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
            "description": "ResNet 18 layers"
        },
        "ResNet-34": {
            "type": "resnet",
            "size": "34",
            "clip_grad": True,
            "description": "ResNet 34 layers"
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
        common_config: Dict[str, Any],
        model_registry: ModelRegistry,
        slurm_config: SlurmConfig = None,
        cpu_mode: bool = False,
    ):
        self.exp_name = exp_name
        self.output_dir = Path(output_dir)
        self.common_config = common_config
        self.registry = model_registry
        self.slurm_config = slurm_config or SlurmConfig()
        self.cpu_mode = cpu_mode
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.yaml_dir = self.output_dir / exp_name
        self.yaml_dir.mkdir(exist_ok=True)
    
    def _build_experiment(self, model_name: str, custom_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Build experiment configuration for a model."""
        # Get model config from registry
        model_cfg = self.registry.get(model_name)
        
        # Create safe job name (replace slashes and special chars)
        safe_name = model_name.replace('/', '-').replace(' ', '_')
        job_name = f"{safe_name}_{self.exp_name}"
        
        # Build config matching run_model_training.py expectations
        exp = {
            "model": {
                "type": model_cfg["type"],
                "name": model_name,
                "size": model_cfg["size"],
                "in_channels": 1,  # grayscale for porous media
                "pretrained_path": None,
            },
        }
        
        # Add version if specified (e.g., for ConvNeXt V2)
        if "version" in model_cfg:
            exp["model"]["version"] = model_cfg["version"]
        
        # Merge common config
        exp.update(self.common_config)
        
        # Add model-specific settings
        exp["clip_grad"] = model_cfg.get("clip_grad", True)
        
        # Apply custom parameter overrides
        if custom_params:
            exp.update(custom_params)
        
        # Add CPU-specific settings
        if self.cpu_mode:
            exp["device"] = "cpu"
            if "batch_size" in exp:
                exp["batch_size"] = min(exp["batch_size"], 8)
        
        return exp
    
    def generate_herbie_mode(self, model_names: List[str]) -> Path:
        """Generate single YAML with all experiments."""
        experiments = [self._build_experiment(name) for name in model_names]
        
        mode_suffix = "_cpu" if self.cpu_mode else ""
        yaml_path = self.yaml_dir / f"{self.exp_name}_all{mode_suffix}.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump({"experiments": experiments}, f, default_flow_style=False)
        
        print(f"✓ Generated herbie mode YAML with {len(experiments)} experiments")
        if self.cpu_mode:
            print(f"  [CPU MODE] - device set to 'cpu', batch size reduced")
        print(f"  {yaml_path}")
        return yaml_path
    
    def generate_cpu_mode(
        self,
        model_names: List[str],
        main_script: str = "run_model_training.py"
    ) -> tuple[List[Path], Path]:
        """Generate CPU-only configs and bash scripts for local testing."""
        yaml_paths = []
        script_paths = []
        
        for model_name in model_names:
            safe_name = model_name.replace('/', '-').replace(' ', '_')
            job_name = f"{safe_name}_{self.exp_name}_cpu"
            
            # Generate YAML
            exp = self._build_experiment(model_name)
            yaml_path = self.yaml_dir / f"{job_name}.yaml"
            
            with open(yaml_path, "w") as f:
                yaml.dump({"experiments": [exp]}, f, default_flow_style=False)
            yaml_paths.append(yaml_path)
            
            # Generate bash script (no SLURM)
            script_content = f"""#!/bin/bash
# CPU test for {job_name}

python3 {main_script} --config "{yaml_path}"
"""
            script_path = self.yaml_dir / f"{job_name}.sh"
            with open(script_path, "w") as f:
                f.write(script_content)
            script_path.chmod(0o755)
            script_paths.append(script_path)
            
            print(f"✓ Generated CPU test: {model_name}")
        
        # Create launcher script
        launcher_path = self.output_dir / f"run_all_{self.exp_name}_cpu.sh"
        with open(launcher_path, "w") as f:
            f.write("#!/bin/bash\n")
            f.write(f"# Run all CPU tests for experiment: {self.exp_name}\n\n")
            for script in script_paths:
                f.write(f"bash {script}\n")
        
        launcher_path.chmod(0o755)
        print(f"\n✓ CPU test launcher: {launcher_path}")
        print(f"  Run with: bash {launcher_path}")
        
        return script_paths, launcher_path
    
    def generate_slurm_mode(
        self, 
        model_names: List[str],
        main_script: str = "run_model_training.py"
    ) -> tuple[List[Path], Path]:
        """Generate individual YAMLs and SLURM scripts."""
        yaml_paths = []
        slurm_paths = []
        
        for model_name in model_names:
            safe_name = model_name.replace('/', '-').replace(' ', '_')
            job_name = f"{safe_name}_{self.exp_name}"
            
            # Generate YAML
            exp = self._build_experiment(model_name)
            yaml_path = self.yaml_dir / f"{job_name}.yaml"
            
            with open(yaml_path, "w") as f:
                yaml.dump({"experiments": [exp]}, f, default_flow_style=False)
            yaml_paths.append(yaml_path)
            
            # Generate SLURM script
            slurm_content = f"""{self.slurm_config.to_header(job_name)}

python3 {main_script} --config "{yaml_path}"
"""
            slurm_path = self.yaml_dir / f"{job_name}.sh"
            with open(slurm_path, "w") as f:
                f.write(slurm_content)
            slurm_paths.append(slurm_path)
            
            print(f"✓ Generated: {model_name}")
        
        # Create launcher script
        launcher_path = self.output_dir / f"submit_all_{self.exp_name}.sh"
        with open(launcher_path, "w") as f:
            f.write("#!/bin/bash\n")
            f.write(f"# Launch all jobs for experiment: {self.exp_name}\n\n")
            for script in slurm_paths:
                f.write(f"sbatch {script}\n")
        
        launcher_path.chmod(0o755)
        print(f"\n✓ Launcher script: {launcher_path}")
        print(f"  Run with: bash {launcher_path}")
        
        return slurm_paths, launcher_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate training configs and SLURM scripts"
    )
    parser.add_argument(
        "--mode",
        choices=["herbie", "slurm", "cpu"],
        default="slurm",
        help="Generation mode: 'herbie' for single YAML, 'slurm' for individual jobs, 'cpu' for local testing"
    )
    parser.add_argument(
        "--exp-name",
        default="all_models_run_5",
        help="Experiment name"
    )
    parser.add_argument(
        "--output-dir",
        default="./experiments",
        help="Output directory for configs"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="Specific models to use (default: all small models for testing)"
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List all available models and exit"
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
    parser.add_argument(
        "--task",
        choices=["permeability", "dispersion"],
        default="permeability",
        help="Task type for the experiments"
    )
    args = parser.parse_args()
    
    # -------------------------
    # Load Model Registry
    # -------------------------
    
    registry_path = Path(args.registry) if args.registry else None
    registry = ModelRegistry(registry_path)
    
    # Handle special commands
    if args.save_registry:
        registry.save(Path(args.save_registry))
        return
    
    if args.list_models:
        registry.print_catalog()
        return
    
    # -------------------------
    # Configuration
    # -------------------------
    
    # Select models to use
    if args.models:
        model_names = args.models
    else:
        # Default to smaller models for quick testing
        model_names = [
            "ViT-T/16",
            "ViT-S/16",
            "ConvNeXt-Tiny",
            "ConvNeXt-Small",
            "ResNet-50",
        ]
    
    # Common configuration matching run_model_training.py expectations
    common_config = {
        "task": args.task,  # or "dispersion"
        "batch_size": 64,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "num_epochs": 1000,
        "decay": "cosine",
        "warmup_steps": 100,
        "save_model_path": "results"+f"/{args.exp_name}",
    }
    
    # SLURM configuration
    slurm_config = SlurmConfig(
        partition="normal",
        cpus_per_task=2,
        gres="gpu:1",
        # time="24:00:00",
        # mem="32G",
    )
    
    # Override for CPU mode
    cpu_mode = args.mode == "cpu"
    if cpu_mode:
        common_config["num_epochs"] = 8
        common_config["batch_size"] = 8
        slurm_config = SlurmConfig(
            partition="normal",
            cpus_per_task=4,
            gres=None,
            time="01:00:00",
            mem="16G",
        )
    
    # -------------------------
    # Generate configs
    # -------------------------
    
    generator = ConfigGenerator(
        exp_name=args.exp_name,
        output_dir=args.output_dir,
        common_config=common_config,
        model_registry=registry,
        slurm_config=slurm_config,
        cpu_mode=cpu_mode,
    )
    
    if args.mode == "herbie":
        generator.generate_herbie_mode(model_names)
    elif args.mode == "cpu":
        generator.generate_cpu_mode(model_names)
    else:
        generator.generate_slurm_mode(model_names)


if __name__ == "__main__":
    main()