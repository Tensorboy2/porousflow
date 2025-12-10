"""
Training configuration generator for PyTorch experiments.
Generates YAML configs and optional SLURM batch scripts.
"""
import yaml
import argparse
import os
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass, asdict


@dataclass
class ModelConfig:
    """Configuration for a single model."""
    name: str
    clip_grad: bool = True
    custom_params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.custom_params is None:
            self.custom_params = {}


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
            f"#SBATCH --gres={self.gres}",
        ]
        if self.time:
            lines.append(f"#SBATCH --time={self.time}")
        if self.mem:
            lines.append(f"#SBATCH --mem={self.mem}")
        return "\n".join(lines)


class ConfigGenerator:
    """Generate training configs and SLURM scripts."""
    
    def __init__(
        self,
        exp_name: str,
        output_dir: Path,
        common_config: Dict[str, Any],
        slurm_config: SlurmConfig = None,
    ):
        self.exp_name = exp_name
        self.output_dir = Path(output_dir)
        self.common_config = common_config
        self.slurm_config = slurm_config or SlurmConfig()
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.yaml_dir = self.output_dir / exp_name
        self.yaml_dir.mkdir(exist_ok=True)
    
    def _build_experiment(self, model: ModelConfig) -> Dict[str, Any]:
        """Build experiment configuration for a model."""
        job_name = f"{model.name}_{self.exp_name}"
        
        # Start with common config
        # Build a structured model dict so loaders can consume it directly
        model_type = None
        size = None
        name = model.name
        # Infer type and size from the model name where possible
        if 'vit' in model.name.lower() or 'vit_' in model.name.lower() or model.name.lower().startswith('vit'):
            model_type = 'vit'
            # Example names: ViT_T16, ViT_S16
            parts = model.name.split('_')
            size = parts[1] if len(parts) > 1 else 'T16'
        elif 'convnext' in model.name.lower():
            model_type = 'convnext'
            # Example names: ConvNeXtTiny, ConvNeXtSmall
            # take trailing part as size if present
            for s in ['atto','femto','pico','nano','tiny','small','base','large']:
                if s in model.name.lower():
                    size = s
                    break
            if size is None:
                size = 'tiny'
        elif 'resnet' in model.name.lower():
            model_type = 'resnet'
            # Example: ResNet50
            # extract digits
            import re
            m = re.search(r"(18|34|50|101|152)", model.name)
            size = m.group(1) if m else '18'
        else:
            model_type = model.name.lower()

        exp = {
            "model": {
                "type": model_type,
                "name": name,
                "size": size,
                # default grayscale input for porous media
                "in_channels": 1,
                "pretrained_path": None,
            },
            "save_model_path": f"{job_name}.pth",
            "save_path": f"{job_name}.csv",
        }
        
        # Merge common config
        exp.update(self.common_config)
        
        # Apply model-specific overrides
        if "hyperparameters" not in exp:
            exp["hyperparameters"] = {}
        
        exp["hyperparameters"]["clip_grad"] = model.clip_grad
        exp["hyperparameters"].update(model.custom_params)
        
        return exp
    
    def generate_herbie_mode(self, models: List[ModelConfig]) -> Path:
        """Generate single YAML with all experiments."""
        experiments = [self._build_experiment(m) for m in models]
        
        yaml_path = self.yaml_dir / f"{self.exp_name}_all.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump({"experiments": experiments}, f, default_flow_style=False)
        
        print(f"✓ Generated herbie mode YAML with {len(experiments)} experiments")
        print(f"  {yaml_path}")
        return yaml_path
    
    def generate_slurm_mode(
        self, 
        models: List[ModelConfig],
        main_script: str = "main.py"
    ) -> tuple[List[Path], Path]:
        """Generate individual YAMLs and SLURM scripts."""
        yaml_paths = []
        slurm_paths = []
        
        for model in models:
            job_name = f"{model.name}_{self.exp_name}"
            
            # Generate YAML
            exp = self._build_experiment(model)
            yaml_path = self.yaml_dir / f"{job_name}.yaml"
            
            with open(yaml_path, "w") as f:
                yaml.dump({"experiments": [exp]}, f, default_flow_style=False)
            yaml_paths.append(yaml_path)
            
            # Generate SLURM script
            slurm_content = f"""{self.slurm_config.to_header(job_name)}

python {main_script} "{yaml_path}"
"""
            slurm_path = self.yaml_dir / f"{job_name}.sh"
            with open(slurm_path, "w") as f:
                f.write(slurm_content)
            slurm_paths.append(slurm_path)
            
            print(f"✓ Generated: {job_name}")
        
        # Create launcher script
        launcher_path = self.output_dir / f"submit_all_{self.exp_name}.sh"
        with open(launcher_path, "w") as f:
            f.write("#!/bin/bash\n")
            f.write(f"# Launch all jobs for experiment: {self.exp_name}\n\n")
            for script in slurm_paths:
                f.write(f"sbatch {script}\n")
        
        launcher_path.chmod(0o755)  # Make executable
        print(f"\n✓ Launcher script: {launcher_path}")
        print(f"  Run with: bash {launcher_path}")
        
        return slurm_paths, launcher_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate training configs and SLURM scripts"
    )
    parser.add_argument(
        "--mode",
        choices=["herbie", "slurm"],
        default="slurm",
        help="Generation mode: 'herbie' for single YAML, 'slurm' for individual jobs"
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
    args = parser.parse_args()
    
    # -------------------------
    # Configuration
    # -------------------------
    
    # Define models
    models = [
        ModelConfig("ViT_T16", clip_grad=False),
        ModelConfig("ViT_S16", clip_grad=False),
        ModelConfig("ConvNeXtSmall", clip_grad=True),
        ModelConfig("ConvNeXtTiny", clip_grad=True),
        ModelConfig("ResNet50", clip_grad=True),
        ModelConfig("ResNet101", clip_grad=True),
        # Add more models with custom params:
        # ModelConfig("ViT_L16", clip_grad=False, custom_params={"lr": 1e-4}),
    ]
    
    # Common configuration for all experiments
    common_config = {
        "hyperparameters": {
            "epochs": 1000,
            "batch_size": 64,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            # Add more common hyperparameters
        },
        "data": {
            "dataset": "imagenet",
            "num_workers": 4,
            # Add more data config
        },
    }
    
    # SLURM configuration
    slurm_config = SlurmConfig(
        partition="normal",
        cpus_per_task=2,
        gres="gpu:1",
        # time="24:00:00",  # Uncomment to set time limit
        # mem="32G",  # Uncomment to set memory limit
    )
    
    # -------------------------
    # Generate configs
    # -------------------------
    
    generator = ConfigGenerator(
        exp_name=args.exp_name,
        output_dir=args.output_dir,
        common_config=common_config,
        slurm_config=slurm_config,
    )
    
    if args.mode == "herbie":
        generator.generate_herbie_mode(models)
    else:
        generator.generate_slurm_mode(models)


if __name__ == "__main__":
    main()