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
        ]
        if self.gres:
            lines.append(f"#SBATCH --gres={self.gres}")
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
        cpu_mode: bool = False,
    ):
        self.exp_name = exp_name
        self.output_dir = Path(output_dir)
        self.common_config = common_config
        self.slurm_config = slurm_config or SlurmConfig()
        self.cpu_mode = cpu_mode
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.yaml_dir = self.output_dir / exp_name
        self.yaml_dir.mkdir(exist_ok=True)
    
    def _build_experiment(self, model: ModelConfig) -> Dict[str, Any]:
        """Build experiment configuration for a model."""
        job_name = f"{model.name}_{self.exp_name}"
        
        # Infer model type and size from the model name
        model_type = None
        size = None
        name = model.name
        
        if 'vit' in model.name.lower() or 'vit_' in model.name.lower() or model.name.lower().startswith('vit'):
            model_type = 'vit'
            # Example names: ViT_T16, ViT_S16
            parts = model.name.split('_')
            size = parts[1] if len(parts) > 1 else 'T16'
        elif 'convnext' in model.name.lower():
            model_type = 'convnext'
            # Example names: ConvNeXtTiny, ConvNeXtSmall
            for s in ['atto','femto','pico','nano','tiny','small','base','large']:
                if s in model.name.lower():
                    size = s
                    break
            if size is None:
                size = 'tiny'
        elif 'resnet' in model.name.lower():
            model_type = 'resnet'
            # Example: ResNet50
            import re
            m = re.search(r"(18|34|50|101|152)", model.name)
            size = m.group(1) if m else '18'
        else:
            model_type = model.name.lower()

        # Build config matching run_model_training.py expectations
        exp = {
            "model": {
                "type": model_type,
                "name": name,
                "size": size,
                "in_channels": 1,  # grayscale for porous media
                "pretrained_path": None,
            },
        }
        
        # Merge common config (will include task, batch_size, etc.)
        exp.update(self.common_config)
        
        # Apply model-specific overrides
        exp["clip_grad"] = model.clip_grad
        if model.custom_params:
            exp.update(model.custom_params)
        
        # Add CPU-specific settings
        if self.cpu_mode:
            exp["device"] = "cpu"
            # Reduce batch size for CPU testing
            if "batch_size" in exp:
                exp["batch_size"] = min(exp["batch_size"], 8)
        
        return exp
    
    def generate_herbie_mode(self, models: List[ModelConfig]) -> Path:
        """Generate single YAML with all experiments."""
        experiments = [self._build_experiment(m) for m in models]
        
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
        models: List[ModelConfig],
        main_script: str = "run_model_training.py"
    ) -> tuple[List[Path], Path]:
        """Generate CPU-only configs and bash scripts for local testing."""
        yaml_paths = []
        script_paths = []
        
        for model in models:
            job_name = f"{model.name}_{self.exp_name}_cpu"
            
            # Generate YAML
            exp = self._build_experiment(model)
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
            script_path.chmod(0o755)  # Make executable
            script_paths.append(script_path)
            
            print(f"✓ Generated CPU test: {job_name}")
        
        # Create launcher script
        launcher_path = self.output_dir / f"run_all_{self.exp_name}_cpu.sh"
        with open(launcher_path, "w") as f:
            f.write("#!/bin/bash\n")
            f.write(f"# Run all CPU tests for experiment: {self.exp_name}\n\n")
            for script in script_paths:
                f.write(f"bash {script}\n")
        
        launcher_path.chmod(0o755)  # Make executable
        print(f"\n✓ CPU test launcher: {launcher_path}")
        print(f"  Run with: bash {launcher_path}")
        
        return script_paths, launcher_path
    
    def generate_slurm_mode(
        self, 
        models: List[ModelConfig],
        main_script: str = "run_model_training.py"
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

python3 {main_script} --config "{yaml_path}"
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
        # ModelConfig("ViT_L16", clip_grad=False, custom_params={"learning_rate": 1e-4}),
    ]
    
    # Common configuration matching run_model_training.py expectations
    common_config = {
        "task": "permeability",  # or "dispersion"
        "batch_size": 64,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "num_epochs": 1000,
        # Add any other config keys your script expects
    }
    
    # SLURM configuration (not used in CPU mode)
    slurm_config = SlurmConfig(
        partition="normal",
        cpus_per_task=2,
        gres="gpu:1",
        # time="24:00:00",  # Uncomment to set time limit
        # mem="32G",  # Uncomment to set memory limit
    )
    
    # Override for CPU mode
    cpu_mode = args.mode == "cpu"
    if cpu_mode:
        # For CPU testing, reduce epochs and use smaller batch
        common_config["num_epochs"] = 2
        common_config["batch_size"] = 8
        slurm_config = SlurmConfig(
            partition="normal",
            cpus_per_task=4,
            gres=None,  # No GPU
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
        slurm_config=slurm_config,
        cpu_mode=cpu_mode,
    )
    
    if args.mode == "herbie":
        generator.generate_herbie_mode(models)
    elif args.mode == "cpu":
        generator.generate_cpu_mode(models)
    else:
        generator.generate_slurm_mode(models)


if __name__ == "__main__":
    main()