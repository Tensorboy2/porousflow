"""
convnext.py

Modular ConvNeXt implementation (V1, V2, RMSNorm) for regression/classification tasks, with size configs from atto to large.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
torch.manual_seed(0)
path = os.path.dirname(__file__)

# --- Normalization Layers ---
class LayerNorm2d(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps)
    
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
        return x

class RMSNorm2d(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(1, dim, 1, 1))
    
    def forward(self, x):
        norm = x.pow(2).mean(dim=1, keepdim=True).add(self.eps).sqrt()
        return x / norm * self.scale

class GRN(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))  # Adjusted for (B, H, W, C)
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))   # Adjusted for (B, H, W, C)
        self.eps = eps
    
    def forward(self, x):
        # Input x is in (B, H, W, C) format
        gx = torch.norm(x, p=2, dim=-1, keepdim=True)  # L2 norm along channel dim
        nx = gx / (gx.mean(dim=(1, 2), keepdim=True) + self.eps)  # Normalize by spatial mean
        return self.gamma * (x * nx) + self.beta + x

# --- ConvNeXt Blocks ---
class ConvNeXtBlockV1(nn.Module):
    def __init__(self, dim, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        # optional gammas
        # self.gamma_dw = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
        self.gamma_mlp = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)

    def forward(self, x):
        # Convert to channels-last for the block
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        shortcut = x

        # Depthwise residual
        dw = self.dwconv(x.permute(0, 3, 1, 2))          # conv expects channels-first
        dw = dw.permute(0, 2, 3, 1)                      # back to channels-last
        dw = self.norm(dw)
        # dw = self.gamma_dw * dw
        # x = shortcut + dw                                # first residual

        # MLP residual
        mlp = self.pwconv1(dw)
        mlp = self.act(mlp)
        mlp = self.pwconv2(mlp)
        mlp = self.gamma_mlp * mlp
        x = x + mlp                                      # second residual

        # Convert back to channels-first for output
        x = x.permute(0, 3, 1, 2)                        # (B, C, H, W)
        return x


class ConvNeXtBlockV2(nn.Module):
    def __init__(self, dim, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm2d(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True) if layer_scale_init_value > 0 else None
    
    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = self.norm(x)  # Apply norm while in (B, C, H, W) format
        x = x.permute(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)  # GRN now expects (B, H, W, C)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
        x = shortcut + x
        return x

class ConvNeXtBlockRMS(nn.Module):
    def __init__(self, dim, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = RMSNorm2d(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True) if layer_scale_init_value > 0 else None
    
    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = self.norm(x)  # Apply norm while in (B, C, H, W) format
        x = x.permute(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
        x = shortcut + x
        return x

# --- Model Configurations ---
CONVNEXT_CONFIGS = {
    'atto':   {'depths': [2, 2, 6, 2],   'dims': [40, 80, 160, 320]},
    'femto':  {'depths': [2, 2, 6, 2],   'dims': [48, 96, 192, 384]},
    'pico':   {'depths': [2, 2, 6, 2],   'dims': [64, 128, 256, 512]},
    'nano':   {'depths': [2, 2, 8, 2],   'dims': [80, 160, 320, 640]},
    'tiny':   {'depths': [3, 3, 9, 3],   'dims': [96, 192, 384, 768]},
    'small':  {'depths': [3, 3, 27, 3],  'dims': [96, 192, 384, 768]},
    'base':   {'depths': [3, 3, 27, 3],  'dims': [128, 256, 512, 1024]},
    'large':  {'depths': [3, 3, 27, 3],  'dims': [192, 384, 768, 1536]},
}

class ConvNeXtEncoder(nn.Module):
    def __init__(self, in_chans, depths, dims, block_type):
        super().__init__()
        
        # Create downsample layers
        self.downsample_layers = nn.ModuleList()
        
        # Stem layer
        if block_type == ConvNeXtBlockRMS:
            norm_layer = RMSNorm2d(dims[0])
        else:
            norm_layer = LayerNorm2d(dims[0])
        
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            norm_layer
        )
        self.downsample_layers.append(stem)
        
        # Intermediate downsample layers
        for i in range(3):
            if block_type == ConvNeXtBlockRMS:
                norm_layer = RMSNorm2d(dims[i])
            else:
                norm_layer = LayerNorm2d(dims[i])
                
            downsample = nn.Sequential(
                norm_layer,
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2)
            )
            self.downsample_layers.append(downsample)
        
        # Create stages
        self.stages = nn.ModuleList()
        for i in range(4):
            stage = nn.Sequential(*[block_type(dims[i]) for _ in range(depths[i])])
            self.stages.append(stage)
    
    def forward(self, x):
        for downsample, stage in zip(self.downsample_layers, self.stages):
            x = downsample(x)
            x = stage(x)
        return x

class ConvNeXt(nn.Module):
    def __init__(self, version='v1', size='tiny', in_channels=1, num_classes=4, task='permeability'):
        super().__init__()
        
        # Validate inputs
        if size not in CONVNEXT_CONFIGS:
            raise ValueError(f"Unknown size: {size}. Available sizes: {list(CONVNEXT_CONFIGS.keys())}")
        
        config = CONVNEXT_CONFIGS[size]
        depths, dims = config['depths'], config['dims']
        
        # Select block type
        if version == 'v1':
            block_type = ConvNeXtBlockV1
        elif version == 'v2':
            block_type = ConvNeXtBlockV2 
        elif version == 'rms':
            block_type = ConvNeXtBlockRMS
        else:
            raise ValueError(f"Unknown version: {version}. Available versions: ['v1', 'v2', 'rms']")
        
        self.task = task  # Store task type
        self.encoder = ConvNeXtEncoder(in_channels, depths, dims, block_type)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Configure head based on task
        if task == 'permeability':
            self.fc = nn.Linear(dims[-1], num_classes)
        elif task == 'dispersion':
            self.pe_mlp = nn.Sequential(nn.Linear(1, 16), nn.Linear(16, 16))
            self.fc = nn.Linear(dims[-1] + 16, num_classes)
        elif task == 'dispersion_direction':
            self.fc = nn.Linear(dims[-1] + 2, num_classes)
        else:
            raise ValueError(f"Unknown task: {task}. Available tasks: ['permeability', 'dispersion', 'dispersion_direction']")
        
        # self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x, Pe=None):
        """
        Args:
            x: Input tensor (B, C, H, W)
            Pe: Peclet number (B, 1) or (B,) - only required for dispersion task
        """
        x = self.encoder(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # (B, dims[-1])
        
        if self.task == 'dispersion':
            if Pe is None:
                raise ValueError("Pe must be provided for dispersion task")
            Pe = torch.ones(x.size(0), 1, device=x.device) * Pe  # Ensure Pe shape is (B, 1)
            Pe = self.pe_mlp(Pe)  # (B, 16)
            x = torch.cat([x, Pe], dim=1)  # (B, dims[-1] + 16)
        
        x = self.fc(x)
        return x

def load_convnext_model(config_or_version='v1', size='tiny', in_channels=1, task='permeability',  pretrained_path = None):
    """
    Flexible loader for ConvNeXt models.

    Accepts either a config dictionary or the traditional signature.
    Recognized config keys: `version`, `size`, `in_channels`, `num_classes`, `pretrained_path`.
    """
    if task == 'permeability':
        num_classes = 4
    elif task == 'dispersion':
        num_classes = 8
    else:
        raise ValueError(f"Unknown task: {cfg['task']}. Supported tasks: ['permeability', 'dispersion']")
    
    if isinstance(config_or_version, dict):
        cfg = config_or_version
        version = cfg.get('version', 'v1')
        size = cfg.get('size', size)
        in_channels = cfg.get('in_channels', in_channels)
        # num_classes = cfg.get('task', num_classes)
        pretrained_path = cfg.get('pretrained_path', pretrained_path)
    else:
        version = config_or_version

    model = ConvNeXt(version=version, size=size, in_channels=in_channels, num_classes=num_classes, task=task)

    if pretrained_path:
        if not os.path.exists(pretrained_path):
            raise FileNotFoundError(f"Pretrained model not found at: {pretrained_path}")
        model.load_state_dict(torch.load(pretrained_path, map_location='cpu'))

    return model


def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    print("Testing ConvNeXt models...")
    
    # Set seed for reproducibility
    torch.manual_seed(42)
    
    x = torch.randn(2, 1, 128, 128)
    
    # Test different versions and sizes
    for version in ['v1', 'v2', 'rms']:
        for size in ['atto', 'pico', 'tiny']:
            try:
                model = load_convnext_model(config_or_version=version, size=size, in_channels=1, num_classes=4)
                out = model(x)
                params = count_parameters(model)
                print(f"ConvNeXt-{version}-{size}: output shape {out.shape}, params {params:,}")
            except Exception as e:
                print(f"Error with ConvNeXt-{version}-{size}: {e}")
    
    # Test large model
    try:
        model = load_convnext_model(config_or_version='v2', size='large', in_channels=1, num_classes=4)
        params = count_parameters(model)
        print(f"\nConvNeXt-v2-large param count: {params:,}")
    except Exception as e:
        print(f"Error with ConvNeXt-v2-large: {e}")