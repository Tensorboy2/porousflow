import torch
import torch.nn as nn
import torch.nn.functional as F

class Stem(nn.Module):
    """Initial downsampling stem: reduces image size by 4x"""
    def __init__(self, in_chans=1, out_chans=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_chans, out_chans // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_chans // 2),
            nn.GELU(),
            nn.Conv2d(out_chans // 2, out_chans, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_chans),
        )

    def forward(self, x):
        return self.conv(x)

class Downsample(nn.Module):
    """Spatial downsampling between stages: reduces resolution by 2x"""
    def __init__(self, in_chans, out_chans):
        super().__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_chans),
            nn.Conv2d(in_chans, out_chans, kernel_size=3, stride=2, padding=1)
        )

    def forward(self, x):
        return self.conv(x)

class SingleHeadAttention(nn.Module):
    """Efficient Single-Head Attention for hierarchical stages"""
    def __init__(self, dim):
        super().__init__()
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.scale = dim ** -0.5

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, C).permute(2, 0, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class SHBlock(nn.Module):
    """A single SHViT Block: Attention + Convolutional FFN"""
    def __init__(self, dim, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SingleHeadAttention(dim)
        
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.dwconv = nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x, H, W):
        # Attention Path
        shortcut = x
        x = self.norm1(x)
        x = shortcut + self.attn(x)
        
        # Conv-FFN Path
        shortcut = x
        x = self.norm2(x)
        x = self.fc1(x)
        
        # Reshape to 2D for depthwise convolution
        B, N, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        
        x = self.act(x)
        x = self.fc2(x)
        return shortcut + x

class HierarchicalSHViT(nn.Module):
    def __init__(self, img_size=128, in_chans=1, num_classes=4, 
                 embed_dims=[64, 128, 256, 512], depths=[2, 2, 6, 2],
                 Pe_encoder=None, include_direction=False):
        super().__init__()
        
        self.stem = Stem(in_chans, embed_dims[0])
        
        # Building Stages
        self.stages = nn.ModuleList()
        for i in range(len(depths)):
            stage = nn.ModuleList([
                SHBlock(dim=embed_dims[i]) for _ in range(depths[i])
            ])
            self.stages.append(stage)
            
            # Add downsampling between stages
            if i < len(depths) - 1:
                self.stages.append(Downsample(embed_dims[i], embed_dims[i+1]))

        self.norm = nn.LayerNorm(embed_dims[-1])
        
        # Physics Integration (Pe and Direction)
        self.Pe_encoder = Pe_encoder
        extra_dim = 0
        if Pe_encoder:
            pe_dim = 5 if Pe_encoder == 'vector' else 1
            self.pe_mlp = nn.Sequential(nn.Linear(pe_dim, 16), nn.GELU(), nn.Linear(16, 16))
            extra_dim += 16
        else: self.pe_mlp = None

        if include_direction:
            self.dir_mlp = nn.Sequential(nn.Linear(2, 16), nn.GELU(), nn.Linear(16, 16))
            extra_dim += 16
        else: self.dir_mlp = None

        self.head = nn.Linear(embed_dims[-1] + extra_dim, num_classes)

    def forward(self, x, Pe=None, Direction=None):
        B = x.shape[0]
        x = self.stem(x) # [B, 64, H/4, W/4]
        
        for stage in self.stages:
            if isinstance(stage, Downsample):
                x = stage(x)
            else:
                # Transformer blocks
                C, H, W = x.shape[1], x.shape[2], x.shape[3]
                x = x.flatten(2).transpose(1, 2) # [B, N, C]
                for block in stage:
                    x = block(x, H, W)
                x = x.transpose(1, 2).reshape(B, C, H, W) # Back to 4D for next stage/downsample

        x = self.norm(x.flatten(2).mean(-1)) # Global Average Pooling

        # Aux inputs
        extras = []
        if self.pe_mlp and Pe is not None:
            if self.Pe_encoder == "log": Pe = torch.log(Pe + 1e-6)
            extras.append(self.pe_mlp(Pe.view(B, -1)))
        if hasattr(self, 'dir_mlp') and Direction is not None:
            extras.append(self.dir_mlp(Direction))
            
        if extras:
            x = torch.cat([x] + extras, dim=1)
            
        return self.head(x)

# --------------------------------------------------------------------------
# CONFIGURATIONS: Comparable to ViT-T, ViT-S, ViT-B
# --------------------------------------------------------------------------
HIER_SHVIT_CONFIGS = {
    'T': { # Tiny: ~6M Params (ViT-T16 is ~5.7M)
        'embed_dims': [32, 64, 160, 256],
        'depths': [2, 2, 6, 2],
    },
    'S': { # Small: ~24M Params (ViT-S16 is ~22M)
        'embed_dims': [64, 128, 320, 512],
        'depths': [2, 2, 6, 2],
    },
    'B': { # Base: ~88M Params (ViT-B16 is ~86M)
        'embed_dims': [96, 192, 448, 768],
        'depths': [2, 2, 18, 2],
    },
    'L': { # Large: ~310M Params (ViT-L16 is ~307M)
        'embed_dims': [128, 256, 640, 1024],
        'depths': [2, 2, 18, 2],
    }
}

def load_hierarchical_shvit(mdl_cfg, in_chans=1, task='permeability', **kwargs):
    """
    Loader for Hierarchical SHViT models with standard size variants.
    """
    # Resolve size mapping
    size = mdl_cfg['size']
    size_key = size.upper()[0] if size.upper().startswith('V') else size.upper()
    if size_key not in HIER_SHVIT_CONFIGS:
        print(f"Size {size} not found, defaulting to Tiny (T)")
        size_key = 'T'
        
    cfg = HIER_SHVIT_CONFIGS[size_key]
    
    # Task to classes
    num_classes = 4 # Default for your permeability/dispersion tasks
    
    model = HierarchicalSHViT(
        in_chans=in_chans,
        num_classes=num_classes,
        embed_dims=cfg['embed_dims'],
        depths=cfg['depths'],
        **kwargs
    )
    return model

# --------------------------------------------------------------------------
# Parameter Comparison Test
# --------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"{'Variant':<10} | {'Parameters (M)':<15} | {'Final Feature Dim':<15}")
    print("-" * 45)
    
    for v in ['T', 'S', 'B', 'L']:
        model = load_hierarchical_shvit(size=v)
        params = sum(p.numel() for p in model.parameters()) / 1e6
        final_dim = HIER_SHVIT_CONFIGS[v]['embed_dims'][-1]
        print(f"{v:<10} | {params:<15.2f} | {final_dim:<15}")

    # Example: Running a forward pass on 'Small' variant
    x = torch.randn(1, 1, 128, 128)
    pe = torch.tensor([[50.0]])
    model_s = load_hierarchical_shvit('S', Pe_encoder='log')
    output = model_s(x, Pe=pe)
    print(f"\nForward Pass (Small) Result: {output.shape}")