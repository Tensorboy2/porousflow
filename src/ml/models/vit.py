'''
vit.py
A module for implementing modular Vision Transformers for permeability and dispersion predictions
'''
import torch
import torch.nn as nn
import os

class PatchEmbedding(nn.Module):
    """Convert image patches to embeddings"""
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_channels: int = 3, embed_dim: int = 768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        # x: (batch_size, channels, height, width)
        x = self.proj(x)  # (batch_size, embed_dim, n_patches_h, n_patches_w)
        x = x.flatten(2)  # (batch_size, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (batch_size, n_patches, embed_dim)
        return x


class MultiHeadAttention(nn.Module):
    """Multi-head self attention mechanism"""
    def __init__(self, embed_dim: int = 768, num_heads: int = 12, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        
    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch_size, num_heads, seq_len, head_dim)
        q, k, v = qkv.unbind(0)
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(batch_size, seq_len, embed_dim)
        x = self.proj(x)
        
        return x


class MLP(nn.Module):
    """MLP block with GELU activation"""
    def __init__(self, embed_dim: int = 768, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer encoder block"""
    def __init__(self, embed_dim: int = 768, num_heads: int = 12, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio, dropout)
        
    def forward(self, x):
        # Pre-norm architecture with proper residuals
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViT(nn.Module):
    """
    Modular Vision Transformer with configurable parameters
    
    Args:
        img_size: Input image size (assumes square images)
        patch_size: Size of image patches
        in_channels: Number of input channels
        num_classes: Number of output classes
        embed_dim: Embedding dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        mlp_ratio: MLP expansion ratio
        dropout: Dropout rate
        use_cls_token: Whether to use classification token
    """
    def __init__(
        self,
        img_size: int = 128,
        patch_size: int = 16,
        in_channels: int = 1,
        num_classes: int = 1000,
        embed_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        task: str = 'permeability'
    ):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.task = task
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.n_patches
            
        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        
        # Classification head
        self.norm = nn.LayerNorm(embed_dim)

        if self.task=='permeability':
            self.head = nn.Linear(embed_dim, num_classes)
        elif self.task =='dispersion':
            # self.pe_mlp = nn.Sequential(nn.Linear(1, 16), nn.GELU(), nn.Linear(16, 16))
            self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        # Initialize position embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # Initialize linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x, Pe=None):
        # Validate inputs
        # if self.task == 'dispersion' and Pe is None:
        #     raise ValueError("Pe number must be provided when mode='dispersion'")
        
        batch_size = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (batch_size, num_patches, embed_dim)
        
        # Add position embedding
        x = x + self.pos_embed
        x = self.dropout(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        x = x.mean(dim=1)  # Global average pooling -> (batch_size, embed_dim)
        
        x = x.view(batch_size, -1)  # (batch_size, embed_dim)
        # Concatenate Péclet number for dispersion mode
        # if self.task == 'dispersion':
        #     if Pe is None:
        #         raise ValueError("Pe must be provided for dispersion task")
        #     Pe = torch.ones(x.size(0), 1, device=x.device) * Pe  # Ensure Pe shape is (B, 1)
        #     Pe = self.pe_mlp(Pe)  # (B, 16)
        #     x = torch.cat([x, Pe], dim=1)  # (batch_size, embed_dim+1)
        
        x = self.head(x)
        
        return x
    
    def get_num_params(self):
        """Get number of parameters"""
        return sum(p.numel() for p in self.parameters())

VIT_CONFIGS = {
    'tiny': {'embed_dim': 192, 'num_layers': 12, 'num_heads': 3},
    'small': {'embed_dim': 384, 'num_layers': 12, 'num_heads': 6},
    'base': {'embed_dim': 768, 'num_layers': 12, 'num_heads': 12},
    'large': {'embed_dim': 1024, 'num_layers': 24, 'num_heads': 16},
}

def load_vit_model(config_or_size='T16', in_channels: int = 1, task = 'permeability', pretrained_path: str = None, **kwargs):
    """
    Flexible loader for ViT models.

    Accepts either:
    - a config dictionary (as produced by `generate_configs.py`), or
    - the original argument signature (size, in_channels, mode, pretrained_path).

    Recognized config keys: `size`, `in_channels`, `mode`, `pretrained_path`, `num_classes`,
    and other ViT kwargs (e.g. `img_size`, `patch_size`, `embed_dim`, `num_layers`, `num_heads`).
    """
    # Default mapping from mode to num_classes
    if task == 'permeability':
        num_classes = 4
    elif task == 'dispersion':
        num_classes = 4
    else:
        raise ValueError(f"Unknown task: {cfg['task']}. Supported tasks: ['permeability', 'dispersion']")

    # Extract from config dict or use provided args
    if isinstance(config_or_size, dict):
        cfg = config_or_size
        size = cfg.get('size', cfg.get('name', 'T16'))
        in_channels = cfg.get('in_channels', in_channels)
        task = cfg.get('task', task)
        pretrained_path = cfg.get('pretrained_path', pretrained_path)
        num_classes = num_classes
        img_size = cfg.get('img_size', cfg.get('image_size', 128))
        patch_size = cfg.get('patch_size', 16)
        embed_dim = cfg.get('embed_dim', None)
        num_layers = cfg.get('num_layers', None)
        num_heads = cfg.get('num_heads', None)
        mlp_ratio = cfg.get('mlp_ratio', 4.0)
        dropout = cfg.get('dropout', 0.1)
    else:
        size = config_or_size
        num_classes = num_classes
        img_size = kwargs.pop('img_size', 128)
        patch_size = kwargs.pop('patch_size', 16)
        embed_dim = kwargs.pop('embed_dim', None)
        num_layers = kwargs.pop('num_layers', None)
        num_heads = kwargs.pop('num_heads', None)
        mlp_ratio = kwargs.pop('mlp_ratio', 4.0)
        dropout = kwargs.pop('dropout', 0.1)

    # If a named size (tiny/small/base/large) is provided and any of embed_dim/num_layers/num_heads are missing,
    # fill them from VIT_CONFIGS
    size_key = str(size).lower()
    if size_key in VIT_CONFIGS:
        cfg_map = VIT_CONFIGS[size_key]
        if embed_dim is None:
            embed_dim = cfg_map.get('embed_dim', 768)
        if num_layers is None:
            num_layers = cfg_map.get('num_layers', 12)
        if num_heads is None:
            num_heads = cfg_map.get('num_heads', 12)
    else:
        # Fallback defaults
        embed_dim = embed_dim or 768
        num_layers = num_layers or 12
        num_heads = num_heads or 12

    # Create model
    model = ViT(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=in_channels,
        num_classes=num_classes,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        dropout=dropout,
        task=task,
    )

    # Load pretrained weights if requested
    if pretrained_path:
        if not os.path.exists(pretrained_path):
            raise FileNotFoundError(f"Pretrained model not found at: {pretrained_path}")

        checkpoint = torch.load(pretrained_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded pretrained weights from: {pretrained_path}")

    return model
    


if __name__ == "__main__":    
    # Example usage
    print("Testing ViT models...")
    
    # Test ViT-Tiny
    x = torch.randn(2, 1, 128, 128)
    model_tiny = load_vit_model(config_or_size='tiny', in_channels=1, task='permeability')
    output_tiny = model_tiny(x)
    print(f"ViT-Tiny output shape: {output_tiny.shape}")
    
    # Test ViT-Base for dispersion
    peclet = torch.tensor([10.0, 20.0])
    model_base = load_vit_model(config_or_size='base', in_channels=1, task='dispersion')
    output_base = model_base(x, peclet=peclet)
    print(f"ViT-Base (dispersion) output shape: {output_base.shape}")