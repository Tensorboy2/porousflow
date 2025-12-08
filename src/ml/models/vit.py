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
        mode: str = 'permeability'
    ):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.mode = mode
        
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
        
        if self.mode=='permeability':
            self.head = nn.Linear(embed_dim, num_classes)
        elif self.mode =='dispersion':
            self.head = nn.Linear(embed_dim+1, num_classes)
        
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
    
    def forward(self, x, peclet=None):
        # Validate inputs
        if self.mode == 'dispersion' and peclet is None:
            raise ValueError("peclet number must be provided when mode='dispersion'")
        
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
        
        # Concatenate Péclet number for dispersion mode
        if self.mode == 'dispersion':
            # Ensure peclet is the right shape (batch_size, 1)
            if peclet.dim() == 1:
                peclet = peclet.unsqueeze(1)
            x = torch.cat((peclet, x), dim=1)  # (batch_size, embed_dim+1)
        
        x = self.head(x)
        
        return x
    
    def get_num_params(self):
        """Get number of parameters"""
        return sum(p.numel() for p in self.parameters())


def load_vit_model(size='T16', in_channels=1, mode='permeability', pretrained_path=None):
    """
    Load a ViT model with specified version, size, input channels, and output classes.
    
    Args:
        size (str): Model size variant (currently supports 'T16')
        in_channels (int): Number of input channels
        mode (str): Task mode - 'permeability' (4 classes) or other (8 classes)
        pretrained_path (str): Path to pretrained model weights file
    
    Returns:
        ViT: An instance of the ViT model
    """
    if mode == 'permeability':
        num_classes = 4
    else:
        num_classes = 8
    
    if size == 'T16':
        model = ViT(
            img_size=128,
            patch_size=16,
            in_channels=in_channels,  # Use the parameter instead of hardcoded 1
            num_classes=num_classes,
            embed_dim=768,   # Base embedding size
            num_layers=12,   # Base depth
            num_heads=12,    # Base heads
            mlp_ratio=4.0,
            dropout=0.1,
            mode=mode
        )
    
    if pretrained_path:
        
        if not os.path.exists(pretrained_path):
            raise FileNotFoundError(f"Pretrained model not found at: {pretrained_path}")
        
        # Load the checkpoint
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Load weights with strict=False to allow partial loading
        # This is useful if the pretrained model has different num_classes
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded pretrained weights from: {pretrained_path}")
    
    return model
    # if not pretrained_path:
    #     return ConvNeXt(version=version, size=size, in_channels=in_channels, num_classes=num_classes)
    # else:
    #     model = ConvNeXt(version=version, size=size, in_channels=in_channels, num_classes=num_classes)
    #     model.load_state_dict(torch.load(pretrained_path,map_location='cpu'))
    #     return model

# For your specific use case (permeability and dispersion)
# def vit_permeability(img_size=128, num_classes=4, mode='permeability', **kwargs):
#     """Lightweight ViT for permeability and dispersion predictions"""
#     return ModularViT(
#         img_size=img_size,
#         patch_size=8,  # Smaller patches for 128x128 images
#         in_channels=1,  # Assuming grayscale input
#         embed_dim=128,  # Smaller embedding dimension
#         num_layers=6,   # Fewer layers for efficiency
#         num_heads=8,
#         mlp_ratio=2.0,
#         num_classes=num_classes,
#         mode=mode,
#         dropout=0.0,
#         **kwargs
#     )
# def vit_b16(num_classes=4, mode='permeability'):
#     """ViT-Base with 16x16 patches"""
#     return ModularViT(
#         img_size=128,
#         patch_size=16,
#         in_channels=1,
#         num_classes=num_classes,
#         embed_dim=768,   # Base embedding size
#         num_layers=12,   # Base depth
#         num_heads=12,    # Base heads
#         mlp_ratio=4.0,
#         dropout=0.1,
#         mode=mode
#     )
# def vit_b8(num_classes=4, mode='permeability'):
#     """ViT-Base with 16x16 patches"""
#     return ModularViT(
#         img_size=128,
#         patch_size=8,
#         in_channels=1,
#         num_classes=num_classes,
#         embed_dim=768,   # Base embedding size
#         num_layers=12,   # Base depth
#         num_heads=12,    # Base heads
#         mlp_ratio=4.0,
#         dropout=0.1,
#         mode=mode
#     )

# def vit_s8(num_classes=4, mode='permeability'):
#     """ViT-Small with 8x8 patches"""
#     return ModularViT(
#         img_size=128,
#         patch_size=8,
#         in_channels=1,
#         num_classes=num_classes,
#         embed_dim=384,   # Smaller embedding
#         num_layers=12,   # Same depth, smaller width
#         num_heads=6,     # Fewer heads
#         mlp_ratio=4.0,
#         dropout=0.1,
#         mode=mode
#     )



if __name__ == '__main__':
    models = {
        "ViT-B/16": vit_b16(),
        "ViT-B/8": vit_b8(),
        "ViT-S/8": vit_s8(),
        "Lightweight-ViT": vit_permeability(img_size=128, num_classes=4, mode='dispersion')
    }

    for name, model in models.items():
        n_params = model.get_num_params()
        print(f"{name:25s} | Params: {n_params/1e6:.2f} M")
