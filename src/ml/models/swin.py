'''
swin.py
A module for implementing Swin Transformer for permeability and dispersion predictions
'''
import torch
torch.manual_seed(0)
import torch.nn as nn
import numpy as np
import os


def window_partition(x, window_size):
    """
    Partition feature map into non-overlapping windows.
    Args:
        x: (B, H, W, C)
        window_size: window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Merge windows back to feature map.
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size: window size
        H: Height of image
        W: Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class PatchEmbedding(nn.Module):
    """Image to Patch Embedding with 4x4 patches"""
    def __init__(self, img_size=128, patch_size=4, in_channels=1, embed_dim=96):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)  # (B, embed_dim, H//patch_size, W//patch_size)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        x = self.norm(x)
        return x


class PatchMerging(nn.Module):
    """Patch Merging Layer - reduces spatial resolution and increases channels"""
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        """
        x: (B, H*W, C)
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        # Downsample by taking every other patch
        x0 = x[:, 0::2, 0::2, :]  # (B, H/2, W/2, C)
        x1 = x[:, 1::2, 0::2, :]  # (B, H/2, W/2, C)
        x2 = x[:, 0::2, 1::2, :]  # (B, H/2, W/2, C)
        x3 = x[:, 1::2, 1::2, :]  # (B, H/2, W/2, C)
        x = torch.cat([x0, x1, x2, x3], -1)  # (B, H/2, W/2, 4*C)
        x = x.view(B, -1, 4 * C)  # (B, H/2*W/2, 4*C)

        x = self.norm(x)
        x = self.reduction(x)

        return x


class WindowAttention(nn.Module):
    """Window based multi-head self attention with relative position bias"""
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )

        # Get pair-wise relative position index
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: (num_windows*B, N, C)
            mask: (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # Add relative position bias
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    """MLP block with GELU activation"""
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block with shifted window attention"""
    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim, window_size=(self.window_size, self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x, H, W, attn_mask=None):
        """
        Args:
            x: (B, H*W, C)
            H, W: spatial resolution
            attn_mask: attention mask for shifted window
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # Partition windows
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # Window attention
        attn_windows = self.attn(x_windows, mask=attn_mask)

        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x


class BasicLayer(nn.Module):
    """A basic Swin Transformer layer for one stage"""
    def __init__(self, dim, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 downsample=None):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.window_size = window_size
        self.shift_size = window_size // 2

        # Build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop)
            for i in range(depth)])

        # Patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim)
        else:
            self.downsample = None

    def forward(self, x, H, W):
        """
        Args:
            x: (B, H*W, C)
            H, W: spatial resolution
        """
        # Calculate attention mask for SW-MSA
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        for blk in self.blocks:
            x = blk(x, H, W, attn_mask)
        
        if self.downsample is not None:
            x = self.downsample(x, H, W)
            H, W = H // 2, W // 2

        return x, H, W


class SwinTransformer(nn.Module):
    """
    Swin Transformer for permeability and dispersion predictions
    
    Args:
        img_size: Input image size (assumes square images)
        patch_size: Patch size (default 4x4)
        in_channels: Number of input channels
        num_classes: Number of output classes
        embed_dim: Patch embedding dimension (C in paper)
        depths: Depth of each Swin Transformer layer
        num_heads: Number of attention heads in different layers
        window_size: Window size
        mlp_ratio: MLP expansion ratio
        qkv_bias: If True, add learnable bias to query, key, value
        drop_rate: Dropout rate
        attn_drop_rate: Attention dropout rate
        task: Task type ('permeability' or 'dispersion')
        Pe_encoder: Peclet number encoding method
        include_direction: Whether to include direction encoding
    """
    def __init__(
        self,
        img_size=128,
        patch_size=4,
        in_channels=1,
        num_classes=4,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=4,
        mlp_ratio=4.,
        qkv_bias=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        task='permeability',
        Pe_encoder=None,
        include_direction=False
    ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio
        self.task = task

        # Calculate patches resolution
        patches_resolution = img_size // patch_size
        self.patches_resolution = patches_resolution
        
        # Validate and adjust window size
        if patches_resolution % window_size != 0:
            # Find the largest divisor of patches_resolution that is <= window_size
            for ws in range(window_size, 0, -1):
                if patches_resolution % ws == 0:
                    print(f"Warning: Adjusted window_size from {window_size} to {ws} to match patches_resolution={patches_resolution}")
                    window_size = ws
                    break
        self.window_size = window_size

        # Patch embedding
        self.patch_embed = PatchEmbedding(
            img_size=img_size, patch_size=patch_size, 
            in_channels=in_channels, embed_dim=embed_dim)
        
        num_patches = patches_resolution ** 2

        self.pos_drop = nn.Dropout(p=drop_rate)

        # Build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=self.window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None)
            self.layers.append(layer)

        self.norm = nn.LayerNorm(int(embed_dim * 2 ** (self.num_layers - 1)))
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        # Peclet and direction encoders
        self.Pe_encoder = Pe_encoder
        self.include_direction = include_direction

        pe_in_dims = {
            'straight': 1,
            'log': 1,
            'vector': 5,
        }

        extra_dim = 0
        fc_in = int(embed_dim * 2 ** (self.num_layers - 1))

        # Peclet encoder
        pe_dim = pe_in_dims.get(self.Pe_encoder)
        if pe_dim is not None:
            self.pe_mlp = nn.Sequential(
                nn.Linear(pe_dim, 16),
                nn.GELU(),
                nn.LayerNorm(16),
                nn.Linear(16, 16),
            )
            extra_dim += 16
        else:
            self.pe_mlp = None

        # Direction encoder
        if self.include_direction:
            dir_dim = 2
            self.dir_mlp = nn.Sequential(
                nn.Linear(dir_dim, 16),
                nn.GELU(),
                nn.LayerNorm(16),
                nn.Linear(16, 16),
            )
            extra_dim += 16
        else:
            self.dir_mlp = None

        # Classification head
        self.head = nn.Sequential(
            nn.LayerNorm(fc_in + extra_dim),
            nn.Linear(fc_in + extra_dim, num_classes),
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def pe_to_vector(self, Pe):
        """Convert Peclet number to a one-hot vector representation."""
        Pe = Pe.to(device=Pe.device)
        B = Pe.size(0)
        vector = torch.zeros((B, 5), device=Pe.device, dtype=Pe.dtype)
        for i in range(B):
            val = Pe[i]
            v = float(val.view(-1).item())
            if v < 1:
                vector[i, 0] = 1
            elif v == 10:
                vector[i, 1] = 1
            elif v == 50:
                vector[i, 2] = 1
            elif v == 100:
                vector[i, 3] = 1
            else:
                vector[i, 4] = 1
        return vector

    def forward(self, x, Pe=None, Direction=None):
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)
        x = self.pos_drop(x)

        H, W = self.patches_resolution, self.patches_resolution

        # Apply Swin Transformer layers
        for layer in self.layers:
            x, H, W = layer(x, H, W)

        x = self.norm(x)  # (B, H*W, C)
        x = self.avgpool(x.transpose(1, 2))  # (B, C, 1)
        x = torch.flatten(x, 1)  # (B, C)

        extra_feats = []

        # Peclet encoder
        if self.pe_mlp is not None:
            if Pe is None:
                raise ValueError("Pe must be provided when Pe_encoder is set")

            Pe = Pe.to(device=x.device, dtype=x.dtype)

            if self.Pe_encoder in ("straight", "log"):
                Pe = Pe.view(B, 1)
                if self.Pe_encoder == "log":
                    Pe = torch.log(Pe)

            elif self.Pe_encoder == "vector":
                if not (Pe.dim() == 2 and Pe.size(1) == 5):
                    Pe = self.pe_to_vector(Pe)

            Pe = self.pe_mlp(Pe)
            extra_feats.append(Pe)

        # Direction encoder
        if self.dir_mlp is not None and Direction is not None:
            Direction = Direction.to(device=x.device, dtype=x.dtype)
            direction = self.dir_mlp(Direction)
            extra_feats.append(direction)

        # Concatenate features
        if extra_feats:
            x = torch.cat([x] + extra_feats, dim=1)

        x = self.head(x)
        return x

    def get_num_params(self):
        """Get number of parameters"""
        return sum(p.numel() for p in self.parameters())


# Pre-defined Swin configurations
SWIN_CONFIGS = {
    'T': {'embed_dim': 96, 'depths': [2, 2, 6, 2], 'num_heads': [3, 6, 12, 24]},
    'S': {'embed_dim': 96, 'depths': [2, 2, 18, 2], 'num_heads': [3, 6, 12, 24]},
    'B': {'embed_dim': 128, 'depths': [2, 2, 18, 2], 'num_heads': [4, 8, 16, 32]},
    'L': {'embed_dim': 192, 'depths': [2, 2, 18, 2], 'num_heads': [6, 12, 24, 48]},
}


def load_swin_model(config_or_size='T', in_channels=1, task='permeability', 
                    pretrained_path=None, Pe_encoder=None, include_direction=False, **kwargs):
    """
    Flexible loader for Swin Transformer models.

    Accepts either:
    - a config dictionary, or
    - the original argument signature (size, in_channels, task, pretrained_path).

    Recognized config keys: `size`, `in_channels`, `task`, `pretrained_path`, `num_classes`,
    and other Swin kwargs (e.g. `img_size`, `patch_size`, `embed_dim`, `depths`, `num_heads`).
    """
    # Default mapping from task to num_classes
    if task == 'permeability':
        num_classes = 4
    elif task == 'dispersion':
        num_classes = 2
    else:
        raise ValueError(f"Unknown task: {task}. Supported tasks: ['permeability', 'dispersion']")

    # Extract from config dict or use provided args
    if isinstance(config_or_size, dict):
        cfg = config_or_size
        size = cfg.get('size', 'T')
        in_channels = cfg.get('in_channels', in_channels)
        task = cfg.get('task', task)
        pretrained_path = cfg.get('pretrained_path', pretrained_path)
        img_size = cfg.get('img_size', cfg.get('image_size', 128))
        patch_size = cfg.get('patch_size', 4)
        embed_dim = cfg.get('embed_dim', None)
        depths = cfg.get('depths', None)
        num_heads = cfg.get('num_heads', None)
        window_size = cfg.get('window_size', 4)
        mlp_ratio = cfg.get('mlp_ratio', 4.0)
        drop_rate = cfg.get('drop_rate', cfg.get('dropout', 0.0))
        attn_drop_rate = cfg.get('attn_drop_rate', 0.0)
    else:
        size = config_or_size
        img_size = kwargs.pop('img_size', 128)
        patch_size = kwargs.pop('patch_size', 4)
        embed_dim = kwargs.pop('embed_dim', None)
        depths = kwargs.pop('depths', None)
        num_heads = kwargs.pop('num_heads', None)
        window_size = kwargs.pop('window_size', 4)
        mlp_ratio = kwargs.pop('mlp_ratio', 4.0)
        drop_rate = kwargs.pop('drop_rate', kwargs.pop('dropout', 0.0))
        attn_drop_rate = kwargs.pop('attn_drop_rate', 0.0)

    # Resolve size configuration
    size_str = str(size)
    size_clean = size_str.lower()
    if size_clean.startswith("swin-"):
        size_clean = size_clean[5:]

    matched_key = None
    for k in SWIN_CONFIGS.keys():
        if k.lower() == size_clean:
            matched_key = k
            break

    if matched_key is None:
        if size_str in SWIN_CONFIGS:
            matched_key = size_str
        else:
            up = size_str.upper()
            if up in SWIN_CONFIGS:
                matched_key = up

    if matched_key:
        cfg_map = SWIN_CONFIGS[matched_key]
        if embed_dim is None:
            embed_dim = cfg_map.get('embed_dim', 96)
        if depths is None:
            depths = cfg_map.get('depths', [2, 2, 6, 2])
        if num_heads is None:
            num_heads = cfg_map.get('num_heads', [3, 6, 12, 24])
    else:
        # Fallback defaults
        embed_dim = embed_dim or 96
        depths = depths or [2, 2, 6, 2]
        num_heads = num_heads or [3, 6, 12, 24]

    # Create model
    model = SwinTransformer(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=in_channels,
        num_classes=num_classes,
        embed_dim=embed_dim,
        depths=depths,
        num_heads=num_heads,
        window_size=window_size,
        mlp_ratio=mlp_ratio,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        task=task,
        Pe_encoder=Pe_encoder,
        include_direction=include_direction
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


def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Example usage
    print("Testing Swin Transformer models...")
    sizes = []
    for size in ['T', 'S', 'B', 'L']:
        model = load_swin_model(size, in_channels=1, task='permeability')
        params = count_parameters(model)
        sizes.append(params)
        print(f"Swin-{size}: {params:,} parameters")

    # Test Swin-Tiny
    x = torch.randn(2, 1, 128, 128)
    model_tiny = load_swin_model(config_or_size='T', in_channels=1, task='permeability')
    output_tiny = model_tiny(x)
    print(f"\nSwin-Tiny output shape: {output_tiny.shape}")
    
    # Test Swin-Base for dispersion
    peclet = torch.tensor([[10.0], [20.0]])
    model_base = load_swin_model(config_or_size='B', in_channels=1, task='dispersion')
    output_base = model_base(x)
    print(f"Swin-Base (dispersion) output shape: {output_base.shape}")

    # Print number of parameters
    print(f"\nSwin-Tiny parameters: {model_tiny.get_num_params():,}")
    print(f"Swin-Base parameters: {model_base.get_num_params():,}")

    # Test Swin with Péclet encoding
    model_disp_pe = load_swin_model(config_or_size='S', in_channels=1, task='dispersion', Pe_encoder='log')
    output_disp_pe = model_disp_pe(x, Pe=peclet)
    print(f"Swin-Small (dispersion with Pe) output shape: {output_disp_pe.shape}")