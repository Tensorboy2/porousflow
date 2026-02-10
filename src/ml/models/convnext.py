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
        # layer-scale for depthwise conv residual (channels-last shape)
        self.gamma_dw = nn.Parameter(layer_scale_init_value * torch.ones(1, 1, 1, dim), requires_grad=True)
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
        # mlp = self.gamma_mlp * mlp
        x = shortcut + mlp                                      # second residual

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
        # if self.gamma is not None:
        #     x = self.gamma * x
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
        # if self.gamma is not None:
        #     x = self.gamma * x
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


# ------------------------------------------------------------
# CONDITION ENCODER (Pe + Direction)
# ------------------------------------------------------------

class ConditionEncoder(nn.Module):
    def __init__(self, pe_mode=None, include_direction=False, out_dim=256):
        super().__init__()
        self.pe_mode = pe_mode
        self.include_direction = include_direction
        
        if pe_mode == "straight":
            self.pe_mlp = self._mlp(1, out_dim)
        elif pe_mode == "log":
            self.pe_mlp = self._mlp(1, out_dim)
        elif pe_mode == "vector":
            self.pe_mlp = self._mlp(5, out_dim)
        else:
            self.pe_mlp = None
        
        self.dir_mlp = self._mlp(2, out_dim) if include_direction else None
    
    def _mlp(self, inp, out):
        return nn.Sequential(
            nn.Linear(inp, out // 2),
            nn.GELU(),
            nn.LayerNorm(out // 2),
            nn.Linear(out // 2, out)
        )
    
    def pe_to_vector(self, Pe):
        B = Pe.size(0)
        vec = torch.zeros(B, 5, device=Pe.device, dtype=Pe.dtype)
        bins = [1, 10, 50, 100]
        
        for i in range(B):
            v = float(Pe[i])
            idx = sum(v >= b for b in bins)
            vec[i, idx] = 1
        
        return vec
    
    def encode_pe(self, Pe):
        if self.pe_mode == "log":
            Pe = torch.log(Pe.clamp(min=1e-8))
        elif self.pe_mode == "vector":
            Pe = self.pe_to_vector(Pe)
        
        return self.pe_mlp(Pe)
    
    def forward(self, Pe=None, Direction=None):
        outs = []
        
        if self.pe_mlp is not None and Pe is not None:
            Pe = Pe.view(-1, 1)
            outs.append(self.encode_pe(Pe))
        
        if self.dir_mlp is not None and Direction is not None:
            outs.append(self.dir_mlp(Direction))
        
        if len(outs) == 0:
            return None
        
        return torch.cat(outs, dim=-1)


# ------------------------------------------------------------
# TOKEN MIXER (Transformer optional)
# ------------------------------------------------------------

class TokenMixer(nn.Module):
    def __init__(self, dim, use_transformer=False, layers=2):
        super().__init__()
        self.use_transformer = use_transformer
        
        if use_transformer:
            layer = nn.TransformerEncoderLayer(
                d_model=dim, nhead=4, batch_first=True
            )
            self.transformer = nn.TransformerEncoder(layer, layers)
            self.pos_emb = None
    
    def forward(self, feat):
        # feat: [B, C, H, W]
        
        if not self.use_transformer:
            return feat.mean(dim=[2, 3])  # GAP
        
        B, C, H, W = feat.shape
        tokens = feat.flatten(2).transpose(1, 2)  # [B, HW, C]
        
        if self.pos_emb is None or self.pos_emb.size(1) != H * W:
            self.pos_emb = nn.Parameter(
                torch.randn(1, H * W, C, device=feat.device) * 0.02
            )
        
        tokens = tokens + self.pos_emb
        tokens = self.transformer(tokens)
        
        return tokens.mean(dim=1)


# ------------------------------------------------------------
# CLEAN CONVNEXT MODEL
# ------------------------------------------------------------

class ConvNeXt(nn.Module):
    def __init__(
        self,
        size="tiny",
        version="v1",
        in_channels=1,
        num_classes=4,
        pe_encoder=None,
        include_direction=True,
        use_transformer=False
    ):
        super().__init__()
        
        config = CONVNEXT_CONFIGS[size]
        depths, dims = config["depths"], config["dims"]
        
        block_map = {
            "v1": ConvNeXtBlockV1,
            "v2": ConvNeXtBlockV2,
            "rms": ConvNeXtBlockRMS
        }
        block = block_map[version]
        
        # Backbone
        self.encoder = ConvNeXtEncoder(in_channels, depths, dims, block)
        feat_dim = dims[-1]
        
        # Conditioning encoder
        self.cond_encoder = ConditionEncoder(
            pe_mode=pe_encoder,
            include_direction=include_direction,
            out_dim=feat_dim
        )
        
        # Token mixer
        self.token_mixer = TokenMixer(
            dim=feat_dim,
            use_transformer=use_transformer
        )
        
        # Conditioning vector dimension (depends on enabled PE and direction)
        cond_dim = 0
        if pe_encoder is not None:
            cond_dim += feat_dim
        if include_direction:
            cond_dim += feat_dim

        # If using transformer (additive fusion) but condition vector doesn't
        # match the pooled feature size, create a small projection to align
        # dimensions.
        self.cond_proj = None
        if use_transformer and cond_dim > 0 and cond_dim != feat_dim:
            self.cond_proj = nn.Linear(cond_dim, feat_dim)

        # Fusion dimension logic for non-transformer (concatenation)
        fusion_dim = feat_dim
        if not use_transformer:
            if pe_encoder is not None:
                fusion_dim += feat_dim
            if include_direction:
                fusion_dim += feat_dim
        
        # Head
        self.head = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, num_classes)
        )
    
    def forward(self, x, Pe=None, Direction=None):
        feat = self.encoder(x)
        pooled = self.token_mixer(feat)
        
        cond = self.cond_encoder(Pe, Direction)
        
        # Transformer mode → additive fusion
        if self.token_mixer.use_transformer:
            if cond is not None:
                # If the condition vector size doesn't match pooled features,
                # project it to the pooled size when a projection exists.
                if cond.size(-1) != pooled.size(-1):
                    if hasattr(self, 'cond_proj') and self.cond_proj is not None:
                        cond = self.cond_proj(cond)
                    else:
                        # Fallback: trim or pad (trim here) to match pooled size
                        cond = cond[:, :pooled.size(-1)]
                pooled = pooled + cond
        
        # CNN mode → concatenative fusion
        else:
            if cond is not None:
                pooled = torch.cat([pooled, cond], dim=-1)
        
        return self.head(pooled)

def load_convnext_model(config_or_version='v1', 
                        size='tiny', 
                        in_channels=1, 
                        task='permeability',  
                        pretrained_path = None, 
                        Pe_encoder=None,
                        include_direction=False):
    """
    Flexible loader for ConvNeXt models.

    Accepts either a config dictionary or the traditional signature.
    Recognized config keys: `version`, `size`, `in_channels`, `num_classes`, `pretrained_path`.
    """
    if task == 'permeability':
        num_classes = 4
    elif task == 'dispersion':
        num_classes = 2
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

    model = ConvNeXt(version=version, 
                     size=size, 
                     in_channels=in_channels, 
                     num_classes=num_classes, 
                    #  task=task, 
                     pe_encoder=Pe_encoder,
                     include_direction=include_direction)

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
        sizes=[]
        for size in ['atto', 'femto', 'pico', 'nano', 'tiny', 'small', 'base', 'large']:
            # try:
            model = load_convnext_model(config_or_version=version, size=size, in_channels=1)
            # out = model(x)
            params = count_parameters(model)
            sizes.append(params)
            #     print(f"ConvNeXt-{version}-{size}: output shape {out.shape}, params {params:,}")
            # except Exception as e:
            #     print(f"Error with ConvNeXt-{version}-{size}: {e}")
        print(f"{version}: {sizes}")
    # # Test large model
    # try:
    #     model = load_convnext_model(config_or_version='v2', size='large', in_channels=1)
    #     params = count_parameters(model)
    #     print(f"\nConvNeXt-v2-large param count: {params:,}")
    # except Exception as e:
    #     print(f"Error with ConvNeXt-v2-large: {e}")

    # Test with Peclet number input
    # try:
    for encoder in [None, 'straight', 'log', 'vector']:
        model = load_convnext_model(config_or_version='v1', size='tiny', in_channels=1, task='dispersion', Pe_encoder=encoder)
        Pe = torch.tensor([[10.0],[100.0]])
        out = model(x, Pe=Pe)
        print(f"\nConvNeXt-v1-tiny with Peclet encoder '{encoder}': output shape {out.shape}")
    # model = load_convnext_model(config_or_version='v1', size='tiny', in_channels=1, task='dispersion', Pe_encoder='vector')
    # Pe = torch.tensor([[1,0,0,0,0],[0,1,0,0,0]])
    # out = model(x, Pe=Pe)
    # print(f"\nConvNeXt-v1-tiny with Peclet encoder 'vector': output shape {out.shape}")
    # model = load_convnext_model(config_or_version='v1', size='tiny', in_channels=1, task='dispersion', Pe_encoder='straight')
    # Pe = torch.tensor([[10.0],[100.0]])
    # out = model(x, Pe=Pe)
    # print(f"\nConvNeXt-v1-tiny with Peclet input: output shape {out.shape}")
    # except Exception as e:
    #     print(f"Error with ConvNeXt-v1-tiny with Peclet input: {e}")