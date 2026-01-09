"""
resnet.py

This file contains the corrected implementation of a ResNet model for regression tasks.
"""
import torch
import torch.nn as nn
torch.manual_seed(0)
import os
path = os.path.dirname(__file__)

class BasicBlock(nn.Module):
    """Basic ResNet block for ResNet-18 and ResNet-34"""
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, 
                              padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, 
                              padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = downsample

    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    """Bottleneck block for ResNet-50, ResNet-101, and ResNet-152"""
    expansion = 4
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        # 1x1 conv to reduce dimensions
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # 3x3 conv
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride,
                              padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 1x1 conv to expand dimensions
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, in_channels=3, num_classes=1000, task='permeability'):
        super(ResNet, self).__init__()
        self.in_channels = 64
        
        # Initial conv layer
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet layers
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Final layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.task = task

        fc_in = 512 * block.expansion
        if self.task == 'dispersion':
            fc_in += 1
        elif self.task == 'dispersion_direction':
            fc_in += 2

        self.fc = nn.Linear(fc_in, num_classes)
        
        # Initialize weights
        self._initialize_weights()

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, Pe=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # Handle task-specific extra inputs
        if self.task == 'dispersion':
            if Pe is None:
                raise ValueError("Pe must be provided for dispersion task")
            Pe = torch.ones(x.size(0), 1, device=x.device) * Pe
            x = torch.cat([x, Pe], dim=1)
        elif self.task == 'dispersion_direction':
            if Pe is None:
                raise ValueError("Pe (and direction) must be provided for dispersion_direction task")
            # Expect Pe to be a tensor or value with two components per sample
            # If scalar provided, replicate to two values
            if isinstance(Pe, (int, float)):
                Pe = torch.ones(x.size(0), 2, device=x.device) * Pe
            elif isinstance(Pe, torch.Tensor) and Pe.dim() == 1:
                Pe = Pe.unsqueeze(1)
            x = torch.cat([x, Pe], dim=1)

        x = self.fc(x)
        return x


def resnet18(in_channels=3, num_classes=1000, task='permeability'):
    """ResNet-18 model"""
    return ResNet(BasicBlock, [2, 2, 2, 2], in_channels, num_classes, task=task)


def resnet34(in_channels=3, num_classes=1000, task='permeability'):
    """ResNet-34 model"""
    return ResNet(BasicBlock, [3, 4, 6, 3], in_channels, num_classes, task=task)


def resnet50(in_channels=3, num_classes=1000, task='permeability'):
    """ResNet-50 model"""
    return ResNet(Bottleneck, [3, 4, 6, 3], in_channels, num_classes, task=task)


def resnet101(in_channels=3, num_classes=1000, task='permeability'):
    """ResNet-101 model"""
    return ResNet(Bottleneck, [3, 4, 23, 3], in_channels, num_classes, task=task)


def resnet152(in_channels=3, num_classes=1000, task='permeability'):
    """ResNet-152 model"""
    return ResNet(Bottleneck, [3, 8, 36, 3], in_channels, num_classes, task=task)


def load_resnet_model(config_or_size='18', in_channels=1, num_classes=4, pretrained_path: str = None, task: str = 'permeability', **kwargs):
    """
    Flexible loader for ResNet models.

    Accepts either a config dictionary (from generated YAML) or the original args.

    Recognized config keys: `size` (one of '18','34','50','101','152'), `in_channels`,
    `num_classes`, and `pretrained_path`.
    """
    if task == 'permeability':
        num_classes = 4
    elif task == 'dispersion':
        num_classes = 8
    else:
        raise ValueError(f"Unknown task: {cfg['task']}. Supported tasks: ['permeability', 'dispersion']")
    
    # Parse config dict if provided
    if isinstance(config_or_size, dict):
        cfg = config_or_size
        size = str(cfg.get('size', '18'))
        in_channels = cfg.get('in_channels', in_channels)
        pretrained_path = cfg.get('pretrained_path', pretrained_path)
    else:
        size = str(config_or_size)
    
    # Create model
    if size == '18':
        model = resnet18(in_channels, num_classes, task=task)
    elif size == '34':
        model = resnet34(in_channels, num_classes, task=task)
    elif size == '50':
        model = resnet50(in_channels, num_classes, task=task)
    elif size == '101':
        model = resnet101(in_channels, num_classes, task=task)
    elif size == '152':
        model = resnet152(in_channels, num_classes, task=task)
    else:
        raise ValueError(f"Invalid size '{size}'. Choose from '18', '34', '50', '101', or '152'.")

    # Optionally load pretrained weights
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
    print("Testing ResNet models...")
    
    # Test ResNet-18 (BasicBlock)
    x = torch.randn(2, 1, 128, 128)
    model18 = load_resnet_model(size='18', in_channels=1, num_classes=4)
    output18 = model18(x)
    print(f"ResNet-18 output shape: {output18.shape}")
    
    # Test ResNet-50 (Bottleneck)
    model50 = load_resnet_model(size='50', in_channels=1, num_classes=4)
    output50 = model50(x)
    print(f"ResNet-50 output shape: {output50.shape}")
    
    # Test with grayscale input
    x_gray = torch.randn(2, 1, 128, 128)
    model_gray = load_resnet_model(size='34', in_channels=1, num_classes=4)
    output_gray = model_gray(x_gray)
    print(f"ResNet-34 (grayscale) output shape: {output_gray.shape}")
    
    # Print parameter counts
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nParameter counts:")
    print(f"ResNet-18: {count_parameters(model18):,}")
    print(f"ResNet-50: {count_parameters(model50):,}")