import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Standard Residual Block to preserve spatial features"""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv_path = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        # Identity shortcut to match dimensions
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        return torch.relu(self.conv_path(x) + self.shortcut(x))

class SketchEncoder(nn.Module):
    def __init__(self , latent_dim = 512):
        super(SketchEncoder, self).__init__()

        # Initial Processing (256x256)
        self.start = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2)
        )

        # Downsampling Stages
        self.layer1 = ResidualBlock(64, 128, stride=2)  # Out: 128x128
        self.layer2 = ResidualBlock(128, 256, stride=2)  # Out: 64x64
        self.layer3 = ResidualBlock(256, 512, stride=2)  # Out: 32x32

        # Refinement Layer
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, latent_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(latent_dim),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        """
        Input:  (Batch, 1, 256, 256)
        Output: (Batch, 512, 32, 32) -> The Spatial Latent Space
        """
        x = self.start(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        z_sketch = self.bottleneck(x)
        return z_sketch
