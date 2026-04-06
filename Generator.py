import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaIN(nn.Module):
    def __init__(self, channels, latent_dim=512):
        super().__init__()
        self.norm = nn.InstanceNorm2d(channels)
        self.style_scale = nn.Linear(latent_dim, channels)
        self.style_shift = nn.Linear(latent_dim, channels)

    def forward(self, x, z_text):
        norm_x = self.norm(x)
        gamma = self.style_scale(z_text).unsqueeze(2).unsqueeze(3)
        beta = self.style_shift(z_text).unsqueeze(2).unsqueeze(3)
        return gamma * norm_x + beta

class ResBlockAdaIN(nn.Module):
    """A Residual Block that incorporates Text Conditioning via AdaIN"""
    def __init__(self, channels, text_dim=768):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.adain1 = AdaIN(channels, text_dim)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.adain2 = AdaIN(channels, text_dim)
        self.leaky = nn.LeakyReLU(0.2)

    def forward(self, x, z_text):
        residual = x
        out = self.leaky(self.adain1(self.conv1(x), z_text))
        out = self.adain2(self.conv2(out), z_text)
        return self.leaky(out + residual)

class Generator(nn.Module):
    def __init__(self , sketch_latent_dim=512 , text_dim=768):
        super(Generator, self).__init__()

        # 1. Processing the 32x32 Input
        self.layer1 = ResBlockAdaIN(512, text_dim)

        # 2. Upsample to 64x64
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = nn.Conv2d(512, 256, 3, padding=1)
        self.res1 = ResBlockAdaIN(256, text_dim)

        # 3. Upsample to 128x128
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv2 = nn.Conv2d(256, 128, 3, padding=1)
        self.res2 = ResBlockAdaIN(128, text_dim)

        # 4. Upsample to 256x256
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv3 = nn.Conv2d(128, 64, 3, padding=1)
        self.res3 = ResBlockAdaIN(64, text_dim)

        # 5. Output to RGB
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, z_sketch, z_text):
        """
        z_sketch: (B, 512, 32, 32) -> From your Sketch Encoder
        z_text:   (B, 512)         -> From your CLIP Encoder
        """
        # Block 1: 32x32
        x = self.layer1(z_sketch, z_text)

        # Block 2: 64x64
        x = self.conv1(self.up1(x))
        x = self.res1(x, z_text)

        # Block 3: 128x128
        x = self.conv2(self.up2(x))
        x = self.res2(x, z_text)

        # Block 4: 256x256
        x = self.conv3(self.up3(x))
        x = self.res3(x, z_text)

        return self.final_conv(x)