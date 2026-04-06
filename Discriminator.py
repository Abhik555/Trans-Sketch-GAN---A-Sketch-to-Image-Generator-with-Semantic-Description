import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm


class LiteDiscriminatorBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=True):
        super().__init__()
        self.downsample = downsample
        self.shortcut = nn.Sequential()
        if in_channels != out_channels or downsample:
            stride = 2 if downsample else 1
            self.shortcut = spectral_norm(nn.Conv2d(in_channels, out_channels, 1, stride=stride))

        self.conv_path = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, out_channels, 3, padding=1)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2d(out_channels, out_channels, 3, padding=1, stride=2 if downsample else 1)),
        )
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.act(self.conv_path(x) + self.shortcut(x))


class LiteMultiModalDiscriminator(nn.Module):
    def __init__(self, text_dim=768):
        super().__init__()

        self.initial = nn.Sequential(
            spectral_norm(nn.Conv2d(4, 64, 3, padding=1)),
            nn.LeakyReLU(0.2)
        )

        self.blocks = nn.Sequential(
            LiteDiscriminatorBlock(64, 128),  # 128x128
            LiteDiscriminatorBlock(128, 256),  # 64x64
            LiteDiscriminatorBlock(256, 512),  # 32x32
            LiteDiscriminatorBlock(512, 512),  # 16x16
            LiteDiscriminatorBlock(512, 512, downsample=False)  # 16x16 refined
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Projection for text matching
        self.text_projector = spectral_norm(nn.Linear(text_dim, 512))
        self.final_linear = spectral_norm(nn.Linear(512, 1))

    def forward(self, image, sketch, text_embedding):
        x = torch.cat([image, sketch], dim=1)
        x = self.initial(x)
        features = self.blocks(x)

        phi = self.global_pool(features).view(features.size(0), -1)

        # Standard GAN score
        out_uncond = self.final_linear(phi)

        # Projection Discriminator logic (Text Alignment)
        proj_text = self.text_projector(text_embedding)
        out_cond = torch.sum(phi * proj_text, dim=1, keepdim=True)

        return out_uncond + out_cond