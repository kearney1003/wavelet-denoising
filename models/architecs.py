import torch
import torch.nn as nn
import torch.nn.functional as F




class SPADE(nn.Module):
    def __init__(self, n_channel):
        super().__init__()

        self.param_free_norm = nn.BatchNorm3d(n_channel)
        ks = 3
        nhidden = 64
        pw = 1
        input_channel = 4
        self.mlp_shared = nn.Sequential(
            nn.Conv3d(input_channel, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv3d(nhidden, n_channel, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv3d(nhidden, n_channel, kernel_size=ks, padding=pw)

    def forward(self, x, segmap):
        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels,
                      kernel_size=(3, 3, 3), padding=(1, 1, 1), stride=(1, 1, 1)),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)



class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, stride=1),

        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class MidConv(nn.Module):
    """Upscaling then conv"""

    def __init__(self, in_channels, out_channels):
        super(MidConv, self).__init__()

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x1, x2):
        # input is BCDHW
        diffY = x2.shape[2] - x1.shape[2]
        diffX = x2.shape[3] - x1.shape[3]
        diffZ = x2.shape[4] - x1.shape[4]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffZ // 2, diffZ - diffZ // 2])

        x = torch.cat([x2, x1], dim=1)
        return x


class Up(nn.Module):
    """Upscaling then conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is BCDHW
        diffX = x2.shape[2] - x1.shape[2]
        diffY = x2.shape[3] - x1.shape[3]
        diffZ = x2.shape[4] - x1.shape[4]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffZ // 2, diffZ - diffZ // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
