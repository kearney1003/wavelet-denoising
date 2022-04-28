from models.architecs import *
import torch.nn as nn


class UNet_with_SPADE(nn.Module):
    def __init__(self, args):
        super(UNet_with_SPADE, self).__init__()
        inc_filters = args.spade_channels
        self.inc = DoubleConv(4, inc_filters)

        self.s0 = SPADE(inc_filters)
        self.relu = nn.ReLU(inplace=True)

        self.down1 = Down(inc_filters, inc_filters*2)
        self.s1_down = SPADE(inc_filters*2)
        
        self.down2 = Down(inc_filters*2, inc_filters*4)
        self.s2_down = SPADE(inc_filters*4)

        self.down3 = Down(inc_filters*4, inc_filters*8)
        self.s3_down = SPADE(inc_filters*8)

        self.mic = MidConv(inc_filters*8, inc_filters*8)
        
        self.up1 = Up(inc_filters*20, inc_filters*4)
        self.s1_up = SPADE(inc_filters*4)

        self.up2 = Up(inc_filters*6, inc_filters*4)
        self.s2_up = SPADE(inc_filters*4)

        self.up3 = Up(inc_filters*5, inc_filters*4)
        self.s3_up = SPADE(inc_filters*4)

        self.outc1 = nn.Conv3d(inc_filters*4, inc_filters, kernel_size=3, padding=1)
        self.s_out = SPADE(inc_filters)
        
        self.outc2 = nn.Conv3d(inc_filters, 1, kernel_size=3, padding=1)
        
        self.dropout = nn.Dropout3d(args.dropout)


    def forward(self, x):
        x1 = self.inc(x)
        x1 = self.s0(x1, x)
        x1 = self.relu(x1)

        x2 = self.down1(x1)
        x2 = self.s1_down(x2, x)
        x2 = self.relu(x2)
        
        x3 = self.down2(x2)
        x3 = self.s2_down(x3, x)
        x3 = self.relu(x3)

        x4 = self.down3(x3)
        x4 = self.s3_down(x4, x)
        x4 = self.relu(x4)

        x5 = self.mic(x4,x4)

        x6 = self.up1(x5, x3)
        x6 = self.s1_up(x6, x)
        x6 = self.relu(x6)

        x6 = self.up2(x6, x2)
        x6 = self.s2_up(x6, x)
        x6 = self.relu(x6)

        x6 = self.up3(x6, x1)
        x6 = self.s3_up(x6, x)
        x6 = self.relu(x6)
        
        x6 = self.outc1(x6)
        x6 = self.s_out(x6, x)
        x6 = self.relu(x6)
        
        out = self.outc2(x6)
        out = self.dropout(out)
        return out
    

class UNet(nn.Module):
    def __init__(self, args):
        super(UNet, self).__init__()
        inc_filters = args.spade_channels

        self.inc = DoubleConv(4, inc_filters)
        self.relu = nn.ReLU(inplace=True)

        self.down1 = Down(inc_filters, inc_filters * 2)

        self.down2 = Down(inc_filters * 2, inc_filters * 4)

        self.down3 = Down(inc_filters * 4, inc_filters * 8)

        self.mic = MidConv(inc_filters * 8, inc_filters * 8)

        self.up1 = Up(inc_filters * 20, inc_filters * 4)

        self.up2 = Up(inc_filters * 6, inc_filters * 4)

        self.up3 = Up(inc_filters * 5, inc_filters * 4)

        self.outc1 = nn.Conv3d(inc_filters * 4, inc_filters, kernel_size=3, padding=1)

        self.outc2 = nn.Conv3d(inc_filters, 1, kernel_size=3, padding=1)

        self.dropout = nn.Dropout3d(args.dropout)

    def forward(self, x):
        x1 = self.inc(x)
        x1 = self.relu(x1)

        x2 = self.down1(x1)
        x2 = self.relu(x2)

        x3 = self.down2(x2)
        x3 = self.relu(x3)

        x4 = self.down3(x3)
        x4 = self.relu(x4)

        x5 = self.mic(x4, x4)

        x6 = self.up1(x5, x3)
        x6 = self.relu(x6)

        x6 = self.up2(x6, x2)
        x6 = self.relu(x6)
        x6 = self.up3(x6, x1)
        x6 = self.relu(x6)

        x6 = self.outc1(x6)
        x6 = self.relu(x6)

        out = self.outc2(x6)
        out = self.dropout(out)
        return out


class Dncnn(nn.Module):
    def __init__(self, args):
        super(Dncnn, self).__init__()
        inc_filters = args.denoise_in
        mid_filters = args.denoise_channels
        out_filters = args.denoise_out

        self.inc = nn.Conv3d(in_channels=inc_filters, out_channels=mid_filters,
                             kernel_size=(3, 3, 3), padding=(1, 1, 1), stride=(1, 1, 1))
        self.conv1 = ConvBlock(mid_filters, mid_filters)
        self.conv2 = ConvBlock(mid_filters, mid_filters)
        self.conv3 = ConvBlock(mid_filters, mid_filters)
        self.conv4 = ConvBlock(mid_filters, mid_filters)
        self.conv5 = ConvBlock(mid_filters, mid_filters)
        self.conv6 = ConvBlock(mid_filters, mid_filters)
        self.conv7 = ConvBlock(mid_filters, mid_filters)
        self.conv8 = ConvBlock(mid_filters, mid_filters)
        self.conv9 = ConvBlock(mid_filters, mid_filters)
        self.conv10 = ConvBlock(mid_filters, mid_filters)
        self.outc = nn.Conv3d(in_channels=mid_filters, out_channels=out_filters,
                              kernel_size=(3, 3, 3), padding=(1, 1, 1), stride=(1, 1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout3d(args.dropout)

    def forward(self, x):
        x1 = self.inc(x)
        x1 = self.relu(x1)
        x1 = self.conv1(x1)
        x1 = self.conv2(x1)
        x1 = self.conv3(x1)
        x1 = self.conv4(x1)
        x1 = self.conv5(x1)
        x1 = self.conv6(x1)
        x1 = self.conv7(x1)
        x1 = self.conv8(x1)
        x1 = self.conv9(x1)
        x1 = self.conv10(x1)
        x1 = self.dropout(x1)
        x1 = self.outc(x1)+x
        return x1


class DeepCNN(nn.Module):
    def __init__(self, mid_channels,in_channels=9, out_channels=7):

        super(DeepCNN, self).__init__()
        self.inc = nn.Conv3d(in_channels=in_channels, out_channels=mid_channels,
                             kernel_size=(3, 3, 3), padding=(1, 1, 1), stride=(1, 1, 1))
        self.conv1 = ConvBlock(mid_channels, mid_channels)
        self.conv2 = ConvBlock(mid_channels, mid_channels)
        self.conv3 = ConvBlock(mid_channels, mid_channels)
        self.conv4 = ConvBlock(mid_channels, mid_channels)
        self.conv5 = ConvBlock(mid_channels, mid_channels)

        self.outc = nn.Conv3d(in_channels=mid_channels, out_channels=out_channels,
                              kernel_size=(3, 3, 3), padding=(1, 1, 1), stride=(1, 1, 1))
        self.relu = nn.ReLU(inplace=True)
        # self.dropout = nn.Dropout3d(0.1)

    def forward(self, x):
        x1 = self.inc(x)
        x1 = self.relu(x1)

        x1 = self.conv1(x1)
        x1 = self.conv2(x1)
        x1 = self.conv3(x1)
        x1 = self.conv4(x1)
        x1 = self.conv5(x1)

        x1 = self.outc(x1) + x[:, :7, :, :, :]
        return x1


