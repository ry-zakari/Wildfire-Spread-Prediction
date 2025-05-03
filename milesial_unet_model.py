# Code from: https://github.com/milesial/Pytorch-UNet


""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)
    

class AttentionUp(nn.Module):
    def __init__(self):
        pass
    
    def forward(self, x, g):
        pass


class Up(nn.Module):
    """Up-conv then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        )
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


 

""" Full assembly of the parts to form the complete network """


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        self.down4 = (Down(512, 1024))
        self.up1 = (Up(1024, 512))
        self.up2 = (Up(512, 256))
        self.up3 = (Up(256, 128))
        self.up4 = (Up(128, 64))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)



import torch
import torch.nn as nn
import torch.nn.functional as F

# Define ASPC Block with Residual Links
class ASPC_Block(nn.Module):
    def __init__(self, in_ch, out_ch, dropout_rate=0.3):
        super(ASPC_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=2, dilation=2)
        self.conv3 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=3, dilation=3)
        self.conv4 = nn.Conv2d(in_ch, out_ch, kernel_size=1)

        # Optional projection layer for residual link if in_ch != out_ch
        self.projection = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else None

        self.batch_norm = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout_rate)

    def forward(self, x):
        # Apply atrous convolutions
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)

        # Combine outputs along the channel dimension
        out = x1 + x2 + x3 + x4

        # Add the residual connection
        residual = self.projection(x) if self.projection else x
        out = out + residual

        # Apply batch normalization, dropout, and activation
        out = self.batch_norm(out)
        out = self.dropout(out)
        return self.relu(out)


# Attention Mechanism for skip connections
class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi

# Up-convolution block
class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.up(x)

# Modified AttU-Net with ASPC Blocks
class APAU_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super(APAU_Net, self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder with ASPC Blocks
        self.Conv1 = ASPC_Block(in_ch=img_ch, out_ch=64)
        self.Conv2 = ASPC_Block(in_ch=64, out_ch=128)
        self.Conv3 = ASPC_Block(in_ch=128, out_ch=256)
        self.Conv4 = ASPC_Block(in_ch=256, out_ch=512)
        self.Conv5 = ASPC_Block(in_ch=512, out_ch=1024)

        # Decoder path with attention mechanism
        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Att5 = AttentionBlock(F_g=512, F_l=512, F_int=256)
        self.Up_conv5 = ASPC_Block(in_ch=1024, out_ch=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Att4 = AttentionBlock(F_g=256, F_l=256, F_int=128)
        self.Up_conv4 = ASPC_Block(in_ch=512, out_ch=256)
        
        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Att3 = AttentionBlock(F_g=128, F_l=128, F_int=64)
        self.Up_conv3 = ASPC_Block(in_ch=256, out_ch=128)
        
        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Att2 = AttentionBlock(F_g=64, F_l=64, F_int=32)
        self.Up_conv2 = ASPC_Block(in_ch=128, out_ch=64)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # Encoding path
        x1 = self.Conv1(x)
        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # Decoding path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        # Output layer
        d1 = self.Conv_1x1(d2)

        return d1




    #################################################################
class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)))
        
        for rate in atrous_rates:
            modules.append(nn.Sequential(
                nn.Conv2d(in_channels, 256, 3, padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)))
        
        # Add global average pooling branch
        modules.append(nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)))
        
        self.convs = nn.ModuleList(modules)
        
        # Calculate total input channels for projection
        self.project = nn.Sequential(
            nn.Conv2d(len(modules) * 256, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5))

    def forward(self, x):
        res = []
        for conv in self.convs[:-1]:  # Apply all except the global pooling branch
            res.append(conv(x))
        
        # Handle global pooling branch separately
        global_feat = self.convs[-1](x)
        global_feat = F.interpolate(global_feat, size=x.shape[2:], mode='bilinear', align_corners=False)
        res.append(global_feat)
        
        res = torch.cat(res, dim=1)
        return self.project(res)

class DeepLabV3(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, backbone='resnet50'):
        super(DeepLabV3, self).__init__()
        # Simplified ResNet backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(n_channels, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        self.aspp = ASPP(512, [6, 12, 18])
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, n_classes, 1)
        )

    def forward(self, x):
        size = x.shape[2:]
        x = self.backbone(x)
        x = self.aspp(x)
        x = self.decoder(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)