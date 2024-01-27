""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F
import torch

from .unet_parts import *

class UNet_bottle_neck(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        #According to our experiment bilinear interpolation for upsampling seem to be more powerful.
        super(UNet_bottle_neck, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

        self.bottle_neck_pooling = nn.AdaptiveAvgPool2d((1,1))
        self.projector = nn.Linear(1024,1)

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
        bottle_neck_pooled = self.bottle_neck_pooling(x5) # global pool the bottleneck
        bottle_neck_pooled = torch.squeeze(bottle_neck_pooled,(-1,-2))
        predicted_similarity = self.projector(bottle_neck_pooled) # we use BCEWithLogitsLoss to enable safe autocast
        return logits, predicted_similarity