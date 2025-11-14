import torch
import torch.nn as nn

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.model(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, use_dropout=False):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if use_dropout:
            layers.append(nn.Dropout(0.5))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class GeneratorUNet(nn.Module):
    def __init__(self, input_nc=9, output_nc=3):
        super().__init__()

        # Encoder
        self.down1 = Down(input_nc, 64)      # 256 -> 128
        self.down2 = Down(64, 128)           # 128 -> 64
        self.down3 = Down(128, 256)          # 64 -> 32
        self.down4 = Down(256, 512)          # 32 -> 16
        self.down5 = Down(512, 512)          # 16 -> 8
        self.down6 = Down(512, 512)          # 8 -> 4

        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, 4, 2, 1),
            nn.ReLU(True)
        )                                      # 4 -> 2

        # Decoder (note 6 upsamplings now)
        self.up1 = Up(512, 512, use_dropout=True)      # 2 -> 4
        self.up2 = Up(512+512, 512, use_dropout=True)  # 4 -> 8
        self.up3 = Up(512+512, 256)                    # 8 -> 16
        self.up4 = Up(256+512, 128)                    # 16 -> 32
        self.up5 = Up(128+256, 64)                     # 32 -> 64
        self.up6 = Up(64+128, 32)                      # 64 -> 128

        self.final = nn.Sequential(
            nn.ConvTranspose2d(32+64, output_nc, 4, 2, 1),  # 128 -> 256
            nn.Sigmoid()       # output in [0,1]
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)

        bn = self.bottleneck(d6)

        u1 = self.up1(bn)
        u2 = self.up2(torch.cat([u1, d6], dim=1))
        u3 = self.up3(torch.cat([u2, d5], dim=1))
        u4 = self.up4(torch.cat([u3, d4], dim=1))
        u5 = self.up5(torch.cat([u4, d3], dim=1))
        u6 = self.up6(torch.cat([u5, d2], dim=1))

        out = self.final(torch.cat([u6, d1], dim=1))
        return out
