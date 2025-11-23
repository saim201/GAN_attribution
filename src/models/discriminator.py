import torch.nn as nn


def conv_block(in_c, out_c, k=4, s=2, p=1, bn=True):
    layers = [nn.Conv2d(in_c, out_c, k, s, p)]
    if bn:
        layers += [nn.BatchNorm2d(out_c)]
    layers += [nn.LeakyReLU(0.2, inplace=True)]
    return nn.Sequential(*layers)


class PatchDiscriminator(nn.Module):
    
    def __init__(self, in_ch=3):
        super().__init__()
        self.net = nn.Sequential(
            conv_block(in_ch, 64, bn=False),
            conv_block(64, 128),
            conv_block(128, 256),
            conv_block(256, 512, s=1),
            nn.Conv2d(512, 1, 4, 1, 1)
        )

    def forward(self, x):
        return self.net(x)
