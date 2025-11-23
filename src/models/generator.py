import torch
import torch.nn as nn
import torch.nn.functional as F


class UNetRes(nn.Module):
    
    def __init__(self, in_ch=3, base=32, scale=0.20):
        super().__init__()
        self.scale = scale
        
        def C(in_c, out_c, k=3, s=1, p=1):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, k, s, p),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )
        
        # Encoder
        self.enc1 = nn.Sequential(C(in_ch, base), C(base, base))
        self.enc2 = nn.Sequential(C(base, base*2, s=2), C(base*2, base*2))
        self.enc3 = nn.Sequential(C(base*2, base*4, s=2), C(base*4, base*4))
        self.enc4 = nn.Sequential(C(base*4, base*8, s=2), C(base*8, base*8))
        
        # Decoder
        self.dec3 = nn.Sequential(C(base*8+base*4, base*4), C(base*4, base*4))
        self.dec2 = nn.Sequential(C(base*4+base*2, base*2), C(base*2, base*2))
        self.dec1 = nn.Sequential(C(base*2+base, base), C(base, base))
        self.outc = nn.Conv2d(base, 3, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        
        d3 = F.interpolate(e4, scale_factor=2, mode='bilinear', align_corners=False)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = F.interpolate(d3, scale_factor=2, mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        r = torch.tanh(self.outc(d1)) * self.scale
        return r
