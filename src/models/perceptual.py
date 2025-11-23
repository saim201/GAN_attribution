import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights


class VGGPerceptual(nn.Module):
    
    def __init__(self):
        super().__init__()
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_FEATURES).features.eval()
        self.slice = nn.Sequential(*list(vgg.children())[:9])
        for p in self.slice.parameters():
            p.requires_grad = False

    def forward(self, x_norm, y_norm):
        fx = self.slice(x_norm)
        fy = self.slice(y_norm)
        return F.l1_loss(fx, fy)
