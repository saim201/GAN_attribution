
import torch.nn as nn
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights, resnet50, ResNet50_Weights


class EncoderClassifier(nn.Module):
    
    def __init__(self, num_classes=3):
        super().__init__()
        weights = EfficientNet_B3_Weights.IMAGENET1K_V1
        self.backbone = efficientnet_b3(weights=weights)
        feat_dim = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        self.head = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        feat = self.backbone(x)
        return self.head(feat)


class RealFakeClassifier(nn.Module):
    
    def __init__(self, num_classes=2):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2
        self.backbone = resnet50(weights=weights)
        feat_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(feat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        feat = self.backbone(x)
        return self.classifier(feat)
