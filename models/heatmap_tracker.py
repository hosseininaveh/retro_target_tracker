import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

class HeatmapTracker(nn.Module):
    def __init__(self, max_targets=2):
        super().__init__()
        self.max_targets = max_targets
        weights = MobileNet_V2_Weights.DEFAULT
        self.backbone = mobilenet_v2(weights=weights).features
        
        # Heatmap head with output channels = max_targets
        self.heatmap_head = nn.Sequential(
            nn.ConvTranspose2d(1280, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Upsample(size=(480, 640), mode='bilinear', align_corners=False),
            nn.Conv2d(64, max_targets, kernel_size=1),  # Key change: output channels
            nn.Sigmoid()
        )
        
    def forward(self, x):
        features = self.backbone(x)
        return self.heatmap_head(features)  # Output shape: (B, max_targets, H, W)