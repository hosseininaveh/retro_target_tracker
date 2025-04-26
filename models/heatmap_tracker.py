import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

class HeatmapTracker(nn.Module):
    def __init__(self):
        super().__init__()
        weights = MobileNet_V2_Weights.DEFAULT
        self.backbone = mobilenet_v2(weights=weights).features
        
        # Heatmap head with proper upsampling
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(1280, 256, 1),
            nn.ReLU(),
            nn.Conv2d(256, 1, 1),
            nn.Upsample(size=256, mode='bilinear', align_corners=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        features = self.backbone(x)
        heatmap = self.heatmap_head(features)
        return heatmap  # Returns (B, 1, 256, 256)