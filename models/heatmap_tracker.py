import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from .shared_blocks import SubpixelDecoder

class HeatmapTracker(nn.Module):
    def __init__(self, backbone='mobilenet_v2'):
        super().__init__()
        
        # Modern weights handling
        weights = MobileNet_V2_Weights.IMAGENET1K_V1  # Changed from V2 to V1
        self.backbone = mobilenet_v2(weights=weights).features
        self.head = nn.Sequential(
            nn.Conv2d(1280, 256, 1),
            nn.ReLU(),
            nn.Conv2d(256, 1, 1)
        )
        self.subpixel = SubpixelDecoder()
        
    def forward(self, x):
        features = self.backbone(x)
        heatmap = self.head(features)
        coords = self.subpixel(heatmap)
        return heatmap, coords