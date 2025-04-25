import torch
import torch.nn as nn
from .shared_blocks import SubpixelDecoder

class HeatmapTracker(nn.Module):
    """Predicts heatmap of target locations"""
    def __init__(self, backbone='mobilenet_v2'):
        super().__init__()
        self.backbone = torch.hub.load('pytorch/vision', backbone, pretrained=True)
        self.head = nn.Sequential(
            nn.Conv2d(self.backbone.last_channel, 256, 1),
            nn.ReLU(),
            nn.Conv2d(256, 1, 1)  # Single-channel heatmap
        )
        self.subpixel = SubpixelDecoder()
        
    def forward(self, x):
        features = self.backbone.features(x)
        heatmap = self.head(features)
        coords = self.subpixel(heatmap)
        return heatmap, coords
