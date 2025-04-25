import torch
import torch.nn as nn
import torch.nn.functional as F

class SubpixelDecoder(nn.Module):
    """Converts heatmaps to subpixel coordinates with dimension validation"""
    def forward(self, heatmap):
        # Get dimensions safely
        batch_size, channels, height, width = heatmap.size()
        
        # Validate dimensions
        if height <= 2 or width <= 2:
            raise ValueError(f"Heatmap size {height}x{width} is too small for coordinate calculation")
        
        # Calculate coordinates
        y_coords = torch.arange(height, device=heatmap.device)
        x_coords = torch.arange(width, device=heatmap.device)
        
        return y_coords, x_coords

class AttentionRefinement(nn.Module):
    """Improves spatial localization with batch normalization"""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, 1, kernel_size=1)
        self.bn = nn.BatchNorm2d(1)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return torch.sigmoid(x)