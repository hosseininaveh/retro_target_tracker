import torch
import torch.nn as nn

class SubpixelDecoder(nn.Module):
    """Converts heatmaps to subpixel coordinates"""
    def forward(self, heatmap):
        batch_size = heatmap.shape[0]
        coords = torch.zeros(batch_size, 2, device=heatmap.device)
        
        # Spatial softmax
        heatmap = nn.functional.softmax(heatmap.view(batch_size, -1), dim=1)
        heatmap = heatmap.view_as(heatmap)
        
        # Expected value calculation
        for i in range(batch_size):
            y_coords = torch.arange(heatmap.size(2), device=heatmap.device)
            x_coords = torch.arange(heatmap.size(3), device=heatmap.device)
            grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
            
            x_mean = (heatmap[i,0] * grid_x).sum()
            y_mean = (heatmap[i,0] * grid_y).sum()
            coords[i] = torch.stack([x_mean, y_mean])
            
        return coords

class AttentionRefinement(nn.Module):
    """Improves spatial localization"""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, 1, kernel_size=1)
        
    def forward(self, x):
        return torch.sigmoid(self.conv(x))
