import torch
import numpy as np

class SubpixelDecoder:
    """Converts heatmaps to subpixel coordinates"""
    def __call__(self, heatmap):
        batch_size = heatmap.shape[0]
        coords = torch.zeros(batch_size, 2)
        
        for i in range(batch_size):
            # Quadratic interpolation
            y, x = np.unravel_index(heatmap[i].argmax(), heatmap.shape[2:])
            if 0 < x < heatmap.shape[3]-1 and 0 < y < heatmap.shape[2]-1:
                dx = 0.5 * (heatmap[i,0,y,x+1] - heatmap[i,0,y,x-1])
                dy = 0.5 * (heatmap[i,0,y+1,x] - heatmap[i,0,y-1,x])
                x += dx.item()
                y += dy.item()
                
            coords[i] = torch.tensor([x, y])
        
        return coords
