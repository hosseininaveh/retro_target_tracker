import torch
import cv2
import numpy as np
from models import HeatmapTracker

class TargetDetector:
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = HeatmapTracker().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
    def preprocess(self, frame):
        """Convert frame to model input format"""
        # Convert to RGB and normalize
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.0
        # Convert to tensor and add batch dimension
        frame = torch.from_numpy(frame).permute(2, 0, 1).float().unsqueeze(0)
        return frame.to(self.device)
    
    def detect(self, frame):
        """Detect target centers with sub-pixel accuracy"""
        with torch.no_grad():
            input_tensor = self.preprocess(frame)
            heatmap = self.model(input_tensor).squeeze().cpu().numpy()
        
        # Get pixel-level center
        y, x = np.unravel_index(heatmap.argmax(), heatmap.shape)
        
        # Quadratic interpolation for sub-pixel accuracy
        if 0 < x < heatmap.shape[1]-1 and 0 < y < heatmap.shape[0]-1:
            dx = 0.5 * (heatmap[y, x+1] - heatmap[y, x-1])
            dy = 0.5 * (heatmap[y+1, x] - heatmap[y-1, x])
            x += dx
            y += dy
        
        return (x, y), heatmap
