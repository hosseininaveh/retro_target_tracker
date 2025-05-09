import torch
import cv2
import numpy as np
from models import HeatmapTracker

class TargetDetector:
    def __init__(self, model_path, device='cuda', max_targets=2):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.max_targets = max_targets
        self.model = HeatmapTracker(max_targets=max_targets).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
    def preprocess(self, frame):
        """Convert frame to model input format"""
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.0
        frame = torch.from_numpy(frame).permute(2, 0, 1).float().unsqueeze(0)
        return frame.to(self.device)
    
    def detect(self, frame):
        """Detect targets ordered left to right with sub-pixel accuracy"""
        with torch.no_grad():
            input_tensor = self.preprocess(frame)
            heatmaps = self.model(input_tensor).squeeze(0).cpu().numpy()
        
        targets = []
        for i in range(self.max_targets):
            heatmap = heatmaps[i] if self.max_targets > 1 else heatmaps
            
            # Get pixel-level center
            y, x = np.unravel_index(heatmap.argmax(), heatmap.shape)
            
            if heatmap[y, x] < 0.1:  # Skip weak detections
                continue
            
            # Sub-pixel refinement
            if 0 < x < heatmap.shape[1]-1 and 0 < y < heatmap.shape[0]-1:
                dx = 0.5 * (heatmap[y, x+1] - heatmap[y, x-1])
                dy = 0.5 * (heatmap[y+1, x] - heatmap[y-1, x])
                x += dx
                y += dy
            
            targets.append((x, y))
        
        # Sort targets left to right by x-coordinate
        targets.sort(key=lambda t: t[0])
        
        # Visual verification (optional)
        debug_img = self.visualize_detections(frame.copy(), targets)
        cv2.imshow("Detections", debug_img)
        cv2.waitKey(1)
        
        return targets, heatmaps
    
    def visualize_detections(self, frame, targets):
        """Visualize targets with left-right labels"""
        for idx, (x, y) in enumerate(targets):
            color = (0, 255, 0) if idx == 0 else (0, 0, 255)  # Green=left, Red=right
            cv2.circle(frame, (int(x), int(y)), 10, color, 2)
            cv2.putText(frame, f"Target {idx}", (int(x)+15, int(y)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return frame