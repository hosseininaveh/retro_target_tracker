import torch
import cv2
import numpy as np
from models import HeatmapTracker
from utils.visualization import draw_tracking_result

class TargetTracker:
    def __init__(self, model_path, device='cuda'):
        """Initialize tracker with trained model"""
        self.device = torch.device(device)
        self.model = HeatmapTracker().to(self.device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
        # Warmup
        dummy_input = torch.randn(1, 3, 256, 256).to(self.device)
        with torch.no_grad():
            _ = self.model(dummy_input)

    def preprocess(self, frame):
        """Convert frame to model input tensor"""
        # Resize and normalize
        frame = cv2.resize(frame, (256, 256))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(frame).float().permute(2,0,1) / 255.0
        return tensor.unsqueeze(0).to(self.device)

    def postprocess(self, coords, input_shape, output_shape):
        """Convert normalized coords to original image coordinates"""
        # Scale coordinates
        x_scale = output_shape[1] / input_shape[1]
        y_scale = output_shape[0] / input_shape[0]
        coords[0] *= x_scale
        coords[1] *= y_scale
        return coords.astype(int)

    def track(self, frame):
        """Track targets in a single frame"""
        original_shape = frame.shape[:2]
        input_tensor = self.preprocess(frame)
        
        with torch.no_grad():
            _, coords = self.model(input_tensor)
        
        # Convert to numpy array
        coords = coords[0].cpu().numpy()
        
        # Post-process coordinates
        return self.postprocess(coords, (256, 256), original_shape)

    def track_video(self, video_path, output_path=None):
        """Process video file or camera stream"""
        cap = cv2.VideoCapture(video_path if video_path else 0)
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output_path, fourcc, 30.0, 
                                (int(cap.get(3)), int(cap.get(4))))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Track and visualize
            (x, y), _ = self.track(frame)
            vis_frame = draw_tracking_result(frame, (x, y))
            
            cv2.imshow('Tracking', vis_frame)
            if output_path:
                out.write(vis_frame)
            
            if cv2.waitKey(1) & 0xFF == 27:
                break
        
        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Path to trained model')
    parser.add_argument('--video', default=None, help='Video file path (leave empty for camera)')
    parser.add_argument('--output', default=None, help='Output video path')
    args = parser.parse_args()
    
    tracker = TargetTracker(args.model)
    tracker.track_video(args.video, args.output)
