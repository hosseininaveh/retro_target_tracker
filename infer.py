import cv2
import torch
import numpy as np
from models import HeatmapTracker
from visualization import draw_tracking_result
from subpixel import SubpixelDecoder

class VideoTracker:
    def __init__(self, model_path, device='cuda', max_targets=2):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.max_targets = max_targets
        self.model = HeatmapTracker(max_targets=max_targets).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.subpixel_decoder = SubpixelDecoder()
        
    def preprocess(self, frame):
        """Convert frame to model input format"""
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.0
        frame = torch.from_numpy(frame).permute(2, 0, 1).float().unsqueeze(0)
        return frame.to(self.device)
    
    def detect(self, frame):
        """Detect targets with sub-pixel accuracy"""
        with torch.no_grad():
            input_tensor = self.preprocess(frame)
            heatmaps = self.model(input_tensor).squeeze(0).cpu().numpy()
        
        targets = []
        confidences = []
        
        for i in range(self.max_targets):
            heatmap = heatmaps[i] if self.max_targets > 1 else heatmaps
            
            # Get pixel-level center
            y, x = np.unravel_index(heatmap.argmax(), heatmap.shape)
            confidence = heatmap[y, x]
            
            if confidence < 0.1:  # Skip weak detections
                continue
            
            # Sub-pixel refinement
            if 0 < x < heatmap.shape[1]-1 and 0 < y < heatmap.shape[0]-1:
                dx = 0.5 * (heatmap[y, x+1] - heatmap[y, x-1])
                dy = 0.5 * (heatmap[y+1, x] - heatmap[y-1, x])
                x += dx
                y += dy
            
            targets.append((x, y))
            confidences.append(confidence)
        
        # Sort targets left to right by x-coordinate (point0 should be leftmost)
        if len(targets) == 2:
            if targets[0][0] > targets[1][0]:
                targets = [targets[1], targets[0]]
                confidences = [confidences[1], confidences[0]]
        
        return targets, confidences, heatmaps
    
    def process_video(self, video_path, output_path=None, show=True):
        """Process video file and display/save results"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Initialize video writer if output path is specified
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Detect targets
            targets, confidences, _ = self.detect(frame)
            
            # Visualize results
            for i, (point, conf) in enumerate(zip(targets, confidences)):
                color = (0, 255, 0) if i == 0 else (0, 0, 255)  # Green for point0, Red for point1
                label = f"Point{i} ({conf:.2f})"
                
                # Draw center point
                cv2.circle(frame, (int(point[0]), int(point[1])), 5, color, -1)
                
                # Draw label
                cv2.putText(frame, label, (int(point[0]) + 10, int(point[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Display frame
            if show:
                cv2.imshow('Target Tracking', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Write frame to output video
            if output_path:
                out.write(frame)
            
            frame_count += 1
            print(f"Processed frame {frame_count}", end='\r')
        
        # Clean up
        cap.release()
        if output_path:
            out.release()
        if show:
            cv2.destroyAllWindows()
        
        print(f"\nFinished processing {frame_count} frames")

if __name__ == "__main__":
    # Initialize tracker with your trained model
    tracker = VideoTracker(model_path="path/to/your/best_model.pth", max_targets=2)
    
    # Process video file
    tracker.process_video(
        video_path="left.avi",
        output_path="output.avi",  # Set to None if you don't want to save
        show=True
    )