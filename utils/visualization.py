import matplotlib.pyplot as plt
import numpy as np

def plot_heatmap_comparison(image, pred_heatmap, true_heatmap):
    """Visualize prediction vs ground truth"""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(131)
    plt.imshow(image)
    plt.title("Input Image")
    
    plt.subplot(132)
    plt.imshow(true_heatmap, cmap='hot')
    plt.title("Ground Truth Heatmap")
    
    plt.subplot(133)
    plt.imshow(pred_heatmap, cmap='hot')
    plt.title("Predicted Heatmap")
    
    plt.tight_layout()
    plt.show()

def draw_tracking_result(frame, pred_coords, true_coords=None):
    """Draw tracking results on frame"""
    frame = frame.copy()
    x, y = pred_coords
    cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
    
    if true_coords is not None:
        tx, ty = true_coords
        cv2.circle(frame, (int(tx), int(ty)), 3, (0, 0, 255), -1)
    
    return frame
