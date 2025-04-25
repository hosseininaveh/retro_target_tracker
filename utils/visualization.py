import matplotlib.pyplot as plt
import numpy as np

def plot_heatmap_comparison(image, pred_heatmap, true_heatmap):
    plt.figure(figsize=(15, 5))
    
    # Remove channel dimension if present (for single-channel heatmaps)
    if len(pred_heatmap.shape) == 3 and pred_heatmap.shape[0] == 1:
        pred_heatmap = pred_heatmap.squeeze(0)
    if len(true_heatmap.shape) == 3 and true_heatmap.shape[0] == 1:
        true_heatmap = true_heatmap.squeeze(0)
    
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("Input Image")
    
    plt.subplot(1, 3, 2)
    plt.imshow(pred_heatmap, cmap='hot')
    plt.title("Predicted Heatmap")
    
    plt.subplot(1, 3, 3)
    plt.imshow(true_heatmap, cmap='hot')
    plt.title("True Heatmap")
    
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
