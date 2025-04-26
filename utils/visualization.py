import matplotlib.pyplot as plt
import cv2
import numpy as np


def draw_tracking_result(frame, point, confidence=None):
    """
    Draw tracking result on the frame
    
    Args:
        frame: Input image
        point: Tuple of (x, y) coordinates
        confidence: Optional confidence score (0-1)
    
    Returns:
        Frame with visualization overlay
    """
    x, y = point
    
    # Draw the center point
    cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
    
    # Draw a larger circle to indicate tracking area
    cv2.circle(frame, (int(x), int(y)), 15, (0, 255, 255), 2)
    
    # Add confidence text if provided
    if confidence is not None:
        text = f"Confidence: {confidence:.2f}"
        cv2.putText(frame, text, (int(x) + 20, int(y) - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return frame



def plot_heatmap_comparison(image, pred_heatmap, true_heatmap):
    """Plot comparison between input image, predicted heatmap and true heatmap"""
    plt.figure(figsize=(15, 5))
    
    # Input image
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("Input Image")
    plt.axis('off')
    
    # Predicted heatmap
    plt.subplot(1, 3, 2)
    if len(pred_heatmap.shape) == 3:  # If (1, H, W)
        pred_heatmap = pred_heatmap.squeeze(0)
    elif len(pred_heatmap.shape) == 4:  # If (B, 1, H, W)
        pred_heatmap = pred_heatmap.squeeze(1)[0]
    plt.imshow(pred_heatmap, cmap='hot')
    plt.title("Predicted Heatmap")
    plt.axis('off')
    
    # True heatmap
    plt.subplot(1, 3, 3)
    plt.imshow(true_heatmap, cmap='hot')
    plt.title("True Heatmap")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
