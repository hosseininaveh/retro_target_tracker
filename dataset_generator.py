import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Configuration
DATASET_ROOT = "data"
IMAGE_SIZE = (640, 480)  # Updated to 640x480
TARGET_SIZES = [7, 9, 11, 13, 15, 17]  # Varying target sizes
NUM_IMAGES = 3000  # Total images to generate
TEST_RATIO = 0.15
VAL_RATIO = 0.15
MAX_TARGETS = 2  # Maximum number of targets per image

def generate_target(center, target_size, base_red_value):
    """Generate a synthetic target with smooth color transitions"""
    # Create blank image
    img = np.zeros(IMAGE_SIZE + (3,), dtype=np.float32)
    
    # Generate smooth red background (BGR format)
    red_layer = np.random.uniform(0.15, 0.3, IMAGE_SIZE)  # Base red level
    img[:,:,2] = red_layer  # Red channel
    
    # Create target mask
    mask = np.zeros(IMAGE_SIZE, dtype=np.uint8)
    half_size = target_size // 2
    cv2.rectangle(mask, 
                 (int(center[0]-half_size), int(center[1]-half_size)),
                 (int(center[0]+half_size), int(center[1]+half_size)), 
                 255, -1)
    
    # Apply smooth red target (slightly brighter than background)
    target_red = base_red_value + np.random.uniform(0.1, 0.2)
    img[:,:,2] = np.where(mask==255, target_red, img[:,:,2])
    
    # Add white circle (smooth transition)
    circle_mask = np.zeros(IMAGE_SIZE, dtype=np.uint8)
    circle_size = max(3, target_size - 4)
    cv2.circle(circle_mask, (int(center[0]), int(center[1])), circle_size//2, 255, -1)  # Fixed this line
    
    # White circle should be 1.5-2x brighter than target red
    white_value = target_red * np.random.uniform(1.5, 2.0)
    img[:,:,2] = np.where(circle_mask==255, white_value, img[:,:,2])
    img[:,:,1] = np.where(circle_mask==255, white_value, img[:,:,1])  # Green channel
    img[:,:,0] = np.where(circle_mask==255, white_value, img[:,:,0])  # Blue channel
    
    # Add realistic noise (less aggressive)
    noise = np.random.normal(0, 0.02, img.shape)
    img = np.clip(img + noise, 0, 1)
    
    # Convert to 8-bit and apply slight blur
    img = (img * 255).astype(np.uint8)
    img = cv2.GaussianBlur(img, (3,3), 0)
    
    return img

def generate_dataset():
    # Create directories
    os.makedirs(f"{DATASET_ROOT}/train/images", exist_ok=True)
    os.makedirs(f"{DATASET_ROOT}/train/annotations", exist_ok=True)
    os.makedirs(f"{DATASET_ROOT}/val/images", exist_ok=True)
    os.makedirs(f"{DATASET_ROOT}/val/annotations", exist_ok=True)
    os.makedirs(f"{DATASET_ROOT}/test/images", exist_ok=True)
    os.makedirs(f"{DATASET_ROOT}/test/annotations", exist_ok=True)

    # Generate consistent base red value that slowly changes
    base_red = np.random.uniform(0.15, 0.3)
    red_delta = np.random.uniform(-0.01, 0.01)  # Small change per frame
    
    all_data = []
    for i in tqdm(range(NUM_IMAGES)):
        # Slowly vary the base red value
        base_red = np.clip(base_red + red_delta, 0.15, 0.3)
        if np.random.rand() < 0.1:  # Occasionally change direction
            red_delta *= -1
        
        # Create blank image
        img = np.zeros(IMAGE_SIZE + (3,), dtype=np.float32)
        
        # Generate between 1 and MAX_TARGETS targets
        num_targets = np.random.randint(1, MAX_TARGETS+1)
        target_info = []
        
        for _ in range(num_targets):
            # Random position (avoid edges and other targets)
            margin = 40
            min_distance = 100  # Minimum distance between targets
            
            while True:
                x = np.random.uniform(margin, IMAGE_SIZE[0] - margin)
                y = np.random.uniform(margin, IMAGE_SIZE[1] - margin)
                
                # Check distance to other targets
                valid_position = True
                for (tx, ty, _) in target_info:
                    if np.sqrt((x-tx)**2 + (y-ty)**2) < min_distance:
                        valid_position = False
                        break
                
                if valid_position:
                    break
            
            # Random target size
            target_size = np.random.choice(TARGET_SIZES)
            
            # Generate target and add to image
            target_img = generate_target((x, y), target_size, base_red)
            img = np.maximum(img, target_img)
            
            # Store target info with sub-pixel accuracy
            target_info.append((x, y, target_size))
        
        # Store metadata
        img_name = f"target_{i:04d}.png"
        all_data.append({
            "image_path": img_name,
            "targets": target_info  # List of (x, y, size) tuples
        })
        
        # Save image
        cv2.imwrite(f"{DATASET_ROOT}/temp_{img_name}", img)

    # Split dataset
    train_data, test_data = train_test_split(all_data, test_size=TEST_RATIO)
    train_data, val_data = train_test_split(train_data, test_size=VAL_RATIO/(1-TEST_RATIO))

    # Save datasets
    def save_subset(data, subset):
        # Convert to DataFrame with proper format
        records = []
        for item in data:
            for target_idx, (x, y, size) in enumerate(item["targets"]):
                records.append({
                    "image_path": item["image_path"],
                    "target_idx": target_idx,
                    "x": x,
                    "y": y,
                    "target_size": size
                })
        
        df = pd.DataFrame(records)
        df.to_csv(f"{DATASET_ROOT}/{subset}/annotations/annotations.csv", index=False)
        
        # Save images
        for item in data:
            os.rename(f"{DATASET_ROOT}/temp_{item['image_path']}", 
                     f"{DATASET_ROOT}/{subset}/images/{item['image_path']}")

    save_subset(train_data, "train")
    save_subset(val_data, "val")
    save_subset(test_data, "test")

    # Cleanup temp files
    for f in os.listdir(f"{DATASET_ROOT}"):
        if f.startswith("temp_"):
            os.remove(f"{DATASET_ROOT}/{f}")

if __name__ == "__main__":
    generate_dataset()
    print(f"Dataset generated in {DATASET_ROOT} directory")