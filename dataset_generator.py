import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Configuration
DATASET_ROOT = "data"
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
TARGET_SIZES = [7, 9, 11, 13, 15, 17]
NUM_IMAGES = 3000
TEST_RATIO = 0.15
VAL_RATIO = 0.15
MAX_TARGETS = 2

def generate_target(center, target_size, base_red_value):
    """Generate a synthetic target with smooth color transitions"""
    img = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.float32)
    
    # Background
    red_layer = np.random.uniform(0.15, 0.3, (IMAGE_HEIGHT, IMAGE_WIDTH))
    img[:,:,2] = red_layer
    
    # Target rectangle
    mask = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8)
    half_size = target_size // 2
    cv2.rectangle(mask, 
                (int(center[0]-half_size), int(center[1]-half_size)),
                (int(center[0]+half_size), int(center[1]+half_size)), 
                255, -1)
    
    target_red = base_red_value + np.random.uniform(0.1, 0.2)
    img[:,:,2] = np.where(mask==255, target_red, img[:,:,2])
    
    # White circle
    circle_mask = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8)
    circle_size = max(3, target_size - 4)
    cv2.circle(circle_mask, (int(center[0]), int(center[1])), circle_size//2, 255, -1)
    
    white_value = target_red * np.random.uniform(1.5, 2.0)
    img[:,:,0:3] = np.where(circle_mask[...,None]==255, white_value, img)
    
    # Final processing
    noise = np.random.normal(0, 0.02, img.shape)
    img = np.clip(img + noise, 0, 1)
    img = (img * 255).astype(np.uint8)
    img = cv2.GaussianBlur(img, (3,3), 0)
    
    return img

def generate_dataset():
    os.makedirs(f"{DATASET_ROOT}/train/images", exist_ok=True)
    os.makedirs(f"{DATASET_ROOT}/train/annotations", exist_ok=True)
    os.makedirs(f"{DATASET_ROOT}/val/images", exist_ok=True)
    os.makedirs(f"{DATASET_ROOT}/val/annotations", exist_ok=True)
    os.makedirs(f"{DATASET_ROOT}/test/images", exist_ok=True)
    os.makedirs(f"{DATASET_ROOT}/test/annotations", exist_ok=True)

    base_red = np.random.uniform(0.15, 0.3)
    red_delta = np.random.uniform(-0.01, 0.01)
    
    all_data = []
    for i in tqdm(range(NUM_IMAGES)):
        base_red = np.clip(base_red + red_delta, 0.15, 0.3)
        if np.random.rand() < 0.1:
            red_delta *= -1
        
        img = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.float32)
        num_targets = np.random.randint(1, MAX_TARGETS+1)
        target_info = []
        
        for _ in range(num_targets):
            margin = 40
            min_distance = 100
            
            while True:
                x = np.random.uniform(margin, IMAGE_WIDTH - margin)
                y = np.random.uniform(margin, IMAGE_HEIGHT - margin)
                
                valid_position = True
                for (tx, ty, _) in target_info:
                    if np.sqrt((x-tx)**2 + (y-ty)**2) < min_distance:
                        valid_position = False
                        break
                
                if valid_position:
                    break
            
            target_size = np.random.choice(TARGET_SIZES)
            target_img = generate_target((x, y), target_size, base_red)
            img = np.maximum(img, target_img)
            target_info.append((x, y, target_size))
        
        # Sort targets by x-coordinate (left to right)
        target_info.sort(key=lambda t: t[0])
        
        img_name = f"target_{i:04d}.png"
        all_data.append({
            "image_path": img_name,
            "targets": target_info
        })
        
        cv2.imwrite(f"{DATASET_ROOT}/temp_{img_name}", img)

    train_data, test_data = train_test_split(all_data, test_size=TEST_RATIO)
    train_data, val_data = train_test_split(train_data, test_size=VAL_RATIO/(1-TEST_RATIO))

    def save_subset(data, subset):
        records = []
        for item in data:
            for target_idx, (x, y, size) in enumerate(item["targets"]):
                records.append({
                    "image_path": item["image_path"],
                    "target_idx": target_idx,  # 0=leftmost, 1=rightmost
                    "x": x,
                    "y": y,
                    "target_size": size
                })
        
        df = pd.DataFrame(records)
        df.to_csv(f"{DATASET_ROOT}/{subset}/annotations/annotations.csv", index=False)
        
        for item in data:
            os.rename(f"{DATASET_ROOT}/temp_{item['image_path']}", 
                     f"{DATASET_ROOT}/{subset}/images/{item['image_path']}")

    save_subset(train_data, "train")
    save_subset(val_data, "val")
    save_subset(test_data, "test")

    for f in os.listdir(f"{DATASET_ROOT}"):
        if f.startswith("temp_"):
            os.remove(f"{DATASET_ROOT}/{f}")

if __name__ == "__main__":
    generate_dataset()
    print(f"Dataset generated in {DATASET_ROOT} directory")