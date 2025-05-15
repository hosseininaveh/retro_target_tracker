import cv2
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from skimage.draw import disk
from sklearn.model_selection import train_test_split
import shutil

# Configuration
DATASET_ROOT = "yolo_dataset"
IMAGE_PAIRS = [
    # Original image pairs
    {
        "left": "./left_frame.jpg",
        "right": "./right_frame.jpg",
        "left_points": {
            "point0": (272, 396),  # (row, col)
            "point1": (269, 412)
        },
        "right_points": {
            "point0": (270, 214),
            "point1": (270, 230)
        }
    },
    {
        "left": "./c_left.jpg",
        "right": "./c_right.jpg",
        "left_points": {
            "point0": (306, 365),
            "point1": (302, 384)
        },
        "right_points": {
            "point0": (302, 162),
            "point1": (300, 178)
        }
    },
    # New image pairs
    {
        "left": "./p_left_100.jpg",
        "right": "./p_right_100.jpg",
        "left_points": {
            "point0": (193, 342),
            "point1": (175, 558)
        },
        "right_points": {
            "point0": (197, 127),
            "point1": (160, 363)
        }
    },
    {
        "left": "./p_left_400.jpg",
        "right": "./p_right_400.jpg",
        "left_points": {
            "point0": (190, 385),
            "point1": (175, 572)
        },
        "right_points": {
            "point0": (193, 153),
            "point1": (161, 397)
        }
    },
    {
        "left": "./p_left_1.jpg",
        "right": "./p_right_1.jpg",
        "left_points": {
            "point0": (191, 360),
            "point1": (177, 558)
        },
        "right_points": {
            "point0": (195, 133),
            "point1": (163, 374)
        }
    }
]
TOTAL_VARIATIONS = 5000  # Total across all image pairs
TARGET_WINDOW_SIZE = 15  # This will be our bounding box size
MARGIN = 50
RANDOM_OFFSET_RANGE = 30
OUTPUT_SIZE = (640, 480)  # YOLO can handle various sizes but consistent is good
TEST_RATIO = 0.15
VAL_RATIO = 0.15

# YOLO class mapping - point0 is class 0, point1 is class 1
CLASS_MAP = {"point0": 0, "point1": 1}

# [Rest of your functions remain exactly the same...]

def generate_yolo_dataset():
    """Main function to generate the YOLO formatted dataset"""
    # Create YOLO directory structure
    for split in ['train', 'val', 'test']:
        os.makedirs(f"{DATASET_ROOT}/images/{split}", exist_ok=True)
        os.makedirs(f"{DATASET_ROOT}/labels/{split}", exist_ok=True)
    
    # Create dataset.yaml file
    yaml_content = f"""path: {os.path.abspath(DATASET_ROOT)}
train: images/train
val: images/val
test: images/test

# Classes
names:
  0: point0
  1: point1
"""
    with open(f"{DATASET_ROOT}/dataset.yaml", "w") as f:
        f.write(yaml_content)
    
    all_data = []
    
    # Calculate variations per image pair
    variations_per_pair = TOTAL_VARIATIONS // len(IMAGE_PAIRS)
    
    for pair in tqdm(IMAGE_PAIRS, desc="Processing image pairs"):
        # Load images
        left_img = cv2.imread(pair["left"])
        right_img = cv2.imread(pair["right"])
        
        if left_img is None or right_img is None:
            print(f"Warning: Could not load image pair: {pair['left']} and {pair['right']}")
            continue
        
        # Process left image
        left_templates = {}
        left_masks = {}
        for point_name, point_pos in pair["left_points"].items():
            template, mask = extract_target_template(left_img, point_pos, TARGET_WINDOW_SIZE)
            left_templates[point_name] = template
            left_masks[point_name] = mask
        
        clean_left = remove_targets(left_img, pair["left_points"])
        left_variations = generate_variations(
            clean_left, left_templates, left_masks, 
            pair["left_points"], variations_per_pair // 2
        )
        
        # Process right image
        right_templates = {}
        right_masks = {}
        for point_name, point_pos in pair["right_points"].items():
            template, mask = extract_target_template(right_img, point_pos, TARGET_WINDOW_SIZE)
            right_templates[point_name] = template
            right_masks[point_name] = mask
        
        clean_right = remove_targets(right_img, pair["right_points"])
        right_variations = generate_variations(
            clean_right, right_templates, right_masks,
            pair["right_points"], variations_per_pair // 2
        )
        
        # Combine and add to dataset
        all_data.extend(left_variations)
        all_data.extend(right_variations)
    
    # Create list of all samples with their annotations
    samples = []
    for i, var in enumerate(all_data):
        img_width = var["image"].shape[1]
        img_height = var["image"].shape[0]
        
        samples.append({
            "image_id": f"image_{i:05d}.jpg",
            "points": var["points"],
            "width": img_width,
            "height": img_height
        })
    
    # Split dataset
    df = pd.DataFrame(samples)
    train_df, test_df = train_test_split(df, test_size=TEST_RATIO, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=VAL_RATIO/(1-TEST_RATIO), random_state=42)
    
    # Save images and annotations in YOLO format
    def save_subset(subset_df, subset_name):
        for idx, row in subset_df.iterrows():
            var = all_data[idx]
            
            # Save image
            img_path = f"{DATASET_ROOT}/images/{subset_name}/{row['image_id']}"
            cv2.imwrite(img_path, var["image"])
            
            # Save annotation
            annotation = create_yolo_annotation(
                var["points"],
                row['width'],
                row['height']
            )
            txt_path = f"{DATASET_ROOT}/labels/{subset_name}/{os.path.splitext(row['image_id'])[0]}.txt"
            with open(txt_path, "w") as f:
                f.write(annotation)
    
    save_subset(train_df, "train")
    save_subset(val_df, "val")
    save_subset(test_df, "test")
    
    print(f"\nYOLO dataset generation complete!")
    print(f"Total images: {len(all_data)}")
    print(f"Train: {len(train_df)} images")
    print(f"Validation: {len(val_df)} images")
    print(f"Test: {len(test_df)} images")
    print(f"Saved to: {os.path.abspath(DATASET_ROOT)}")
    print(f"Dataset config file: {os.path.abspath(DATASET_ROOT)}/dataset.yaml")

if __name__ == "__main__":
    # Clear existing dataset if it exists
    if os.path.exists(DATASET_ROOT):
        shutil.rmtree(DATASET_ROOT)
    
    generate_yolo_dataset()