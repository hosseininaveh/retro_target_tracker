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

def extract_target_template(img, center, window_size=15):
    """Extract target template with precise masking"""
    h, w = img.shape[:2]
    y, x = center
    
    y1 = max(0, int(y) - window_size//2)
    y2 = min(h, int(y) + window_size//2 + 1)
    x1 = max(0, int(x) - window_size//2)
    x2 = min(w, int(x) + window_size//2 + 1)
    
    mask = np.zeros((y2-y1, x2-x1), dtype=np.uint8)
    cy, cx = (y2-y1)//2, (x2-x1)//2
    cv2.circle(mask, (cx, cy), window_size//2, 255, -1)
    
    target_window = img[y1:y2, x1:x2].copy()
    return target_window, mask

def remove_targets(img, target_positions, window_size=15):
    """Remove targets using precise inpainting"""
    clean_img = img.copy()
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    
    for center in target_positions.values():
        y, x = center
        radius = window_size//2 + 2
        rr, cc = disk((y, x), radius, shape=img.shape[:2])
        mask[rr, cc] = 255
    
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.dilate(mask, kernel)
    clean_img = cv2.inpaint(clean_img, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
    return clean_img

def place_target(img, target_template, target_mask, center):
    """Place target at new position with proper blending"""
    h, w = img.shape[:2]
    temp_h, temp_w = target_template.shape[:2]
    
    y = int(round(center[0]))
    x = int(round(center[1]))
    y1 = max(0, y - temp_h//2)
    y2 = min(h, y1 + temp_h)
    x1 = max(0, x - temp_w//2)
    x2 = min(w, x1 + temp_w)
    
    template = target_template
    tmask = target_mask
    if (y2-y1) != temp_h or (x2-x1) != temp_w:
        ty1 = max(0, temp_h//2 - y)
        ty2 = temp_h - max(0, (y + temp_h//2) - h)
        tx1 = max(0, temp_w//2 - x)
        tx2 = temp_w - max(0, (x + temp_w//2) - w)
        template = template[ty1:ty2, tx1:tx2]
        tmask = tmask[ty1:ty2, tx1:tx2]
    
    roi = img[y1:y2, x1:x2]
    mask = tmask[..., None].astype(float)/255.0
    img[y1:y2, x1:x2] = (roi * (1 - mask) + template * mask).astype(np.uint8)
    return img

def generate_variations(clean_img, target_templates, target_masks, original_points, num_variations):
    """Generate image variations with targets in new positions"""
    variations = []
    h, w = clean_img.shape[:2]
    
    for _ in range(num_variations):
        new_img = clean_img.copy()
        new_points = {}
        
        for point_name, original_point in original_points.items():
            row_offset = np.random.uniform(-RANDOM_OFFSET_RANGE, RANDOM_OFFSET_RANGE)
            col_offset = np.random.uniform(-RANDOM_OFFSET_RANGE, RANDOM_OFFSET_RANGE)
            
            new_row = original_point[0] + row_offset
            new_col = original_point[1] + col_offset
            
            new_row = np.clip(new_row, MARGIN, h-MARGIN-1)
            new_col = np.clip(new_col, MARGIN, w-MARGIN-1)
            
            new_points[point_name] = (new_row, new_col)
        
        sorted_points = sorted(new_points.items(), key=lambda x: x[1][1])
        
        for idx, (point_name, new_center) in enumerate(sorted_points):
            template = target_templates[point_name]
            mask = target_masks[point_name]
            new_img = place_target(new_img, template, mask, new_center)
        
        if OUTPUT_SIZE != (w, h):
            new_img = cv2.resize(new_img, OUTPUT_SIZE)
            scale_x = OUTPUT_SIZE[0] / w
            scale_y = OUTPUT_SIZE[1] / h
            sorted_points = [
                (name, (row*scale_y, col*scale_x)) 
                for name, (row, col) in sorted_points
            ]
        
        variations.append({
            "image": new_img,
            "points": {
                "point0": sorted_points[0][1],  # (row, col)
                "point1": sorted_points[1][1]   # (row, col)
            },
            "original_image": os.path.basename(clean_img.filename) if hasattr(clean_img, 'filename') else "generated"
        })
    
    return variations

def create_yolo_annotation(points, img_width, img_height):
    """Convert points to YOLO format (normalized xywh)"""
    annotations = []
    box_size = TARGET_WINDOW_SIZE
    
    for point_name, (y, x) in points.items():
        # Convert to center coordinates
        x_center = x / img_width
        y_center = y / img_height
        
        # Normalized width and height
        width = box_size / img_width
        height = box_size / img_height
        
        # Class ID
        class_id = CLASS_MAP[point_name]
        
        annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    
    return "\n".join(annotations)

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
            raise ValueError(f"Could not load image pair: {pair['left']} and {pair['right']}")
        
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