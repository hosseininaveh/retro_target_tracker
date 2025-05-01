import cv2
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from skimage.draw import disk
from sklearn.model_selection import train_test_split

# Configuration
DATASET_ROOT = "enhanced_dataset"
IMAGE_PAIRS = [
    {
        "left": "./left_frame.jpg",
        "right": "./right_frame.jpg",
        "left_points": {
            "point0": (272, 396),
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
TARGET_WINDOW_SIZE = 15
MARGIN = 50
RANDOM_OFFSET_RANGE = 30
OUTPUT_SIZE = (640, 480)
TEST_RATIO = 0.15
VAL_RATIO = 0.15

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
                "point0": sorted_points[0][1],
                "point1": sorted_points[1][1]
            },
            "original_image": os.path.basename(clean_img.filename) if hasattr(clean_img, 'filename') else "generated"
        })
    
    return variations

def generate_dataset():
    # Create directory structure
    os.makedirs(f"{DATASET_ROOT}/train/images", exist_ok=True)
    os.makedirs(f"{DATASET_ROOT}/train/annotations", exist_ok=True)
    os.makedirs(f"{DATASET_ROOT}/val/images", exist_ok=True)
    os.makedirs(f"{DATASET_ROOT}/val/annotations", exist_ok=True)
    os.makedirs(f"{DATASET_ROOT}/test/images", exist_ok=True)
    os.makedirs(f"{DATASET_ROOT}/test/annotations", exist_ok=True)

    all_data = []
    
    # Calculate variations per image pair
    variations_per_pair = TOTAL_VARIATIONS // len(IMAGE_PAIRS)
    
    for pair in IMAGE_PAIRS:
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
    
    # Convert to DataFrame
    df = pd.DataFrame([{
        "image_path": f"image_{i:05d}.jpg",
        "point0_row": var["points"]["point0"][0],
        "point0_col": var["points"]["point0"][1],
        "point1_row": var["points"]["point1"][0],
        "point1_col": var["points"]["point1"][1],
        "original_image": var["original_image"]
    } for i, var in enumerate(all_data)])
    
    # Split dataset
    train_df, test_df = train_test_split(df, test_size=TEST_RATIO)
    train_df, val_df = train_test_split(train_df, test_size=VAL_RATIO/(1-TEST_RATIO))
    
    # Save images and annotations
    def save_subset(subset_df, subset_name):
        os.makedirs(f"{DATASET_ROOT}/{subset_name}/images", exist_ok=True)
        for idx, row in subset_df.iterrows():
            var = all_data[idx]
            cv2.imwrite(f"{DATASET_ROOT}/{subset_name}/images/{row['image_path']}", var["image"])
        subset_df.to_csv(f"{DATASET_ROOT}/{subset_name}/annotations/annotations.csv", index=False)
    
    save_subset(train_df, "train")
    save_subset(val_df, "val")
    save_subset(test_df, "test")
    
    print(f"Dataset generated with {len(all_data)} images")
    print(f"Train: {len(train_df)} images")
    print(f"Validation: {len(val_df)} images")
    print(f"Test: {len(test_df)} images")
    print(f"Saved to: {os.path.abspath(DATASET_ROOT)}")

if __name__ == "__main__":
    generate_dataset()