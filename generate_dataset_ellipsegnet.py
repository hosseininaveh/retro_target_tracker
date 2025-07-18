import cv2
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from skimage.draw import disk
from sklearn.model_selection import train_test_split
import shutil
from scipy.ndimage import gaussian_filter

# Configuration
DATASET_ROOT = "/home/mehdi/test_concrete_4/MarkerPose/dataset/ellipsegnet_dataset"
IMAGE_PAIRS = [
    {
        "left": "/home/mehdi/test_concrete_4/MarkerPose/images/image_00008.jpg",
        "left_points": {
            "point0": (258.864132, 276.019174),  # (row, col)
            "point1": (249.733477, 496.697453)
        }
    },
    # Add more images with keypoints as needed
]
TOTAL_VARIATIONS = 1000
TARGET_WINDOW_SIZE = 15
MARGIN = 50
RANDOM_OFFSET_RANGE = 30
OUTPUT_SIZE = (120, 120)  # EllipSegNet patch size
TEST_RATIO = 0.15
VAL_RATIO = 0.15
MASK_SIZE = (120, 120)
MASK_RADIUS = 10  # Radius of marker in patch

def extract_target_template(img, center, window_size=15):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape
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
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clean_img = img.copy()
    mask = np.zeros(img.shape, dtype=np.uint8)
    
    for center in target_positions.values():
        y, x = center
        radius = window_size//2 + 2
        rr, cc = disk((y, x), radius, shape=img.shape)
        mask[rr, cc] = 255
    
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.dilate(mask, kernel)
    clean_img = cv2.inpaint(clean_img, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
    return clean_img

def place_target(img, target_template, target_mask, center):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if len(target_template.shape) == 3:
        target_template = cv2.cvtColor(target_template, cv2.COLOR_BGR2GRAY)
    h, w = img.shape
    temp_h, temp_w = target_template.shape
    
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
    mask = tmask.astype(float)/255.0
    img[y1:y2, x1:x2] = (roi * (1 - mask) + template * mask).astype(np.uint8)
    return img

def create_segmentation_mask(point, img_size=(120, 120), radius=10):
    mask = np.zeros(img_size, dtype=np.float32)
    y, x = point
    center_y, center_x = img_size[0]//2, img_size[1]//2  # Center of patch
    rr, cc = disk((center_y, center_x), radius, shape=img_size)
    mask[rr, cc] = 1.0
    return mask

def generate_variations(clean_img, target_templates, target_masks, original_points, num_variations):
    variations = []
    h, w = clean_img.shape
    
    for i in range(num_variations):
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
        new_points = {name: pos for name, pos in sorted_points}
        
        for point_name, new_center in new_points.items():
            template = target_templates[point_name]
            mask = target_masks[point_name]
            new_img = place_target(new_img, template, mask, new_center)
            
            # Crop 120x120 patch around the point
            y, x = new_center
            y1 = max(0, int(y - OUTPUT_SIZE[1]//2))
            y2 = min(h, y1 + OUTPUT_SIZE[1])
            x1 = max(0, int(x - OUTPUT_SIZE[0]//2))
            x2 = min(w, x1 + OUTPUT_SIZE[0])
            
            patch = new_img[y1:y2, x1:x1+OUTPUT_SIZE[0]].copy()
            if patch.shape != OUTPUT_SIZE:
                patch = cv2.resize(patch, OUTPUT_SIZE)
            
            # Create segmentation mask (centered in patch)
            seg_mask = create_segmentation_mask(new_center, OUTPUT_SIZE, MASK_RADIUS)
            
            variations.append({
                "image": patch,
                "mask": seg_mask,
                "point_name": point_name,
                "original_image": os.path.basename(clean_img.filename) if hasattr(clean_img, 'filename') else f"generated_{i}"
            })
    
    return variations

def generate_ellipsegnet_dataset():
    for split in ['train', 'val', 'test']:
        os.makedirs(f"{DATASET_ROOT}/images/{split}", exist_ok=True)
        os.makedirs(f"{DATASET_ROOT}/annotations/{split}", exist_ok=True)
    
    yaml_content = f"""path: {os.path.abspath(DATASET_ROOT)}
train: images/train
val: images/val
test: images/test
"""
    with open(f"{DATASET_ROOT}/dataset.yaml", "w") as f:
        f.write(yaml_content)
    
    all_data = []
    variations_per_image = TOTAL_VARIATIONS // (len(IMAGE_PAIRS) * 2)  # Per point
    
    for pair in tqdm(IMAGE_PAIRS, desc="Processing images"):
        left_img = cv2.imread(pair["left"])
        
        if left_img is None:
            print(f"Warning: Could not load image: {pair['left']}")
            continue
        
        left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        
        left_templates = {}
        left_masks = {}
        for point_name, point_pos in pair["left_points"].items():
            template, mask = extract_target_template(left_img, point_pos, TARGET_WINDOW_SIZE)
            left_templates[point_name] = template
            left_masks[point_name] = mask
        
        clean_left = remove_targets(left_img, pair["left_points"])
        for point_name in pair["left_points"]:
            left_variations = generate_variations(
                clean_left, {point_name: left_templates[point_name]}, {point_name: left_masks[point_name]},
                {point_name: pair["left_points"][point_name]}, variations_per_image
            )
            all_data.extend(left_variations)
    
    samples = []
    for i, var in enumerate(all_data):
        samples.append({
            "image_id": f"image_{i:05d}.jpg",
            "mask": var["mask"],
            "point_name": var["point_name"],
            "width": OUTPUT_SIZE[0],
            "height": OUTPUT_SIZE[1]
        })
    
    df = pd.DataFrame(samples)
    train_df, test_df = train_test_split(df, test_size=TEST_RATIO, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=VAL_RATIO/(1-TEST_RATIO), random_state=42)
    
    def save_subset(subset_df, subset_name):
        for idx, row in subset_df.iterrows():
            var = all_data[idx]
            img_path = f"{DATASET_ROOT}/images/{subset_name}/{row['image_id']}"
            cv2.imwrite(img_path, var["image"])
            
            mask_path = f"{DATASET_ROOT}/annotations/{subset_name}/{os.path.splitext(row['image_id'])[0]}.npy"
            np.save(mask_path, var["mask"], allow_pickle=False)
    
    save_subset(train_df, "train")
    save_subset(val_df, "val")
    save_subset(test_df, "test")
    
    print(f"\nEllipSegNet dataset generation complete!")
    print(f"Total patches: {len(all_data)}")
    print(f"Train: {len(train_df)} patches")
    print(f"Validation: {len(val_df)} patches")
    print(f"Test: {len(test_df)} patches")
    print(f"Saved to: {os.path.abspath(DATASET_ROOT)}")

if __name__ == "__main__":
    if os.path.exists(DATASET_ROOT):
        shutil.rmtree(DATASET_ROOT)
    
    generate_ellipsegnet_dataset()