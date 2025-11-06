import cv2
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import shutil
from sklearn.model_selection import train_test_split

# Configuration
DATASET_ROOT = "/content/retro_target_tracker/dataset/ellipsegnet_dataset"
IMAGE_ROOT = "/content/retro_target_tracker/images"
IMAGE_COORDINATES_FILE = "extracted_frames/image_coordinates.csv"  # Path to your CSV file
TOTAL_VARIATIONS = 1000
TARGET_WINDOW_SIZE = 15
MARGIN = 50
RANDOM_OFFSET_RANGE = 30
OUTPUT_SIZE = (120, 120)  # Matches crop_sz in config.txt
TEST_RATIO = 0.15
VAL_RATIO = 0.15
MASK_SIZE = (120, 120)
MASK_AXES = (10, 10)  # Ellipse axes for retro-reflective targets
MASK_ANGLE = 0  # Rotation angle for ellipse

def load_image_observations(csv_file, image_root):
    """Load image observations from CSV file and convert to IMAGE_PAIRS format"""
    # Read CSV with tab delimiter
    df = pd.read_csv(csv_file, delimiter='\t')
    image_pairs = []
    
    # Group frames by pairs (assuming alternating left/right frames)
    for i in range(0, len(df), 2):
        if i + 1 >= len(df):
            break
            
        left_frame_info = df.iloc[i]
        right_frame_info = df.iloc[i + 1]
        
        # Extract frame names and coordinates
        left_frame = left_frame_info['Frame_Names']
        right_frame = right_frame_info['Frame_Names']
        
        # CSV columns: Frame_Names, point1_col, point1_row, point2_col, point2_row
        # Note: This code uses (col, row) format for points
        left_point0 = (left_frame_info['point1_col'], left_frame_info['point1_row'])
        left_point1 = (left_frame_info['point2_col'], left_frame_info['point2_row'])
        right_point0 = (right_frame_info['point1_col'], right_frame_info['point1_row'])
        right_point1 = (right_frame_info['point2_col'], right_frame_info['point2_row'])
        
        image_pair = {
            "left": os.path.join(image_root, left_frame),
            "right": os.path.join(image_root, right_frame),
            "left_points": {
                "point0": left_point0,
                "point1": left_point1
            },
            "right_points": {
                "point0": right_point0,
                "point1": right_point1
            }
        }
        image_pairs.append(image_pair)
    
    return image_pairs

def extract_target_template(img, center, window_size=15):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape
    y, x = int(center[1]), int(center[0])  # (row, col)
    
    y1 = max(0, y - window_size//2)
    y2 = min(h, y + window_size//2 + 1)
    x1 = max(0, x - window_size//2)
    x2 = min(w, x + window_size//2 + 1)
    
    mask = np.zeros((y2-y1, x2-x1), dtype=np.uint8)
    cy, cx = (y2-y1)//2, (x2-x1)//2
    cv2.ellipse(mask, (cx, cy), (window_size//2, window_size//2), 0, 0, 360, 255, -1)
    
    target_window = img[y1:y2, x1:x2].copy()
    return target_window, mask

def remove_targets(img, target_positions, window_size=15):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clean_img = img.copy()
    mask = np.zeros(img.shape, dtype=np.uint8)
    
    for center in target_positions.values():
        y, x = int(center[1]), int(center[0])  # (row, col)
        radius = window_size//2 + 2
        cv2.circle(mask, (x, y), radius, 255, -1)
    
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
    
    y, x = int(center[1]), int(center[0])  # (row, col)
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

def create_segmentation_mask(img_size=(120, 120), axes=(10, 10), angle=0):
    mask = np.zeros(img_size, dtype=np.float32)
    center = (img_size[1]//2, img_size[0]//2)  # (x, y) for OpenCV
    cv2.ellipse(mask, center, axes, angle, 0, 360, 1.0, -1)
    return mask

def generate_variations(clean_img, target_templates, target_masks, original_points, num_variations, side, pair_idx):
    variations = []
    h, w = clean_img.shape
    
    for i in range(num_variations):
        new_img = clean_img.copy()
        new_points = {}
        
        for point_name, original_point in original_points.items():
            col_offset = np.random.uniform(-RANDOM_OFFSET_RANGE, RANDOM_OFFSET_RANGE)
            row_offset = np.random.uniform(-RANDOM_OFFSET_RANGE, RANDOM_OFFSET_RANGE)
            
            new_col = original_point[0] + col_offset  # col
            new_row = original_point[1] + row_offset  # row
            
            new_row = np.clip(new_row, MARGIN, h-MARGIN-1)
            new_col = np.clip(new_col, MARGIN, w-MARGIN-1)
            
            new_points[point_name] = (new_col, new_row)
        
        sorted_points = sorted(new_points.items(), key=lambda x: x[1][0])  # Sort by col
        new_points = {name: pos for name, pos in sorted_points}
        
        for point_name, new_center in new_points.items():
            template = target_templates[point_name]
            mask = target_masks[point_name]
            new_img = place_target(new_img, template, mask, new_center)
            
            # Crop 120x120 patch around the point
            y, x = int(new_center[1]), int(new_center[0])  # (row, col)
            y1 = max(0, y - OUTPUT_SIZE[0]//2)
            y2 = min(h, y1 + OUTPUT_SIZE[0])
            x1 = max(0, x - OUTPUT_SIZE[1]//2)
            x2 = min(w, x1 + OUTPUT_SIZE[1])
            
            patch = new_img[y1:y2, x1:x2].copy()
            if patch.shape != OUTPUT_SIZE:
                patch = cv2.resize(patch, OUTPUT_SIZE, interpolation=cv2.INTER_AREA)
            
            patch = patch.astype(np.float32) / 255.0  # Normalize to [0, 1]
            
            # Create segmentation mask (centered in patch)
            seg_mask = create_segmentation_mask(OUTPUT_SIZE, MASK_AXES, MASK_ANGLE)
            
            variations.append({
                "image": patch,
                "mask": seg_mask,
                "point_name": point_name,
                "side": side,
                "pair_idx": pair_idx
            })
    
    return variations

def generate_ellipsegnet_dataset():
    # Load image observations from CSV
    print("Loading image observations from CSV...")
    IMAGE_PAIRS = load_image_observations(IMAGE_COORDINATES_FILE, IMAGE_ROOT)
    
    if not IMAGE_PAIRS:
        print(f"Error: No image pairs loaded from {IMAGE_COORDINATES_FILE}")
        return
    
    print(f"Loaded {len(IMAGE_PAIRS)} image pairs from CSV")
    
    # Create directories
    for split in ['train', 'val', 'test']:
        os.makedirs(f"{DATASET_ROOT}/ellipsegnet_patches/{split}", exist_ok=True)
        os.makedirs(f"{DATASET_ROOT}/ellipsegnet_masks/{split}", exist_ok=True)
    
    all_data = []
    variations_per_point = TOTAL_VARIATIONS // (len(IMAGE_PAIRS) * 2 * 2)  # Per point (left + right, 2 points each)
    print(f"Generating {variations_per_point} variations per point")
    
    for pair_idx, pair in enumerate(tqdm(IMAGE_PAIRS, desc="Processing image pairs")):
        for side, points_key in [('left', 'left_points'), ('right', 'right_points')]:
            img_path = pair[side]
            img = cv2.imread(img_path)
            
            if img is None:
                print(f"Warning: Could not load image: {img_path}")
                continue
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            templates = {}
            masks = {}
            for point_name, point_pos in pair[points_key].items():
                template, mask = extract_target_template(img, point_pos, TARGET_WINDOW_SIZE)
                templates[point_name] = template
                masks[point_name] = mask
            
            clean_img = remove_targets(img, pair[points_key])
            variations = generate_variations(
                clean_img, templates, masks, pair[points_key], variations_per_point, side, pair_idx
            )
            all_data.extend(variations)
    
    # Split data
    samples = []
    for i, var in enumerate(all_data):
        samples.append({
            "image_id": f"patch_{var['pair_idx']:03d}_{var['side']}_{var['point_name']}_{i:05d}.npy",
            "mask_id": f"mask_{var['pair_idx']:03d}_{var['side']}_{var['point_name']}_{i:05d}.npy",
            "point_name": var["point_name"],
            "side": var["side"],
            "width": OUTPUT_SIZE[0],
            "height": OUTPUT_SIZE[1]
        })
    
    df = pd.DataFrame(samples)
    train_df, test_df = train_test_split(df, test_size=TEST_RATIO, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=VAL_RATIO/(1-TEST_RATIO), random_state=42)
    
    def save_subset(subset_df, subset_name):
        for idx, row in subset_df.iterrows():
            var = all_data[idx]
            img_path = f"{DATASET_ROOT}/ellipsegnet_patches/{subset_name}/{row['image_id']}"
            mask_path = f"{DATASET_ROOT}/ellipsegnet_masks/{subset_name}/{row['mask_id']}"
            np.save(img_path, var["image"], allow_pickle=False)
            np.save(mask_path, var["mask"], allow_pickle=False)
    
    save_subset(train_df, "train")
    save_subset(val_df, "val")
    save_subset(test_df, "test")
    
    # Create dataset.yaml
    yaml_content = f"""path: {os.path.abspath(DATASET_ROOT)}
train: ellipsegnet_patches/train
val: ellipsegnet_patches/val
test: ellipsegnet_patches/test
"""
    with open(f"{DATASET_ROOT}/dataset.yaml", "w") as f:
        f.write(yaml_content)
    
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