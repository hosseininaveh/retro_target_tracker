import cv2
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from sklearn.model_selection import train_test_split
import shutil
from skimage.draw import disk

# Configuration
DATASET_ROOT = "/content/retro_target_tracker/dataset/superpoint_dataset"
IMAGE_DIR = "/content/retro_target_tracker/dataset/retro_target_tracker"
IMAGE_COORDINATES_FILE = "extracted_frames/image_coordinates.csv"  # Path to your CSV file
TOTAL_VARIATIONS = 10000  # Increased for more diversity
OUTPUT_SIZE = (640, 480)  # Width x Height
HEATMAP_SIZE = (80, 60)   # Width/8 x Height/8
NUM_CLASSES = 3           # Background + 2 keypoints
HEATMAP_SIGMA = 3.0       # Smoother heatmaps
TARGET_WINDOW_SIZE = 21   # Marker patch size
MARGIN = 50
RANDOM_OFFSET_RANGE = 50   # Larger offsets for robustness

def load_image_observations(csv_file, image_dir):
    """Load image observations from CSV file and convert to IMAGE_PAIRS format"""
    df = pd.read_csv(csv_file)
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
        
        # Assuming CSV columns: Frame_Names, point1_col, point1_row, point2_col, point2_row
        left_point0 = (left_frame_info['point1_row'], left_frame_info['point1_col'])
        left_point1 = (left_frame_info['point2_row'], left_frame_info['point2_col'])
        right_point0 = (right_frame_info['point1_row'], right_frame_info['point1_col'])
        right_point1 = (right_frame_info['point2_row'], right_frame_info['point2_col'])
        
        image_pair = {
            "left": os.path.join(image_dir, left_frame),
            "right": os.path.join(image_dir, right_frame),
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

def extract_target_template(img, center, window_size=TARGET_WINDOW_SIZE):
    """Extract target template with precise masking."""
    h, w = img.shape[:2]
    y, x = center
    y1 = max(0, int(round(y)) - window_size // 2)
    y2 = min(h, y1 + window_size)
    x1 = max(0, int(round(x)) - window_size // 2)
    x2 = min(w, x1 + window_size)
    mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
    cy, cx = (y2 - y1) // 2, (x2 - x1) // 2
    cv2.circle(mask, (cx, cy), window_size // 2, 255, -1)
    target_window = img[y1:y2, x1:x2].copy()
    if len(target_window.shape) > 2:
        target_window = cv2.cvtColor(target_window, cv2.COLOR_BGR2GRAY)
    print(f"extract_target_template: center=({x:.1f}, {y:.1f}), target_window shape={target_window.shape}, mask shape={mask.shape}")
    cv2.imwrite(f"/tmp/debug/template_{x}_{y}.png", target_window)
    cv2.imwrite(f"/tmp/debug/mask_{x}_{y}.png", mask)
    return target_window, mask

def remove_targets(img, target_positions, window_size=TARGET_WINDOW_SIZE):
    """Remove targets using precise inpainting."""
    clean_img = img.copy()
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    for center in target_positions.values():
        y, x = center
        radius = window_size // 2 + 2
        rr, cc = disk((y, x), radius, shape=img.shape[:2])
        mask[rr, cc] = 255
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel)
    clean_img = cv2.inpaint(clean_img, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
    print(f"remove_targets: clean_img shape={clean_img.shape}")
    cv2.imwrite(f"/tmp/debug/clean_img.png", clean_img)
    return clean_img

def place_target(img, target_template, target_mask, center):
    """Place target at new position with proper blending."""
    h, w = img.shape[:2]
    temp_h, temp_w = target_template.shape[:2]
    y, x = center
    y1 = max(0, int(round(y - temp_h / 2)))
    y2 = min(h, y1 + temp_h)
    x1 = max(0, int(round(x - temp_w / 2)))
    x2 = min(w, x1 + temp_w)
    template = target_template
    tmask = target_mask
    if (y2 - y1) != temp_h or (x2 - x1) != temp_w:
        ty1 = max(0, int(temp_h / 2 - y))
        ty2 = temp_h - max(0, int(y + temp_h / 2 - h))
        tx1 = max(0, int(temp_w / 2 - x))
        tx2 = temp_w - max(0, int(x + temp_w / 2 - w))
        template = template[ty1:ty2, tx1:tx2]
        tmask = tmask[ty1:ty2, tx1:tx2]
    roi = img[y1:y2, x1:x2]
    print(f"place_target: center=({x:.1f}, {y:.1f}), roi shape={roi.shape}, template shape={template.shape}, tmask shape={tmask.shape}")
    if len(roi.shape) > 2:
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    if len(template.shape) > 2:
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    mask = tmask.astype(float) / 255.0
    img[y1:y2, x1:x2] = (roi * (1 - mask) + template * mask).astype(np.uint8)
    cv2.imwrite(f"/tmp/debug/placed_{x}_{y}.png", img)
    return img

def create_heatmap(points, img_width, img_height, heatmap_size=(80, 60), sigma=3.0):
    """Create a heatmap for SuperPoint training with Nc=2."""
    heatmap = np.zeros((NUM_CLASSES, heatmap_size[1], heatmap_size[0]), dtype=np.float32)
    scale_x = img_width / heatmap_size[0]  # 640/80 = 8
    scale_y = img_height / heatmap_size[1]  # 480/60 = 8
    for idx, (point_name, (y, x)) in enumerate(points.items(), 1):
        if idx >= NUM_CLASSES:
            break
        grid_x = x / scale_x
        grid_y = y / scale_y
        grid_x_int = min(int(round(grid_x)), heatmap_size[0] - 1)
        grid_y_int = min(int(round(grid_y)), heatmap_size[1] - 1)
        heatmap[idx, grid_y_int, grid_x_int] = 1.0
        heatmap[idx] = gaussian_filter(heatmap[idx], sigma=sigma)
        heatmap[idx] /= heatmap[idx].max() if heatmap[idx].max() > 0 else 1.0
        print(f"Keypoint {point_name}: Image=({x:.1f}, {y:.1f}), Grid=({grid_x:.3f}, {grid_y:.3f})")
    heatmap_sum = np.sum(heatmap[1:], axis=0)
    heatmap[0] = 1.0 - heatmap_sum
    heatmap[0] = np.clip(heatmap[0], 0, 1)
    heatmap /= heatmap.sum(axis=0, keepdims=True) + 1e-6  # Normalize across channels
    return heatmap

def generate_variations(clean_img, target_templates, target_masks, original_points, num_variations, original_filename):
    """Generate image variations with targets in new positions."""
    variations = []
    h, w = clean_img.shape[:2]
    for i in range(num_variations):
        new_img = clean_img.copy()
        new_points = {}
        for point_name, original_point in original_points.items():
            row_offset = np.random.uniform(-RANDOM_OFFSET_RANGE, RANDOM_OFFSET_RANGE)
            col_offset = np.random.uniform(-RANDOM_OFFSET_RANGE, RANDOM_OFFSET_RANGE)
            new_row = original_point[0] + row_offset
            new_col = original_point[1] + col_offset
            new_row = np.clip(new_row, MARGIN, h - MARGIN - 1)
            new_col = np.clip(new_col, MARGIN, w - MARGIN - 1)
            new_points[point_name] = (new_row, new_col)
        sorted_points = sorted(new_points.items(), key=lambda x: x[1][1])
        for idx, (point_name, new_center) in enumerate(sorted_points):
            template = target_templates[point_name]
            mask = target_masks[point_name]
            new_img = place_target(new_img, template, mask, new_center)
        if OUTPUT_SIZE != (w, h):
            scale_x = OUTPUT_SIZE[0] / w
            scale_y = OUTPUT_SIZE[1] / h
            new_img = cv2.resize(new_img, OUTPUT_SIZE, interpolation=cv2.INTER_LINEAR)
            sorted_points = [
                (name, (row * scale_y, col * scale_x))
                for name, (row, col) in sorted_points
            ]
        heatmap = create_heatmap(dict(sorted_points), OUTPUT_SIZE[0], OUTPUT_SIZE[1], HEATMAP_SIZE)
        variations.append({
            "image": new_img,
            "points": dict(sorted_points),
            "heatmap": heatmap,
            "original_image": f"{os.path.basename(original_filename)}_var_{i}"
        })
        cv2.imwrite(f"/tmp/debug/variation_{i}.png", new_img)
    return variations

def generate_superpoint_dataset():
    """Main function to generate the SuperPoint formatted dataset."""
    # Load image observations from CSV
    print("Loading image observations from CSV...")
    IMAGE_PAIRS = load_image_observations(IMAGE_COORDINATES_FILE, IMAGE_DIR)
    
    if not IMAGE_PAIRS:
        print(f"Error: No image pairs loaded from {IMAGE_COORDINATES_FILE}")
        return
    
    print(f"Loaded {len(IMAGE_PAIRS)} image pairs from CSV")
    
    os.makedirs("/tmp/debug", exist_ok=True)
    if os.path.exists(DATASET_ROOT):
        shutil.rmtree(DATASET_ROOT)
    for split in ['train', 'val', 'test']:
        os.makedirs(f"{DATASET_ROOT}/images/{split}", exist_ok=True)
        os.makedirs(f"{DATASET_ROOT}/annotations/{split}", exist_ok=True)
    
    yaml_content = f"""path: {os.path.abspath(DATASET_ROOT)}
train: images/train
val: images/val
test: images/test
nc: {NUM_CLASSES}
names: ['background', 'point0', 'point1']
"""
    with open(f"{DATASET_ROOT}/dataset.yaml", "w") as f:
        f.write(yaml_content)
    
    all_data = []
    variations_per_pair = TOTAL_VARIATIONS // (len(IMAGE_PAIRS) * 2)
    
    for pair_idx, pair in enumerate(tqdm(IMAGE_PAIRS, desc="Processing image pairs")):
        left_path = pair["left"]
        right_path = pair["right"]
        left_img = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
        right_img = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)
        
        if left_img is None or right_img is None:
            print(f"Warning: Could not load image pair: {left_path}, {right_path}")
            continue
        
        print(f"Loaded left_img shape={left_img.shape}, right_img shape={right_img.shape}")
        if left_img.shape != OUTPUT_SIZE[::-1]:
            left_img = cv2.resize(left_img, OUTPUT_SIZE, interpolation=cv2.INTER_LINEAR)
        if right_img.shape != OUTPUT_SIZE[::-1]:
            right_img = cv2.resize(right_img, OUTPUT_SIZE, interpolation=cv2.INTER_LINEAR)
        
        left_templates = {}
        left_masks = {}
        for point_name, (row, col) in pair["left_points"].items():
            template, mask = extract_target_template(left_img, (row, col), TARGET_WINDOW_SIZE)
            left_templates[point_name] = template
            left_masks[point_name] = mask
        clean_left = remove_targets(left_img, pair["left_points"])
        left_variations = generate_variations(
            clean_left, left_templates, left_masks, pair["left_points"], variations_per_pair, pair["left"]
        )
        
        right_templates = {}
        right_masks = {}
        for point_name, (row, col) in pair["right_points"].items():
            template, mask = extract_target_template(right_img, (row, col), TARGET_WINDOW_SIZE)
            right_templates[point_name] = template
            right_masks[point_name] = mask
        clean_right = remove_targets(right_img, pair["right_points"])
        right_variations = generate_variations(
            clean_right, right_templates, right_masks, pair["right_points"], variations_per_pair, pair["right"]
        )
        
        all_data.extend(left_variations)
        all_data.extend(right_variations)
    
    if not all_data:
        raise ValueError("No valid image pairs loaded. Check image paths in IMAGE_PAIRS.")
    
    samples = []
    for i, var in enumerate(all_data):
        samples.append({
            "image_id": f"image_{i:05d}.jpg",
            "points": var["points"],
            "heatmap": var["heatmap"],
            "width": var["image"].shape[1],
            "height": var["image"].shape[0]
        })
    
    df = pd.DataFrame(samples)
    train_df, test_df = train_test_split(df, test_size=0.15, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.15 / (1 - 0.15), random_state=42)
    
    def save_subset(subset_df, subset_name):
        for idx, row in subset_df.iterrows():
            var = all_data[idx]
            img_path = f"{DATASET_ROOT}/images/{subset_name}/{row['image_id']}"
            cv2.imwrite(img_path, var["image"])
            annotation = ""
            for point_name, (y, x) in var["points"].items():
                annotation += f"{point_name} {x:.6f} {y:.6f}\n"
            txt_path = f"{DATASET_ROOT}/annotations/{subset_name}/{os.path.splitext(row['image_id'])[0]}.txt"
            with open(txt_path, "w") as f:
                f.write(annotation)
            heatmap_path = f"{DATASET_ROOT}/annotations/{subset_name}/{os.path.splitext(row['image_id'])[0]}.npy"
            np.save(heatmap_path, var["heatmap"], allow_pickle=False)
    
    save_subset(train_df, "train")
    save_subset(val_df, "val")
    save_subset(test_df, "test")
    
    print(f"\nSuperPoint dataset generation complete!")
    print(f"Total images: {len(all_data)}")
    print(f"Train: {len(train_df)} images")
    print(f"Validation: {len(val_df)} images")
    print(f"Test: {len(test_df)} images")
    print(f"Saved to: {os.path.abspath(DATASET_ROOT)}")
    print(f"Dataset config file: {os.path.abspath(DATASET_ROOT)}/dataset.yaml")
    print(f"Debug images saved to: /tmp/debug")

if __name__ == "__main__":
    generate_superpoint_dataset()