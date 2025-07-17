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
DATASET_ROOT = "superpoint_dataset"
IMAGE_PAIRS = [
    {
        "left": "./left_frame_001.jpg",
        "right": "./right_frame_001.jpg",
        "left_points": {
            "point0": (240, 298),  # (row, col)
            "point1": (234, 502)
        },
        "right_points": {
            "point0": (240, 130),
            "point1": (232, 305)
        }
    },
    {
        "left": "./left_frame_151.jpg",
        "right": "./right_frame_151.jpg",
        "left_points": {
            "point0": (242, 333),  # (row, col)
            "point1": (231, 526)
        },
        "right_points": {
            "point0": (242, 154),
            "point1": (228, 350)
        }
    },
    {
        "left": "./left_frame_137.jpg",
        "right": "./right_frame_137.jpg",
        "left_points": {
            "point0": (243, 393),  # (row, col)
            "point1": (225, 564)
        },
        "right_points": {
            "point0": (242, 206),
            "point1": (219, 411)
        }
    },
    {
        "left": "./left_frame_034.jpg",
        "right": "./right_frame_034.jpg",
        "left_points": {
            "point0": (242, 383),  # (row, col)
            "point1": (226, 550)
        },
        "right_points": {
            "point0": (242, 191),
            "point1": (222, 396)
        }
    },
    {
        "left": "./left_frame_011.jpg",
        "right": "./right_frame_011.jpg",
        "left_points": {
            "point0": (240, 281),  # (row, col)
            "point1": (236, 488)
        },
        "right_points": {
            "point0": (239, 118),
            "point1": (233, 286)
        }
    },
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
    {
        "left": "./left_frame.jpg",
        "right": "./right_frame.jpg",
        "left_points": {
            "point0": (272, 396),
            "point1": (269, 412)
        },
        "right_points": {
            "point0": (271, 214),
            "point1": (270, 230)
        }
    },
    {
        "left": "./left_frame_2562.jpg",
        "right": "./right_frame_2562.jpg",
        "left_points": {
            "point0": (272, 396),
            "point1": (268, 438)
        },
        "right_points": {
            "point0": (271, 214),
            "point1": (267, 259)
        }
    },
    {
        "left": "./left_frame_3146.jpg",
        "right": "./right_frame_3146.jpg",
        "left_points": {
            "point0": (272, 396),
            "point1": (268, 428)
        },
        "right_points": {
            "point0": (271, 214),
            "point1": (268, 248)
        }
    },
    {
        "left": "./left_frame_737.jpg",
        "right": "./right_frame_737.jpg",
        "left_points": {
            "point0": (272, 396),
            "point1": (270, 418)
        },
        "right_points": {
            "point0": (271, 214),
            "point1": (269, 237)
        }
    },
]
TOTAL_VARIATIONS = 5000  # Total across all image pairs
TARGET_WINDOW_SIZE = 15  # Size of marker region for template extraction
MARGIN = 50
RANDOM_OFFSET_RANGE = 30
OUTPUT_SIZE = (640, 480)  # SuperPoint input size (match your stereo_3d_tracker2)
TEST_RATIO = 0.15
VAL_RATIO = 0.15
HEATMAP_SIZE = (30, 40)  # Matches out_det shape [2, 65, 30, 40]
HEATMAP_SIGMA = 2.0  # For Gaussian heatmap

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

def create_heatmap(points, img_width, img_height, heatmap_size=(30, 40), sigma=2.0):
    """Create a heatmap for SuperPoint training"""
    heatmap = np.zeros((65, heatmap_size[0], heatmap_size[1]), dtype=np.float32)
    scale_x = heatmap_size[1] / img_width  # 40 / 640
    scale_y = heatmap_size[0] / img_height  # 30 / 480
    
    for point_name, (y, x) in points.items():
        # Map pixel coordinates to heatmap grid
        grid_x = min(int(x * scale_x), heatmap_size[1] - 1)
        grid_y = min(int(y * scale_y), heatmap_size[0] - 1)
        
        # Create a Gaussian blob
        heatmap[0, grid_y, grid_x] = 1.0  # Single point activation
        # Apply Gaussian blur
        heatmap[0] = gaussian_filter(heatmap[0], sigma=sigma)
        # Normalize to [0, 1]
        heatmap[0] = heatmap[0] / (heatmap[0].max() + 1e-6)
    
    return heatmap

def generate_variations(clean_img, target_templates, target_masks, original_points, num_variations):
    """Generate image variations with targets in new positions"""
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
            
            new_row = np.clip(new_row, MARGIN, h-MARGIN-1)
            new_col = np.clip(new_col, MARGIN, w-MARGIN-1)
            
            new_points[point_name] = (new_row, new_col)
        
        sorted_points = sorted(new_points.items(), key=lambda x: x[1][1])
        new_points = {name: pos for name, pos in sorted_points}
        
        for point_name, new_center in new_points.items():
            template = target_templates[point_name]
            mask = target_masks[point_name]
            new_img = place_target(new_img, template, mask, new_center)
        
        if OUTPUT_SIZE != (w, h):
            new_img = cv2.resize(new_img, OUTPUT_SIZE)
            scale_x = OUTPUT_SIZE[0] / w
            scale_y = OUTPUT_SIZE[1] / h
            new_points = {
                name: (row * scale_y, col * scale_x)
                for name, (row, col) in new_points.items()
            }
        
        # Generate heatmap
        heatmap = create_heatmap(new_points, OUTPUT_SIZE[0], OUTPUT_SIZE[1], HEATMAP_SIZE)
        
        variations.append({
            "image": new_img,
            "points": new_points,
            "heatmap": heatmap,
            "original_image": os.path.basename(clean_img.filename) if hasattr(clean_img, 'filename') else f"generated_{i}"
        })
    
    return variations

def generate_superpoint_dataset():
    """Main function to generate the SuperPoint formatted dataset"""
    # Create directory structure
    for split in ['train', 'val', 'test']:
        os.makedirs(f"{DATASET_ROOT}/images/{split}", exist_ok=True)
        os.makedirs(f"{DATASET_ROOT}/annotations/{split}", exist_ok=True)
    
    # Create dataset.yaml file
    yaml_content = f"""path: {os.path.abspath(DATASET_ROOT)}
train: images/train
val: images/val
test: images/test
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
        
        # Convert to grayscale
        left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
        
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
            "heatmap": var["heatmap"],
            "width": img_width,
            "height": img_height
        })
    
    # Split dataset
    df = pd.DataFrame(samples)
    train_df, test_df = train_test_split(df, test_size=TEST_RATIO, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=VAL_RATIO/(1-TEST_RATIO), random_state=42)
    
    # Save images, annotations, and heatmaps
    def save_subset(subset_df, subset_name):
        for idx, row in subset_df.iterrows():
            var = all_data[idx]
            
            # Save image
            img_path = f"{DATASET_ROOT}/images/{subset_name}/{row['image_id']}"
            cv2.imwrite(img_path, var["image"])
            
            # Save annotation (keypoint coordinates)
            annotation = ""
            for point_name, (y, x) in var["points"].items():
                annotation += f"{point_name} {x:.6f} {y:.6f}\n"
            txt_path = f"{DATASET_ROOT}/annotations/{subset_name}/{os.path.splitext(row['image_id'])[0]}.txt"
            with open(txt_path, "w") as f:
                f.write(annotation)
            
            # Save heatmap
            heatmap_path = f"{DATASET_ROOT}/annotations/{subset_name}/{os.path.splitext(row['image_id'])[0]}.npy"
            np.save(heatmap_path, var["heatmap"])
    
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

if __name__ == "__main__":
    # Clear existing dataset if it exists
    if os.path.exists(DATASET_ROOT):
        shutil.rmtree(DATASET_ROOT)
    
    generate_superpoint_dataset()