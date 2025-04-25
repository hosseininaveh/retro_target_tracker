import os
import cv2
import torch
import numpy as np
import pandas as pd
import random
from torch.utils.data import Dataset

class Augmentation:
    """Data augmentation pipeline"""
    def __call__(self, image, heatmap):
        # Random horizontal flip
        if random.random() > 0.5:
            image = cv2.flip(image, 1)
            heatmap = cv2.flip(heatmap, 1)
            # Adjust x-coordinate if flipped
            heatmap = np.fliplr(heatmap)
        
        # Random rotation (-15 to 15 degrees)
        angle = random.uniform(-15, 15)
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        image = cv2.warpAffine(image, M, (w, h))
        heatmap = cv2.warpAffine(heatmap, M, (w, h))
        
        # Random brightness/contrast
        alpha = random.uniform(0.8, 1.2)  # Contrast control
        beta = random.uniform(-0.1, 0.1)  # Brightness control
        image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        
        return image, heatmap

class TargetDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform if transform else Augmentation()  # Default augmentation
        self.annotations = self._load_annotations()
        
        if self.annotations is None:
            raise ValueError(
                f"Annotations not loaded. Check:\n"
                f"1. File exists at: {os.path.join(image_dir, 'annotations/annotations.csv')}\n"
                f"2. CSV has columns: 'image_path', 'x', 'y'\n"
                f"3. First image path exists at: {os.path.join(image_dir, 'images', self._get_sample_image_path())}"
            )


    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]
        img_path = os.path.join(self.image_dir, 'images', row['image_path'])
        
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {img_path}")
        
        # Convert BGR to RGB and normalize
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
        target = (row['x'], row['y'])
        heatmap = self._create_heatmap(image.shape[:2], target)
        
        if self.transform:
            image, heatmap = self.transform(image, heatmap)
            
        # Convert to channels-first format
        image = np.moveaxis(image, -1, 0)
        return torch.from_numpy(image).float(), torch.from_numpy(heatmap).float()

    def _create_heatmap(self, img_shape, center, sigma=3):
        h, w = img_shape
        y, x = np.indices((h, w))
        return np.exp(-((x - center[0])**2 + (y - center[1])**2) / (2 * sigma**2))

    def _load_annotations(self):
        try:
            csv_path = os.path.join(self.image_dir, 'annotations/annotations.csv')
            df = pd.read_csv(csv_path)
            
            # Validate required columns
            if not all(col in df.columns for col in ['image_path', 'x', 'y']):
                print("CSV missing required columns. Needs: 'image_path', 'x', 'y'")
                return None
                
            return df
        except Exception as e:
            print(f"Error loading annotations: {str(e)}")
            return None

    def _get_sample_image_path(self):
        """Helper to get a sample image path for error messages"""
        try:
            csv_path = os.path.join(self.image_dir, 'annotations/annotations.csv')
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                return df.iloc[0]['image_path'] if len(df) > 0 else "no_images_in_csv"
        except:
            return "cannot_access_csv"
        return "unknown_path"