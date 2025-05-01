import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class TargetDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        
        # Load annotations
        annot_path = os.path.join(image_dir, 'annotations/annotations.csv')
        self.annotations = pd.read_csv(annot_path)
        
        # Verify required columns
        required_cols = ['image_path', 'x0', 'y0', 'x1', 'y1']
        if not all(col in self.annotations.columns for col in required_cols):
            raise ValueError(
                f"CSV missing required columns. Needs: {required_cols}\n"
                f"Found: {self.annotations.columns.tolist()}"
            )
        
        # Verify first image exists
        first_img = os.path.join(image_dir, 'images', self.annotations.iloc[0]['image_path'])
        if not os.path.exists(first_img):
            raise ValueError(f"First image path does not exist: {first_img}")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, 'images', self.annotations.iloc[idx]['image_path'])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
        
        # Get both points
        points = np.array([
            [self.annotations.iloc[idx]['x0'], self.annotations.iloc[idx]['y0']],  # point0
            [self.annotations.iloc[idx]['x1'], self.annotations.iloc[idx]['y1']]   # point1
        ], dtype=np.float32)
        
        if self.transform:
            if isinstance(self.transform, A.Compose):
                transformed = self.transform(image=image, keypoints=points)
                image = transformed['image']  # This will be numpy array in HWC format
                # Convert to CHW format immediately
                image = torch.from_numpy(image).permute(2, 0, 1).float()
                points = np.array(transformed['keypoints'])
            else:
                image, points = self.transform(image, points)
        
        return image, points

class Augmentation:
    """Albumentations-based augmentation pipeline"""
    def __init__(self):
        self.transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ], keypoint_params=A.KeypointParams(format='xy'))
    
    def __call__(self, image, points):
        # Convert points to list of tuples for albumentations
        points_list = [tuple(p) for p in points]
        transformed = self.transform(image=image, keypoints=points_list)
        return transformed['image'], np.array(transformed['keypoints'])