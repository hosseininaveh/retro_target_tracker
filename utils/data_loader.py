import cv2
import torch
from torch.utils.data import Dataset

class TargetDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        # Load annotations from CSV or JSON
        self.annotations = self._load_annotations()  

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = self.annotations[idx]['image_path']
        image = cv2.imread(img_path)
        target = self.annotations[idx]['target_pos']  # [x,y]
        
        # Create Gaussian heatmap
        heatmap = self._create_heatmap(image.shape[:2], target)
        
        if self.transform:
            image, heatmap = self.transform(image, heatmap)
            
        return torch.from_numpy(image).float(), torch.from_numpy(heatmap).float()

    def _create_heatmap(self, img_shape, center, sigma=3):
        """Generate 2D Gaussian heatmap"""
        h, w = img_shape
        y, x = np.indices((h, w))
        heatmap = np.exp(-((x - center[0])**2 + (y - center[1])**2) / (2 * sigma**2))
        return heatmap

    def _load_annotations(self):
        # Implement your annotation loading logic
        pass

class Augmentation:
    """Data augmentation pipeline"""
    def __call__(self, image, heatmap):
        # Random flip
        if np.random.rand() > 0.5:
            image = cv2.flip(image, 1)
            heatmap = cv2.flip(heatmap, 1)
        
        # Add noise
        if np.random.rand() > 0.5:
            noise = np.random.randn(*image.shape) * 5
            image = np.clip(image + noise, 0, 255).astype(np.uint8)
            
        return image, heatmap
