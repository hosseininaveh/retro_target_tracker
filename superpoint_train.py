import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import os
from tqdm import tqdm
import torch.nn.functional as F

# Configuration
CHECKPOINT_DIR = "/content/retro_target_tracker/checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class SuperPointDataset(Dataset):
    def __init__(self, dataset_dir, split, img_size=(640, 480)):
        print(f"Initializing dataset for split: {split}")
        self.dataset_dir = dataset_dir
        self.split = split
        self.img_size = img_size
        self.image_dir = os.path.join(dataset_dir, f"images/{split}")
        self.annotation_dir = os.path.join(dataset_dir, f"annotations/{split}")
        
        if not os.path.exists(self.image_dir):
            raise FileNotFoundError(f"Image directory {self.image_dir} does not exist")
        if not os.path.exists(self.annotation_dir):
            raise FileNotFoundError(f"Annotation directory {self.annotation_dir} does not exist")
        
        self.image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.jpg')])
        if not self.image_files:
            raise ValueError(f"No .jpg files found in {self.image_dir}")
        print(f"Found {len(self.image_files)} images in {self.image_dir}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        heatmap_path = os.path.join(self.annotation_dir, os.path.splitext(img_name)[0] + '.npy')
        
        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Failed to load image: {img_path}")
            img = cv2.resize(img, self.img_size, interpolation=cv2.INTER_LINEAR)
            img = torch.from_numpy(img).float() / 255.0
            img = img.unsqueeze(0)  # [1, 480, 640]
            
            heatmap = np.load(heatmap_path, allow_pickle=False).astype(np.float32)  # [3, 60, 80]
            if heatmap.shape != (3, 60, 80):
                raise ValueError(f"Invalid heatmap shape {heatmap.shape} for {heatmap_path}, expected (3, 60, 80)")
            heatmap = torch.from_numpy(heatmap).float()  # [3, 60, 80]
            heatmap = heatmap / (heatmap.sum(dim=0, keepdims=True) + 1e-6)  # Normalize across channels
            heatmap = torch.cat([heatmap, torch.zeros(1, 60, 80)], dim=0)  # [4, 60, 80]
            
            det_target = torch.zeros((65, 60, 80), dtype=torch.float32)
            for c in range(1, 3):
                hmap = heatmap[c]
                max_val = hmap.max().item()
                if max_val == 0:
                    continue
                max_idx = torch.where(hmap == max_val)
                y, x = max_idx[0][0], max_idx[1][0]
                cell_idx = (y // 8) * 8 + (x // 8)
                det_target[cell_idx, y, x] = 1.0
            det_target[64] = 1.0 - torch.sum(det_target[:64], dim=0)
            det_target[64] = torch.clamp(det_target[64], 0, 1)
            
            return img, heatmap[:3], det_target, img_name  # Return [3, 60, 80] for cls
        except Exception as e:
            print(f"Error loading sample {img_name}: {e}")
            raise
    
    def verify_files(self):
        print(f"Verifying files in {self.image_dir}")
        for img_name in self.image_files:
            img_path = os.path.join(self.image_dir, img_name)
            heatmap_path = os.path.join(self.annotation_dir, os.path.splitext(img_name)[0] + '.npy')
            if not os.path.exists(heatmap_path):
                print(f"Missing heatmap for {img_name}: {heatmap_path}")
            try:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"Corrupt image: {img_path}")
                heatmap = np.load(heatmap_path, allow_pickle=False)
                if heatmap.shape != (3, 60, 80):
                    print(f"Invalid heatmap shape {heatmap.shape} for {heatmap_path}")
            except Exception as e:
                print(f"Error verifying {img_name}: {e}")

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)

class SuperPointNet(nn.Module):
    def __init__(self, Nc=2):
        super(SuperPointNet, self).__init__()
        self.pool = nn.MaxPool2d(2, stride=2)
        self.conv1 = DoubleConv(1, 64)
        self.conv2 = DoubleConv(64, 64)
        self.conv3 = DoubleConv(64, 128)
        self.conv4 = DoubleConv(128, 128)
        self.convD = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 65, 1)
        )
        self.convC = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, Nc+1, 1)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.pool(x)
        x = self.conv4(x)
        det = self.convD(x)
        cls = self.convC(x)
        det = torch.softmax(det, dim=1)
        cls = torch.softmax(cls, dim=1)
        return det, cls

def focal_loss(pred, target, alpha=0.75, gamma=2.0):
    bce = -(target * torch.log(pred + 1e-6) + (1.0 - target) * torch.log(1.0 - pred + 1e-6))
    pt = torch.exp(-bce)
    focal_loss = alpha * (1.0 - pt) ** gamma * bce
    return focal_loss.mean()

def weighted_mse_loss(pred, target, weight_positive=1000.0):
    mse = (pred - target) ** 2
    weight = torch.ones_like(target)
    weight[:, :64, :, :] = weight_positive
    return (mse * weight).mean()

def extract_keypoints_from_heatmap(cls, det, conf_threshold=0.5):
    if cls.ndim == 4:
        cls = cls[0]
        det = det[0]
    height, width = cls.shape[1:]  # [60, 80]
    keypoints = []
    for c in range(1, 3):
        hmap = cls[c]
        max_val = hmap.max()
        if max_val < conf_threshold:
            print(f"Channel {c} skipped: max value {max_val:.8f} < {conf_threshold}")
            continue
        y, x = np.unravel_index(np.argmax(hmap), hmap.shape)
        img_x, img_y = x * (640 / width), y * (480 / height)
        keypoints.append([img_x, img_y, max_val])
        print(f"Keypoint (channel {c}): Image=({img_x:.3f}, {img_y:.3f}), Score={max_val:.8f}")
    return np.array(keypoints) if len(keypoints) > 0 else np.empty((0, 3), dtype=np.float32)

def load_specific_image(img_name, img_dir, annot_dir, img_size=(640, 480)):
    img_path = os.path.join(img_dir, img_name)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Failed to load {img_path}")
        return None, None, None
    img = cv2.resize(img, img_size, interpolation=cv2.INTER_LINEAR)
    img = torch.from_numpy(img).float() / 255.0
    img = img.unsqueeze(0).unsqueeze(0)  # [1, 1, 480, 640]
    
    annot_path = os.path.join(annot_dir, img_name.replace('.jpg', '.npy'))
    heatmap = np.load(annot_path, allow_pickle=False).astype(np.float32)  # [3, 60, 80]
    if heatmap.shape != (3, 60, 80):
        print(f"Invalid heatmap shape {heatmap.shape} for {annot_path}")
        return None, None, None
    heatmap = torch.from_numpy(heatmap).float()  # [3, 60, 80]
    heatmap = heatmap / (heatmap.sum(dim=0, keepdims=True) + 1e-6)  # Normalize across channels
    heatmap = torch.cat([heatmap, torch.zeros(1, 60, 80)], dim=0)  # [4, 60, 80]
    
    det_target = torch.zeros((65, 60, 80), dtype=torch.float32)
    for c in range(1, 3):
        hmap = heatmap[c]
        max_val = hmap.max().item()
        if max_val == 0:
            continue
        max_idx = torch.where(hmap == max_val)
        y, x = max_idx[0][0], max_idx[1][0]
        cell_idx = (y // 8) * 8 + (x // 8)
        det_target[cell_idx, y, x] = 1.0
    det_target[64] = 1.0 - torch.sum(det_target[:64], dim=0)
    det_target[64] = torch.clamp(det_target[64], 0, 1)
    
    return img, heatmap[:3], det_target  # Return [3, 60, 80] for cls

def train_model():
    # Initialize datasets
    train_dataset = SuperPointDataset(
        '/content/retro_target_tracker/superpoint_dataset', 'train'
    )
    val_dataset = SuperPointDataset(
        '/content/retro_target_tracker/superpoint_dataset', 'val'
    )
    
    print("Verifying dataset integrity...")
    train_dataset.verify_files()
    val_dataset.verify_files()
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)
    print(f"Train dataset size: {len(train_dataset)} samples")
    print(f"Validation dataset size: {len(val_dataset)} samples")
    
    # Initialize model and optimizer
    model = SuperPointNet(Nc=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Reduced LR
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, min_lr=1e-6, verbose=True)
    num_epochs = 150  # Increased for better convergence
    best_val_loss = float('inf')
    
    # Classifier head weights
    weights = torch.tensor([1.0, 75.0, 75.0]).to(device)  # [background, point0, point1]
    
    # Monitor specific images
    monitor_images = ['image_00000.jpg', 'image_00983.jpg', 'image_02149.jpg']
    monitor_data = {}
    for img_name in monitor_images:
        img, heatmap, det_target = load_specific_image(
            img_name,
            '/content/retro_target_tracker/superpoint_dataset/images/train',
            '/content/retro_target_tracker/superpoint_dataset/annotations/train'
        )
        if img is not None:
            monitor_data[img_name] = (img.to(device), heatmap.to(device), det_target.to(device))
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for imgs, heatmaps, det_targets, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            imgs, heatmaps, det_targets = imgs.to(device), heatmaps.to(device), det_targets.to(device)
            optimizer.zero_grad()
            det, cls = model(imgs)
            cls_loss = focal_loss(cls, heatmaps, alpha=0.75, gamma=2.0)
            cls_loss = cls_loss * weights.view(1, 3, 1, 1)
            cls_loss = cls_loss.mean()
            det_loss = weighted_mse_loss(det, det_targets, weight_positive=1000.0)
            loss = cls_loss + 0.05 * det_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, heatmaps, det_targets, _ in val_loader:
                imgs, heatmaps, det_targets = imgs.to(device), heatmaps.to(device), det_targets.to(device)
                det, cls = model(imgs)
                cls_loss = focal_loss(cls, heatmaps, alpha=0.75, gamma=2.0)
                cls_loss = cls_loss * weights.view(1, 3, 1, 1)
                cls_loss = cls_loss.mean()
                det_loss = weighted_mse_loss(det, det_targets, weight_positive=1000.0)
                loss = cls_loss + 0.05 * det_loss
                val_loss += loss.item() * imgs.size(0)
        
        # Monitor keypoints
        monitor_keypoints = {}
        with torch.no_grad():
            for img_name, (img, _, _) in monitor_data.items():
                det, cls = model(img)
                pred_keypoints = extract_keypoints_from_heatmap(cls.cpu().numpy(), det.cpu().numpy())
                monitor_keypoints[img_name] = pred_keypoints[:, :3].tolist()
        
        train_loss = train_loss / len(train_loader.dataset)
        val_loss = val_loss / len(val_loader.dataset)
        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.8f}, Val Loss: {val_loss:.8f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        for img_name, keypoints in monitor_keypoints.items():
            if keypoints:
                print(f"{img_name} Keypoints: {[[round(x, 3), round(y, 3), round(s, 4)] for x, y, s in keypoints]}")
            else:
                print(f"{img_name}: No keypoints detected")
        
        # Save checkpoints
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f'epoch_{epoch+1}.pt')
            torch.jit.script(model).save(checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(CHECKPOINT_DIR, 'best_model.pt')
            torch.jit.script(model).save(best_model_path)
            print(f"Saved best model to {best_model_path} with Val Loss: {val_loss:.8f}")
        
        if epoch == 48:
            last_model_path = os.path.join(CHECKPOINT_DIR, 'last_model_before_50.pt')
            torch.jit.script(model).save(last_model_path)
            print(f"Saved last model before epoch 50 to {last_model_path}")
        
        scheduler.step(val_loss)
    
    # Save final model
    final_model_path = '/content/retro_target_tracker/cpp_superpoint_retrained.pt'
    torch.jit.script(model).save(final_model_path)
    print(f"Saved final model to {final_model_path}")

if __name__ == "__main__":
    try:
        train_model()
    except Exception as e:
        print(f"Training failed: {e}")