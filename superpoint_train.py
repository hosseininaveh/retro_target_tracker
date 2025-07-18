import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os

# Check CUDA availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class SuperPointDataset(Dataset):
    def __init__(self, dataset_dir, split):
        print(f"Initializing dataset for split: {split}")
        self.dataset_dir = dataset_dir
        self.split = split
        self.image_dir = os.path.join(dataset_dir, f"images/{split}")
        self.annotation_dir = os.path.join(dataset_dir, f"annotations/{split}")
        
        if not os.path.exists(self.image_dir):
            raise FileNotFoundError(f"Image directory {self.image_dir} does not exist")
        if not os.path.exists(self.annotation_dir):
            raise FileNotFoundError(f"Annotation directory {self.annotation_dir} does not exist")
        
        self.image_files = [f for f in os.listdir(self.image_dir) if f.endswith('.jpg')]
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
            img = cv2.resize(img, (640, 480))  # Match new dataset input size
            img = torch.from_numpy(img).float() / 255.0
            img = img.unsqueeze(0)  # [1, 480, 640]
            
            heatmap = np.load(heatmap_path, allow_pickle=False).astype(np.float32)
            if heatmap.shape != (65, 60, 80):
                raise ValueError(f"Invalid heatmap shape {heatmap.shape} for {heatmap_path}, expected (65, 60, 80)")
            heatmap = torch.from_numpy(heatmap).float()  # [65, 60, 80]
            
            return {'image': img, 'heatmap': heatmap}
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
                if heatmap.shape != (65, 60, 80):
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
    def __init__(self, Nc=2):  # Nc=2 for two points (point0, point1) + background
        super(SuperPointNet, self).__init__()
        self.pool = nn.MaxPool2d(2, stride=2)
        self.conv1 = DoubleConv(1, 64)
        self.conv2 = DoubleConv(64, 64)
        self.conv3 = DoubleConv(64, 128)
        self.conv4 = DoubleConv(128, 128)
        self.convD = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 65, 1, stride=1, padding=0)
        )
        self.convC = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, Nc+1, 1, stride=1, padding=0)
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
        return det, cls

# Training loop
try:
    dataset = SuperPointDataset('/home/mehdi/test_concrete_4/MarkerPose/dataset/superpoint_dataset', 'train')
    dataset.verify_files()  # Verify dataset integrity
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    print(f"Dataset size: {len(dataset)} samples")
    
    model = SuperPointNet(Nc=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    for epoch in range(50):
        total_loss = 0
        for batch_idx, batch in enumerate(dataloader):
            images = batch['image'].to(device)
            heatmaps = batch['heatmap'].to(device)
            print(f"Processing batch {batch_idx+1}/{len(dataloader)}: image shape={images.shape}, heatmap shape={heatmaps.shape}")
            
            optimizer.zero_grad()
            det, cls = model(images)
            loss = criterion(det, heatmaps)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            print(f"Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {loss.item()}")
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}, Average Loss: {avg_loss}")
    
    # Save model
    torch.save(model.state_dict(), '/home/mehdi/test_concrete_4/MarkerPose/dataset/superpoint_retrained.pt')
    print("Saved model state dictionary to superpoint_retrained.pt")
    
    # Convert to TorchScript
    model.eval()
    traced_model = torch.jit.trace(model, torch.randn(1, 1, 480, 640).to(device))
    traced_model.save('/home/mehdi/test_concrete_4/MarkerPose/dataset/cpp_superpoint_retrained.pt')
    print("Saved TorchScript model to cpp_superpoint_retrained.pt")

except Exception as e:
    print(f"Training failed: {e}")