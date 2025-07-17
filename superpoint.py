import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os

class SuperPointDataset(Dataset):
    def __init__(self, dataset_dir, split):
        self.dataset_dir = dataset_dir
        self.split = split
        self.image_dir = os.path.join(dataset_dir, f"images/{split}")
        self.annotation_dir = os.path.join(dataset_dir, f"annotations/{split}")
        self.image_files = [f for f in os.listdir(self.image_dir) if f.endswith('.jpg')]
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = torch.from_numpy(img).float() / 255.0
        img = img.unsqueeze(0)  # [1, H, W]
        
        heatmap_path = os.path.join(self.annotation_dir, os.path.splitext(img_name)[0] + '.npy')
        heatmap = np.load(heatmap_path)
        heatmap = torch.from_numpy(heatmap).float()
        
        return {'image': img, 'heatmap': heatmap}

class SuperPoint(nn.Module):
    def __init__(self):
        super(SuperPoint, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
        )
        self.detector = nn.Conv2d(256, 65, 1)  # Heatmap output
        self.descriptor = nn.Conv2d(256, 256, 1)  # Descriptor output
    
    def forward(self, x):
        x = self.conv(x)
        det = self.detector(x)  # [batch, 65, 30, 40]
        desc = self.descriptor(x)  # [batch, 256, 30, 40]
        det = torch.softmax(det, dim=1)  # Softmax over channel dimension
        return det, desc

# Training loop
dataset = SuperPointDataset('superpoint_dataset', 'train')
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
model = SuperPoint().cuda()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

for epoch in range(50):
    total_loss = 0
    for batch in dataloader:
        images = batch['image'].cuda()
        heatmaps = batch['heatmap'].cuda()
        optimizer.zero_grad()
        det, _ = model(images)
        loss = criterion(det, heatmaps)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader)}")

# Save model
torch.save(model.state_dict(), 'superpoint_retrained.pt')

# Convert to TorchScript
model.eval()
traced_model = torch.jit.trace(model, torch.randn(1, 1, 480, 640).cuda())
traced_model.save('/home/mehdi/test_concrete_4/MarkerPose/dataset/cpp_superpoint_retrained.pt')