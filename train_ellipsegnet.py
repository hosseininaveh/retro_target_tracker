import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os

class EllipSegNetDataset(Dataset):
    def __init__(self, dataset_dir, split):
        self.dataset_dir = dataset_dir
        self.split = split
        self.image_dir = os.path.join(dataset_dir, f"images/{split}")
        self.annotation_dir = os.path.join(dataset_dir, f"annotations/{split}")
        self.image_files = [f for f in os.listdir(self.image_dir) if f.endswith('.jpg')]
        print(f"Found {len(self.image_files)} images in {self.image_dir}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.annotation_dir, os.path.splitext(img_name)[0] + '.npy')
        
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = torch.from_numpy(img).float() / 255.0
        img = img.unsqueeze(0)  # [1, 120, 120]
        
        mask = np.load(mask_path, allow_pickle=False).astype(np.float32)
        mask = torch.from_numpy(mask).float().unsqueeze(0)  # [1, 120, 120]
        
        return {'image': img, 'mask': mask}

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

class EllipSegNet(nn.Module):
    def __init__(self, init_f=64, num_outputs=1):
        super(EllipSegNet, self).__init__()
        self.pool = nn.MaxPool2d(2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.inc = DoubleConv(1, init_f)
        self.down1 = DoubleConv(init_f, 2*init_f)
        self.down2 = DoubleConv(2*init_f, 4*init_f)
        self.down3 = DoubleConv(4*init_f, 4*init_f)
        self.up1 = DoubleConv(2*4*init_f, 2*init_f, 4*init_f)
        self.up2 = DoubleConv(2*2*init_f, init_f, 2*init_f)
        self.up3 = DoubleConv(2*init_f, init_f)
        self.outc = nn.Conv2d(init_f, num_outputs, 1)
    
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(self.pool(x1))
        x3 = self.down2(self.pool(x2))
        x4 = self.down3(self.pool(x3))
        x = torch.cat([self.upsample(x4), x3], dim=1)
        x = self.up1(x)
        x = torch.cat([self.upsample(x), x2], dim=1)
        x = self.up2(x)
        x = torch.cat([self.upsample(x), x1], dim=1)
        x = self.up3(x)
        x = self.outc(x)
        return torch.sigmoid(x)

# Training loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = EllipSegNetDataset('/home/mehdi/test_concrete_4/MarkerPose/dataset/ellipsegnet_dataset', 'train')
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
model = EllipSegNet(init_f=64, num_outputs=1).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

for epoch in range(50):
    total_loss = 0
    for batch_idx, batch in enumerate(dataloader):
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        print(f"Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {loss.item()}")
    print(f"Epoch {epoch+1}, Average Loss: {total_loss / len(dataloader)}")

# Save model
torch.save(model.state_dict(), '/home/mehdi/test_concrete_4/MarkerPose/dataset/ellipsegnet.pt')
model.eval()
traced_model = torch.jit.trace(model, torch.randn(1, 1, 120, 120).to(device))
traced_model.save('/home/mehdi/test_concrete_4/MarkerPose/dataset/cpp_ellipsegnet.pt')