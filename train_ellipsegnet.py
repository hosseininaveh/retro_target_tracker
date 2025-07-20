import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from tqdm import tqdm

class EllipSegNetDataset(Dataset):
    def __init__(self, dataset_dir, split):
        self.patch_dir = os.path.join(dataset_dir, f'ellipsegnet_patches/{split}')
        self.mask_dir = os.path.join(dataset_dir, f'ellipsegnet_masks/{split}')
        self.patch_files = sorted([f for f in os.listdir(self.patch_dir) if f.endswith('.npy')])
        if not self.patch_files:
            raise ValueError(f"No patch files found in {self.patch_dir}")
        print(f"Found {len(self.patch_files)} patches in {self.patch_dir}")

    def __len__(self):
        return len(self.patch_files)

    def __getitem__(self, idx):
        patch_name = self.patch_files[idx]
        patch_path = os.path.join(self.patch_dir, patch_name)
        mask_path = os.path.join(self.mask_dir, patch_name.replace('patch_', 'mask_'))
        
        try:
            patch = np.load(patch_path, allow_pickle=False).astype(np.float32)
            mask = np.load(mask_path, allow_pickle=False).astype(np.float32)
            if patch.shape != (120, 120) or mask.shape != (120, 120):
                raise ValueError(f"Invalid shapes: patch={patch.shape}, mask={mask.shape} for {patch_name}")
            patch = torch.from_numpy(patch).unsqueeze(0)  # [1, 120, 120]
            mask = torch.from_numpy(mask).unsqueeze(0)   # [1, 120, 120]
            return {'patch': patch, 'mask': mask, 'patch_name': patch_name}
        except Exception as e:
            print(f"Error loading {patch_name}: {e}")
            raise

class EllipSegNet(nn.Module):
    def __init__(self, in_channels=1, n_channels=16):
        super(EllipSegNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, n_channels, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(n_channels, n_channels*2, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(n_channels*2, n_channels*4, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(n_channels*4, n_channels*8, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )
        self.upconv4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(n_channels*8, n_channels*4, 3, stride=1, padding=1),
            nn.ReLU()
        )
        self.upconv3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(n_channels*4, n_channels*2, 3, stride=1, padding=1),
            nn.ReLU()
        )
        self.upconv2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(n_channels*2, n_channels, 3, stride=1, padding=1),
            nn.ReLU()
        )
        self.upconv1 = nn.Conv2d(n_channels, 1, 3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.upconv4(x)
        x = self.upconv3(x)
        x = self.upconv2(x)
        x = self.upconv1(x)
        return x

# Training setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load datasets
train_dataset = EllipSegNetDataset('/content/retro_target_tracker/dataset/ellipsegnet_dataset', '

train')
val_dataset = EllipSegNetDataset('/content/retro_target_tracker/dataset/ellipsegnet_dataset', 'val')
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)

model = EllipSegNet(in_channels=1, n_channels=16).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()

# Training loop
num_epochs = 50
best_val_loss = float('inf')
for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = 0
    for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Train)")):
        patches = batch['patch'].to(device)  # [batch, 1, 120, 120]
        masks = batch['mask'].to(device)    # [batch, 1, 120, 120]
        
        optimizer.zero_grad()
        outputs = model(patches)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch+1}, Train Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
    
    avg_train_loss = train_loss / len(train_loader)
    print(f"Epoch {epoch+1}, Average Train Loss: {avg_train_loss:.4f}")
    
    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Val)")):
            patches = batch['patch'].to(device)
            masks = batch['mask'].to(device)
            outputs = model(patches)
            loss = criterion(outputs, masks)
            val_loss += loss.item()
    
    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch {epoch+1}, Average Val Loss: {avg_val_loss:.4f}")
    
    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), '/content/retro_target_tracker/dataset/ellipsegnet_best.pt')
        print(f"Saved best model with Val Loss: {best_val_loss:.4f}")

# Save final state dict
torch.save(model.state_dict(), '/content/retro_target_tracker/dataset/ellipsegnet.pt')

# Convert to TorchScript
model.eval()
example_input = torch.randn(1, 1, 120, 120).to(device)
traced_model = torch.jit.trace(model, example_input)
traced_model.save('/content/retro_target_tracker/dataset/cpp_ellipsegnet.pt')
print("Saved TorchScript model to /content/retro_target_tracker/dataset/cpp_ellipsegnet.pt")