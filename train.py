import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
import yaml
import os
from datetime import datetime
from models import HeatmapTracker
from utils.data_loader import TargetDataset, Augmentation
from utils.visualization import plot_heatmap_comparison

# Configuration
with open('configs/train_config.yaml') as f:
    config = yaml.safe_load(f)

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs(config['training']['checkpoint_dir'], exist_ok=True)
writer = SummaryWriter(f"logs/{timestamp}")

def heatmap_loss(pred, target):
    """Weighted MSE loss focusing on target center"""
    weights = 1 + 9 * target  # Emphasize peak accuracy
    return (weights * (pred - target)**2).mean()

def validate(model, dataloader, device):
    """Run validation loop"""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for images, heatmaps in dataloader:
            images, heatmaps = images.to(device), heatmaps.to(device)
            pred_heatmaps, _ = model(images)
            loss = heatmap_loss(pred_heatmaps, heatmaps)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

def train():
    # Initialize
    model = HeatmapTracker().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['model']['learning_rate'])
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, verbose=True)
    
    # Data
    train_dataset = TargetDataset(
        config['data']['train_path'],
        transform=Augmentation()
    )
    val_dataset = TargetDataset(config['data']['val_path'])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        num_workers=2
    )

    # Training loop
    best_loss = float('inf')
    for epoch in range(config['training']['epochs']):
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, (images, heatmaps) in enumerate(train_loader):
            images, heatmaps = images.to(device), heatmaps.to(device)
            
            optimizer.zero_grad()
            pred_heatmaps, pred_coords = model(images)
            loss = heatmap_loss(pred_heatmaps, heatmaps)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Log batch progress
            if batch_idx % 20 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")
                
                # Visualize sample prediction
                if batch_idx == 0:
                    sample_img = images[0].cpu().permute(1,2,0).numpy()
                    sample_heatmap = heatmaps[0].cpu().numpy()
                    pred_heatmap = pred_heatmaps[0].detach().cpu().numpy()
                    plot_heatmap_comparison(sample_img, pred_heatmap, sample_heatmap)

        # Validation
        avg_epoch_loss = epoch_loss / len(train_loader)
        val_loss = validate(model, val_loader, device)
        
        # Logging
        writer.add_scalar('Loss/train', avg_epoch_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        print(f"Epoch {epoch} Summary:")
        print(f"Train Loss: {avg_epoch_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Save checkpoints
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), 
                      f"{config['training']['checkpoint_dir']}/best_model.pth")
            print("Saved new best model")
            
        if epoch % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'loss': val_loss,
            }, f"{config['training']['checkpoint_dir']}/checkpoint_epoch{epoch}.pth")
        
        scheduler.step(val_loss)

if __name__ == "__main__":
    train()
    writer.close()
