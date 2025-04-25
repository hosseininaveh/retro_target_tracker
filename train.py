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

def load_config(config_path):
    """Load and validate configuration"""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    # Validate required keys
    required_keys = {
        'training': ['checkpoint_dir', 'epochs'],
        'data': ['train_path', 'val_path', 'batch_size'],
        'model': ['learning_rate']
    }
    for section, keys in required_keys.items():
        if section not in config:
            raise ValueError(f"Missing configuration section: {section}")
        for key in keys:
            if key not in config[section]:
                raise ValueError(f"Missing configuration key: {section}.{key}")
    return config

def heatmap_loss(pred, target):
    """Weighted MSE loss focusing on target center"""
    weights = 1 + 9 * target  # Emphasize peak accuracy
    return (weights * (pred - target)**2).mean()

def validate(model, dataloader, device, writer=None, epoch=None):
    """Run validation loop with optional TensorBoard logging"""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for images, heatmaps in dataloader:
            images, heatmaps = images.to(device), heatmaps.to(device)
            pred_heatmaps, _ = model(images)
            loss = heatmap_loss(pred_heatmaps, heatmaps)
            total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    if writer is not None and epoch is not None:
        writer.add_scalar('Loss/val', avg_loss, epoch)
    return avg_loss

def train():
    # Load configuration
    config = load_config('configs/train_config.yaml')
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = config['training']['checkpoint_dir']
    os.makedirs(checkpoint_dir, exist_ok=True)
    writer = SummaryWriter(f"logs/{timestamp}")
    
    # Initialize model and optimizer
    model = HeatmapTracker().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['model']['learning_rate'],
        weight_decay=1e-4  # Add weight decay for regularization
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=3,
        verbose=True,
        min_lr=1e-6  # Add minimum learning rate
    )
    
    # Create datasets and data loaders
    train_dataset = TargetDataset(
        config['data']['train_path'],
        transform=Augmentation()
    )
    val_dataset = TargetDataset(config['data']['val_path'])
    
    # Use dynamic worker count based on batch size
    num_workers = min(4, config['data']['batch_size'] * 2)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True  # Ensure complete batches
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        num_workers=num_workers // 2,
        pin_memory=True
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
        val_loss = validate(model, val_loader, device, writer, epoch)
        
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
                      f"{checkpoint_dir}/best_model.pth")
            print("Saved new best model")
        
        if epoch % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'loss': val_loss,
            }, f"{checkpoint_dir}/checkpoint_epoch{epoch}.pth")
        
        scheduler.step(val_loss)
    
    writer.close()

if __name__ == "__main__":
    train()