import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
import yaml
import os
import numpy as np
from datetime import datetime
from models import HeatmapTracker
from utils.data_loader import TargetDataset, Augmentation
from utils.visualization import plot_heatmap_comparison

def load_config(config_path):
    """Load and validate configuration"""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    required_keys = {
        'training': ['checkpoint_dir', 'epochs', 'early_stop_patience'],
        'data': ['train_path', 'val_path', 'batch_size'],
        'model': ['learning_rate', 'weight_decay']
    }
    
    for section, keys in required_keys.items():
        if section not in config:
            raise ValueError(f"Missing configuration section: {section}")
        for key in keys:
            if key not in config[section]:
                raise ValueError(f"Missing configuration key: {section}.{key}")
    return config

def heatmap_loss(pred, target):
    """Improved focal loss for heatmap prediction"""
    # Shape handling
    if len(pred.shape) == 4:
        pred = pred.squeeze(1)
    if len(target.shape) == 3:
        target = target.unsqueeze(1)
    
    # Focal loss parameters
    alpha = 2.0
    beta = 4.0
    pos_mask = (target > 0.1).float()
    neg_mask = (target <= 0.1).float()
    
    # Focal loss calculation
    pos_loss = -pos_mask * (1 - pred)**alpha * torch.log(pred + 1e-12)
    neg_loss = -neg_mask * (1 - target)**beta * (pred)**alpha * torch.log(1 - pred + 1e-12)
    
    return (pos_loss + neg_loss).mean()

def calculate_metrics(pred, target):
    """Compute evaluation metrics with proper dimension handling"""
    # Ensure correct dimensions
    if len(pred.shape) == 4:
        pred = pred.squeeze(1)  # Remove channel dim if present
    if len(target.shape) == 4:
        target = target.squeeze(1)
    
    # Binarize
    pred_bin = (pred > 0.5).float()
    target_bin = (target > 0.5).float()
    
    # IoU calculation
    intersection = (pred_bin * target_bin).sum(dim=(-2, -1))  # Sum over last two dims (H,W)
    union = ((pred_bin + target_bin) > 0).float().sum(dim=(-2, -1))
    iou = (intersection / (union + 1e-6)).mean().item()
    
    # Center error calculation
    errors = []
    for i in range(pred.shape[0]):
        pred_center = np.unravel_index(pred[i].argmax().item(), pred.shape[-2:])
        target_center = np.unravel_index(target[i].argmax().item(), target.shape[-2:])
        error = np.sqrt((pred_center[0]-target_center[0])**2 + 
                      (pred_center[1]-target_center[1])**2)
        errors.append(error)
    avg_error = np.mean(errors)
    
    return iou, avg_error


def validate(model, dataloader, device, writer=None, epoch=None):
    """Enhanced validation with robust dimension handling"""
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    total_error = 0.0
    
    with torch.no_grad():
        for images, heatmaps in dataloader:
            images = images.to(device)
            heatmaps = heatmaps.to(device).float()
            
            # Forward pass
            pred_heatmaps = model(images)
            
            # Loss calculation
            loss = heatmap_loss(pred_heatmaps, heatmaps)
            total_loss += loss.item()
            
            # Calculate metrics with proper shape handling
            pred = pred_heatmaps.squeeze(1) if len(pred_heatmaps.shape) == 4 else pred_heatmaps
            target = heatmaps.squeeze(1) if len(heatmaps.shape) == 4 else heatmaps
            
            iou, error = calculate_metrics(pred, target)
            total_iou += iou
            total_error += error
    
    # Compute averages
    avg_loss = total_loss / len(dataloader)
    avg_iou = total_iou / len(dataloader)
    avg_error = total_error / len(dataloader)
    
    # Logging
    if writer is not None and epoch is not None:
        writer.add_scalar('Loss/val', avg_loss, epoch)
        writer.add_scalar('Metrics/IoU', avg_iou, epoch)
        writer.add_scalar('Metrics/Center_Error', avg_error, epoch)
    
    return avg_loss, avg_iou, avg_error
    
def train():
    # Load configuration
    config = load_config('configs/train_config.yaml')
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = config['training']['checkpoint_dir']
    os.makedirs(checkpoint_dir, exist_ok=True)
    writer = SummaryWriter(f"logs/{timestamp}")
    
    # Initialize model
    model = HeatmapTracker().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['model']['learning_rate'],
        weight_decay=config['model']['weight_decay']
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=3,
        factor=0.5,
        verbose=True
    )
    
    # Create datasets
    train_dataset = TargetDataset(
        config['data']['train_path'],
        transform=Augmentation()
    )
    val_dataset = TargetDataset(config['data']['val_path'])
    
    # Data loaders
    num_workers = min(4, os.cpu_count())
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Training loop
    best_loss = float('inf')
    patience = config['training']['early_stop_patience']
    no_improve = 0
    
    for epoch in range(config['training']['epochs']):
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, (images, heatmaps) in enumerate(train_loader):
            images, heatmaps = images.to(device), heatmaps.to(device).float()
            
            optimizer.zero_grad()
            pred_heatmaps = model(images)
            
            # Debug shapes and ranges
            if batch_idx == 0 and epoch == 0:
                print(f"\nInitial Batch Debug:")
                print(f"Input shape: {images.shape}")
                print(f"Target range: {heatmaps.min().item():.3f}-{heatmaps.max().item():.3f}")
                print(f"Pred range: {pred_heatmaps.min().item():.3f}-{pred_heatmaps.max().item():.3f}")
                
                # Visualize first sample
                with torch.no_grad():
                    sample_img = images[0].cpu().permute(1, 2, 0).numpy()
                    sample_heatmap = heatmaps[0].cpu().numpy()
                    pred_heatmap = pred_heatmaps[0].cpu().numpy()
                    plot_heatmap_comparison(sample_img, pred_heatmap, sample_heatmap)
            
            loss = heatmap_loss(pred_heatmaps, heatmaps)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            epoch_loss += loss.item()
            
            # Log batch progress
            if batch_idx % 20 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.6f}")
                
                # Visualize sample prediction periodically
                if batch_idx == 0:
                    with torch.no_grad():
                        sample_img = images[0].cpu().permute(1, 2, 0).numpy()
                        sample_heatmap = heatmaps[0].cpu().numpy()
                        pred_heatmap = pred_heatmaps[0].cpu().numpy()
                        plot_heatmap_comparison(sample_img, pred_heatmap, sample_heatmap)
        
        # Validation
        val_loss, val_iou, val_error = validate(model, val_loader, device, writer, epoch)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Logging
        writer.add_scalar('Loss/train', epoch_loss/len(train_loader), epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        print(f"\nEpoch {epoch} Summary:")
        print(f"Train Loss: {epoch_loss/len(train_loader):.6f} | Val Loss: {val_loss:.6f}")
        print(f"Val IoU: {val_iou:.4f} | Center Error: {val_error:.2f} px")
        
        # Checkpointing and early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            no_improve = 0
            torch.save(model.state_dict(), f"{checkpoint_dir}/best_model.pth")
            print("Saved new best model")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch}")
                break
        
        # Save periodic checkpoints
        if epoch % 5 == 0 or epoch == config['training']['epochs'] - 1:
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'loss': val_loss,
            }, f"{checkpoint_dir}/checkpoint_epoch{epoch}.pth")
    
    writer.close()
    print("\nTraining completed!")

if __name__ == "__main__":
    train()