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
from utils.visualization import plot_heatmap_comparison
from utils.data_loader import TargetDataset, Augmentation
import matplotlib.pyplot as plt

# Suppress unnecessary warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging

def load_config(config_path):
    """Load and validate configuration"""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    required_keys = {
        'training': ['checkpoint_dir', 'epochs', 'early_stop_patience'],
        'data': ['train_path', 'val_path', 'batch_size', 'input_size'],
        'model': ['learning_rate', 'weight_decay', 'heatmap_sigma']
    }
    
    for section, keys in required_keys.items():
        if section not in config:
            raise ValueError(f"Missing configuration section: {section}")
        for key in keys:
            if key not in config[section]:
                raise ValueError(f"Missing configuration key: {section}.{key}")
    return config

def generate_heatmaps(points, img_size, sigma=5.0):
    """Generate heatmaps from point coordinates for both targets"""
    height, width = img_size
    batch_size = points.shape[0]
    heatmaps = torch.zeros(batch_size, 2, height, width)  # CHANNELS FIRST format
    
    for i in range(batch_size):
        for j in range(2):  # For each target point
            x, y = points[i, j]
            if x >= 0 and y >= 0:  # Only create heatmap if point is valid
                # Create meshgrid (note the order: height first, then width)
                y_grid, x_grid = torch.meshgrid(
                    torch.arange(height, dtype=torch.float32, device=points.device),
                    torch.arange(width, dtype=torch.float32, device=points.device),
                    indexing='ij'
                )
                
                # Calculate squared distances
                dist_sq = (x_grid - x)**2 + (y_grid - y)**2
                
                # Create Gaussian heatmap
                heatmap = torch.exp(-dist_sq / (2 * sigma**2))
                heatmaps[i, j] = heatmap
                
    return heatmaps

def heatmap_loss(pred, target):
    """Focal loss for heatmap prediction"""
    # Handle size mismatches
    if pred.size()[-2:] != target.size()[-2:]:
        target = F.interpolate(target.float(), size=pred.size()[-2:], mode='bilinear')
    
    # Focal loss parameters
    alpha = 2.0
    beta = 4.0
    total_loss = 0.0
    
    for i in range(pred.size(1)):  # Loop over target channels
        pred_ch = pred[:, i:i+1]
        target_ch = target[:, i:i+1]
        
        pos_mask = (target_ch > 0.5).float()
        neg_mask = (target_ch <= 0.5).float()
        
        pos_loss = -pos_mask * (1 - pred_ch)**alpha * torch.log(pred_ch + 1e-12)
        neg_loss = -neg_mask * (1 - target_ch)**beta * (pred_ch)**alpha * torch.log(1 - pred_ch + 1e-12)
        
        total_loss += (pos_loss + neg_loss).mean()
    
    return total_loss / pred.size(1)  # Average over channels

def calculate_metrics(pred, target):
    """Compute evaluation metrics for both points"""
    # Ensure correct dimensions
    if len(pred.shape) == 4:
        pred = pred.squeeze(1) if pred.size(1) == 1 else pred
    if len(target.shape) == 4:
        target = target.squeeze(1) if target.size(1) == 1 else target
    
    # Initialize metrics
    total_iou = 0.0
    total_error = 0.0
    num_points = 0
    
    for i in range(pred.size(1)):  # For each point channel
        pred_ch = pred[:, i]
        target_ch = target[:, i]
        
        # Binarize
        pred_bin = (pred_ch > 0.3).float()
        target_bin = (target_ch > 0.3).float()

        # Add area check
        min_area = 10  # Minimum expected target area
        valid = (target_bin.sum(dim=(-2,-1)) > min_area)
        
        if valid.any():
             intersection = (pred_bin[valid] * target_bin[valid]).sum(dim=(-2,-1))
             union = ((pred_bin[valid] + target_bin[valid]) > 0).float().sum(dim=(-2,-1))
             iou = (intersection / (union + 1e-6)).mean().item()
        else:
             iou = 0.0

        total_iou += iou
        
        # Center error calculation
        errors = []
        for j in range(pred_ch.shape[0]):
            pred_center = np.unravel_index(pred_ch[j].argmax().item(), pred_ch.shape[-2:])
            target_center = np.unravel_index(target_ch[j].argmax().item(), target_ch.shape[-2:])
            error = np.sqrt((pred_center[0]-target_center[0])**2 + 
                          (pred_center[1]-target_center[1])**2)
            errors.append(error)
        
        total_error += np.mean(errors) if errors else 0
        num_points += 1 if errors else 0
    
    avg_iou = total_iou / num_points if num_points > 0 else 0
    avg_error = total_error / num_points if num_points > 0 else 0
    
    return avg_iou, avg_error

def validate(model, dataloader, device, writer=None, epoch=None):
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    total_error = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for images, points in dataloader:
            # Convert to NCHW format and proper dtype
            images = images.permute(0, 3, 1, 2).float().to(device)
            
            # Generate heatmaps (will be [B, 2, H, W])
            heatmaps = generate_heatmaps(
                points, 
                (images.size(-2), images.size(-1)),
                sigma=3.0
            ).to(device)
            
            # Forward pass
            pred_heatmaps = model(images)
            
            # Ensure shapes match
            if pred_heatmaps.shape[-2:] != heatmaps.shape[-2:]:
                heatmaps = F.interpolate(heatmaps, size=pred_heatmaps.shape[-2:], mode='bilinear')
            
            # Loss calculation
            loss = heatmap_loss(pred_heatmaps, heatmaps)
            total_loss += loss.item()
            
            # Calculate metrics
            iou, error = calculate_metrics(pred_heatmaps, heatmaps)
            total_iou += iou
            total_error += error
            num_batches += 1
    
    return total_loss/num_batches, total_iou/num_batches, total_error/num_batches

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
    model = HeatmapTracker(max_targets=2).to(device)  # Modified for 2 targets
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
    
    # Data verification
    sample = next(iter(train_loader))
    img, points = sample
    print("\nData Verification:")
    print(f"Image shape: {img.shape}")
    print(f"Points shape: {points.shape}")
    print(f"Sample point coordinates: {points[0]}")
    
    # Training loop
    best_loss = float('inf')
    patience = config['training']['early_stop_patience']
    no_improve = 0
    
    for epoch in range(config['training']['epochs']):
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, (images, points) in enumerate(train_loader):
            # Ensure proper tensor format (NCHW)
            #images = images.permute(0, 3, 1, 2).to(device)  # Convert from NHWC to NCHW
            if images.dtype != torch.float32:
                images = images.float()
            if images.max() > 1.0:
                images = images / 255.0
        
            images = images.to(device)
            # Generate heatmaps from points
            heatmaps = generate_heatmaps(
                points,
                (images.size(-2), images.size(-1)),
                sigma=config['model']['heatmap_sigma']
            ).to(device)
            
            optimizer.zero_grad()
            pred_heatmaps = model(images)
            
            # Debug shapes and ranges
            if batch_idx == 0 and epoch == 0:
                print(f"\nInitial Batch Debug:")
                print(f"Input shape: {images.shape}")
                print(f"Heatmap shape: {heatmaps.shape}")
                print(f"Prediction shape: {pred_heatmaps.shape}")
            
            loss = heatmap_loss(pred_heatmaps, heatmaps)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            epoch_loss += loss.item()
            
            # Log batch progress
            if batch_idx % 20 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.6f}")
        
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