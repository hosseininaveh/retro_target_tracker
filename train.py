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
from torchvision.transforms import Normalize

inv_normalize = Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)
# Suppress unnecessary warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging
def find_subpixel_center(heatmap):
    """Find subpixel-accurate center coordinates from heatmap"""
    # Get pixel-level max location
    y_max, x_max = np.unravel_index(heatmap.argmax(), heatmap.shape)
    
    # Quadratic interpolation for subpixel accuracy
    if 0 < x_max < heatmap.shape[1]-1 and 0 < y_max < heatmap.shape[0]-1:
        dx = 0.5 * (heatmap[y_max, x_max+1] - heatmap[y_max, x_max-1])
        dy = 0.5 * (heatmap[y_max+1, x_max] - heatmap[y_max-1, x_max])
        x_max += dx
        y_max += dy
    
    return torch.tensor([x_max, y_max], device=heatmap.device)
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

def generate_heatmaps(points, img_size, sigma=8.0):
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
    #print("Sample heatmap values:")
    #print(heatmaps[0,0].max().item())  # Should be ~1.0 at target location
    #print(heatmaps[0,0].mean().item()) # Should be very low (~0.001)
    #plt.imshow(heatmaps[0,0].cpu().numpy())
    #plt.title("Generated Heatmap")
    #plt.show()
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
    """Compute evaluation metrics with robust error handling"""
    # Ensure we're working with float tensors
    pred = pred.float()
    target = target.float()
    
    # Use lower threshold for binary conversion
    pred_bin = (pred > 0.1).float()
    target_bin = (target > 0.1).float()
    
    # Only calculate for heatmaps with sufficient activation
    min_pixels = 5
    valid = target_bin.sum(dim=(-2,-1)) > min_pixels
    
    if valid.any():
        # Calculate IoU
        intersection = (pred_bin[valid] * target_bin[valid]).sum(dim=(-2,-1))
        union = (pred_bin[valid] + target_bin[valid]).clamp(0,1).sum(dim=(-2,-1))
        iou = (intersection / (union + 1e-6)).mean().item()
        
        # Calculate center error
        pred_centers = []
        target_centers = []
        for i in range(valid.sum()):
            for c in range(pred.shape[1]):  # For each channel
                if target_bin[valid][i,c].sum() > 0:
                    pred_centers.append(find_subpixel_center(pred[valid][i,c]))
                    target_centers.append(find_subpixel_center(target[valid][i,c]))
        
        if pred_centers:
            errors = [torch.norm(p-t) for p,t in zip(pred_centers, target_centers)]
            avg_error = sum(errors)/len(errors)
        else:
            avg_error = 0.0
    else:
        iou = 0.0
        avg_error = 0.0
    
    return iou, avg_error

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
                sigma=8.0
            ).to(device)
            
            # Forward pass
            pred_heatmaps = model(images)
            if num_batches == 0 and epoch % 2 == 0:
                idx = 0  # First sample
                img = images[idx].cpu().permute(1,2,0).numpy()
                pred_hm = pred_heatmaps[idx,0].cpu().numpy()
                true_hm = heatmaps[idx,0].cpu().numpy()
                
                #plt.figure(figsize=(15,5))
                #plt.subplot(131); plt.imshow(img); plt.title("Image")
                #plt.subplot(132); plt.imshow(pred_hm); plt.title("Predicted Heatmap")
                #plt.subplot(133); plt.imshow(true_hm); plt.title("True Heatmap")
                #plt.show()
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
        if epoch % 1 == 0:  # Every epoch
            idx = 0  # First sample in batch
            img = inv_normalize(images[idx]).cpu().permute(1,2,0).numpy()
            pred_hm = pred_heatmaps[idx,0].cpu().detach().numpy()
            true_hm = heatmaps[idx,0].cpu().numpy()
    
            plt.figure(figsize=(15,5))
            plt.subplot(131); plt.imshow(img); plt.title("Image")
            plt.subplot(132); plt.imshow(pred_hm); plt.title("Predicted Heatmap")
            plt.subplot(133); plt.imshow(true_hm); plt.title("True Heatmap")
            plt.show()
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