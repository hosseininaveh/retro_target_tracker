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
    print("\nDEBUGGING METRICS CALCULATION")  # Debug line
    
    # Ensure we're working with float tensors
    pred = pred.float()
    target = target.float()
    
    # Use lower threshold for binary conversion
    pred_bin = (pred > 0.1).float()
    target_bin = (target > 0.1).float()
    
    # Only calculate for heatmaps with sufficient activation
    min_pixels = 5  # Try reducing this if needed
    valid = target_bin.sum(dim=(-2,-1)) > min_pixels
    
    print(f"Number of valid targets: {valid.sum().item()}/{target.shape[0]}")  # Debug
    
    if valid.any():
        # Calculate IoU
        intersection = (pred_bin[valid] * target_bin[valid]).sum(dim=(-2,-1))
        union = (pred_bin[valid] + target_bin[valid]).clamp(0,1).sum(dim=(-2,-1))
        iou = (intersection / (union + 1e-6)).mean().item()
        
        # Calculate center error - NEW ROBUST VERSION
        errors = []
        for i in range(pred.shape[0]):  # For each sample
            for c in range(pred.shape[1]):  # For each channel
                if target_bin[i,c].sum() > 0:  # If target exists
                    pred_center = find_subpixel_center(pred[i,c])
                    target_center = find_subpixel_center(target[i,c])
                    error = torch.norm(pred_center - target_center)
                    errors.append(error.item())
                    print(f"Sample {i} target {c} - Error: {error.item():.2f}px")  # Debug
        
        avg_error = sum(errors)/len(errors) if errors else float('nan')
        print(f"Calculated average error: {avg_error:.2f}px")  # Debug
    else:
        iou = 0.0
        avg_error = float('nan')
        print("No valid targets found for metrics calculation")  # Debug
    
    return iou, avg_error
def visualize_heatmaps(images, pred_heatmaps, true_heatmaps, idx=0):
    """Enhanced visualization for Colab with proper scaling and display"""
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Move tensors to CPU and convert to numpy
    img = images[idx].cpu().float()
    pred_hm = pred_heatmaps[idx,0].cpu().numpy()
    true_hm = true_heatmaps[idx,0].cpu().numpy()
    
    # Inverse normalize if needed (assuming ImageNet normalization)
    if img.max() > 1.0:  # Check if normalization was applied
        inv_normalize = Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        )
        img = inv_normalize(img)
    img = img.clamp(0,1).permute(1,2,0).numpy()
    
    # Find centers
    pred_center = find_subpixel_center(torch.from_numpy(pred_hm))
    true_center = find_subpixel_center(torch.from_numpy(true_hm))
    error = torch.norm(pred_center - true_center).item()
    
    # Create figure
    plt.figure(figsize=(18, 6))
    
    # Input image with centers
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.scatter([pred_center[0]], [pred_center[1]], 
                c='red', marker='x', s=100, linewidths=2, label='Predicted')
    plt.scatter([true_center[0]], [true_center[1]], 
                c='lime', marker='+', s=100, linewidths=3, label='Ground Truth')
    plt.title(f'Input Image\nError: {error:.2f}px')
    plt.legend()
    
    # Predicted heatmap
    plt.subplot(1, 3, 2)
    plt.imshow(pred_hm, cmap='hot')
    plt.scatter([pred_center[0]], [pred_center[1]], 
                c='cyan', marker='o', s=50, alpha=0.5)
    plt.colorbar(label='Activation')
    plt.title('Predicted Heatmap')
    
    # Ground truth heatmap
    plt.subplot(1, 3, 3)
    plt.imshow(true_hm, cmap='hot')
    plt.scatter([true_center[0]], [true_center[1]], 
                c='cyan', marker='o', s=50, alpha=0.5)
    plt.colorbar(label='Activation')
    plt.title('Ground Truth Heatmap')
    
    plt.tight_layout()
    plt.show()
    
    return error
    
def validate(model, dataloader, device, writer=None, epoch=None):
    """Complete validation function with Colab-friendly visualization"""
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    total_error = 0.0
    num_batches = 0
    valid_samples = 0
    
    # For visualization
    viz_images, viz_preds, viz_targets = [], [], []
    
    with torch.no_grad():
        for batch_idx, (images, points) in enumerate(dataloader):
            # Move data to device
            images = images.float().to(device)
            batch_size = images.size(0)
            
            # Generate heatmaps
            heatmaps = generate_heatmaps(
                points, 
                (images.size(-2), images.size(-1)),
                sigma=5.0
            ).to(device)
            
            # Forward pass
            pred_heatmaps = model(images)
            
            # Store first batch for visualization
            if batch_idx == 0:
                viz_images = images[:2].cpu()  # Store first 2 samples
                viz_preds = pred_heatmaps[:2].cpu()
                viz_targets = heatmaps[:2].cpu()
            
            # Calculate loss
            loss = heatmap_loss(pred_heatmaps, heatmaps)
            total_loss += loss.item()
            
            # Calculate metrics
            batch_iou, batch_error = calculate_metrics(pred_heatmaps, heatmaps)
            
            if not np.isnan(batch_iou):
                total_iou += batch_iou * batch_size
                valid_samples += batch_size
            if not np.isnan(batch_error):
                total_error += batch_error * batch_size
            
            num_batches += 1
    
    # Visualization (only for first batch)
    if len(viz_images) > 0:
        for i in range(min(2, len(viz_images))):  # Visualize up to 2 samples
            visualize_sample(
                image=viz_images[i],
                pred_heatmap=viz_preds[i],
                true_heatmap=viz_targets[i],
                sample_idx=i,
                epoch=epoch
            )
    
    # Calculate averages
    avg_loss = total_loss / num_batches
    avg_iou = total_iou / valid_samples if valid_samples > 0 else 0
    avg_error = total_error / valid_samples if valid_samples > 0 else float('nan')
    
    # Print summary
    print(f"\nValidation Summary:")
    print(f"Samples: {valid_samples}/{len(dataloader.dataset)}")
    print(f"Loss: {avg_loss:.6f} | IoU: {avg_iou:.4f} | Error: {avg_error:.2f}px")
    
    # TensorBoard logging
    if writer is not None and epoch is not None:
        writer.add_scalar('Loss/val', avg_loss, epoch)
        writer.add_scalar('Metrics/IoU', avg_iou, epoch)
        writer.add_scalar('Metrics/Center_Error', avg_error, epoch)
    
    return avg_loss, avg_iou, avg_error

def visualize_sample(image, pred_heatmap, true_heatmap, sample_idx=0, epoch=None):
    """Enhanced visualization for Colab"""
    import matplotlib.pyplot as plt
    
    # Inverse normalize if needed
    if image.max() > 1.0:
        inv_normalize = Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        )
        image = inv_normalize(image)
    image = image.clamp(0,1).permute(1,2,0).numpy()
    
    # Process heatmaps
    pred_hm = pred_heatmap[0].numpy() if pred_heatmap.dim() == 3 else pred_heatmap.numpy()
    true_hm = true_heatmap[0].numpy() if true_heatmap.dim() == 3 else true_heatmap.numpy()
    
    # Find centers
    pred_center = find_subpixel_center(torch.from_numpy(pred_hm))
    true_center = find_subpixel_center(torch.from_numpy(true_hm))
    error = torch.norm(pred_center - true_center).item()
    
    # Create figure
    plt.figure(figsize=(18, 6))
    plt.suptitle(f"Sample {sample_idx} | Epoch {epoch} | Error: {error:.2f}px", y=1.05)
    
    # Input image
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.scatter([pred_center[0]], [pred_center[1]], 
                c='red', marker='x', s=100, linewidths=2, label='Predicted')
    plt.scatter([true_center[0]], [true_center[1]], 
                c='lime', marker='+', s=100, linewidths=3, label='Ground Truth')
    plt.title('Input Image')
    plt.legend()
    
    # Predicted heatmap
    plt.subplot(1, 3, 2)
    plt.imshow(pred_hm, cmap='hot', vmin=0, vmax=1)
    plt.scatter([pred_center[0]], [pred_center[1]], 
                c='cyan', marker='o', s=50, alpha=0.7)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title('Predicted Heatmap')
    
    # Ground truth heatmap
    plt.subplot(1, 3, 3)
    plt.imshow(true_hm, cmap='hot', vmin=0, vmax=1)
    plt.scatter([true_center[0]], [true_center[1]], 
                c='cyan', marker='o', s=50, alpha=0.7)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title('Ground Truth Heatmap')
    
    plt.tight_layout()
    plt.show()

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
    pretrained_path = "./best_model_two_points.pth"  # Path to your existing model
    if os.path.exists(pretrained_path):
        print(f"Loading pretrained weights from {pretrained_path}")
        model.load_state_dict(torch.load(pretrained_path, map_location=device))
        print("Pretrained weights loaded successfully")
    else:
        print("No pretrained weights found, starting from scratch")
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