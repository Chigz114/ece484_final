"""
Train Neural Pose Estimator (NPE) for vision-based quadrotor navigation.
Input: RGB image (640x480)
Output: 6-DOF pose [x, y, z, sin(yaw), cos(yaw)]
        (roll and pitch are assumed to be 0)

Based on FalconGym paper approach but with lighter backbone for faster iteration.
"""
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt


class NPEDataset(Dataset):
    """Dataset for Neural Pose Estimator training."""
    
    def __init__(self, data_dir, transform=None, split='train', train_ratio=0.9):
        self.data_dir = data_dir
        self.images_dir = os.path.join(data_dir, 'images')
        self.transform = transform
        
        # Load metadata
        with open(os.path.join(data_dir, 'metadata.json'), 'r') as f:
            metadata = json.load(f)
        
        self.poses = metadata['poses']  # List of [x, y, z, roll, pitch, yaw]
        n_samples = len(self.poses)
        
        # Split into train/val
        indices = np.arange(n_samples)
        np.random.seed(42)
        np.random.shuffle(indices)
        
        split_idx = int(n_samples * train_ratio)
        if split == 'train':
            self.indices = indices[:split_idx]
        else:
            self.indices = indices[split_idx:]
        
        print(f"{split.capitalize()} set: {len(self.indices)} samples")
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        
        # Load image
        img_path = os.path.join(self.images_dir, f'frame_{real_idx:05d}.png')
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Get pose: [x, y, z, roll, pitch, yaw]
        pose = self.poses[real_idx]
        x, y, z, roll, pitch, yaw = pose
        
        # Convert yaw to sin/cos to avoid discontinuity at ±π
        # Output: [x, y, z, sin(yaw), cos(yaw)]
        target = torch.tensor([
            x, y, z,
            np.sin(yaw), np.cos(yaw)
        ], dtype=torch.float32)
        
        return image, target


class NPEModel(nn.Module):
    """
    Neural Pose Estimator using ResNet50 backbone.
    Larger model for better accuracy.
    """
    
    def __init__(self, pretrained=True, backbone='resnet50'):
        super().__init__()
        
        self.backbone_name = backbone
        
        if backbone == 'resnet50':
            # Use ResNet50 as backbone (pretrained on ImageNet)
            base = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
            feature_dim = 2048
        elif backbone == 'resnet34':
            base = models.resnet34(weights='IMAGENET1K_V1' if pretrained else None)
            feature_dim = 512
        else:  # resnet18
            base = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
            feature_dim = 512
        
        # Remove the final FC layer
        self.features = nn.Sequential(*list(base.children())[:-1])
        
        # Regression head with larger capacity for ResNet50
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 5)  # [x, y, z, sin(yaw), cos(yaw)]
        )
    
    def forward(self, x):
        features = self.features(x)
        pose = self.regressor(features)
        return pose


def pose_loss(pred, target):
    """
    Combined loss for pose estimation.
    - MSE for position (x, y, z)
    - MSE for orientation (sin/cos of yaw)
    """
    # Position loss (first 3 components)
    pos_loss = nn.functional.mse_loss(pred[:, :3], target[:, :3])
    
    # Orientation loss (sin/cos of yaw)
    ori_loss = nn.functional.mse_loss(pred[:, 3:], target[:, 3:])
    
    # Weight position more since it's the primary concern
    total_loss = pos_loss + 0.5 * ori_loss
    
    return total_loss, pos_loss, ori_loss


def compute_metrics(pred, target):
    """Compute position error in cm and yaw error in degrees."""
    with torch.no_grad():
        # Position error (Euclidean distance in meters, convert to cm)
        pos_error = torch.sqrt(((pred[:, :3] - target[:, :3]) ** 2).sum(dim=1))
        pos_error_cm = pos_error * 100  # meters to cm
        
        # Yaw error: reconstruct yaw from sin/cos
        pred_yaw = torch.atan2(pred[:, 3], pred[:, 4])
        target_yaw = torch.atan2(target[:, 3], target[:, 4])
        
        # Angular difference (handle wraparound)
        yaw_diff = pred_yaw - target_yaw
        yaw_diff = torch.atan2(torch.sin(yaw_diff), torch.cos(yaw_diff))
        yaw_error_deg = torch.abs(yaw_diff) * 180 / np.pi
        
        return pos_error_cm.mean().item(), yaw_error_deg.mean().item()


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    total_pos_err = 0
    total_yaw_err = 0
    
    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        pred = model(images)
        
        loss, pos_loss, ori_loss = pose_loss(pred, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        pos_err, yaw_err = compute_metrics(pred, targets)
        total_pos_err += pos_err
        total_yaw_err += yaw_err
    
    n = len(loader)
    return total_loss / n, total_pos_err / n, total_yaw_err / n


def validate(model, loader, device):
    model.eval()
    total_loss = 0
    total_pos_err = 0
    total_yaw_err = 0
    
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            
            pred = model(images)
            loss, _, _ = pose_loss(pred, targets)
            
            total_loss += loss.item()
            
            pos_err, yaw_err = compute_metrics(pred, targets)
            total_pos_err += pos_err
            total_yaw_err += yaw_err
    
    n = len(loader)
    return total_loss / n, total_pos_err / n, total_yaw_err / n


def main():
    # Track configurations
    TRACK_CONFIG = {
        "circle": {
            "data_dir": "npe_datasets/circle",
            "output_dir": "npe_models/circle",
        },
        "uturn": {
            "data_dir": "npe_datasets/uturn",
            "output_dir": "npe_models/uturn",
        },
        "lemniscate": {
            "data_dir": "npe_datasets/lemniscate",
            "output_dir": "npe_models/lemniscate",
        },
    }
    
    parser = argparse.ArgumentParser(description='Train NPE model')
    parser.add_argument('--track', type=str, default=None,
                        choices=list(TRACK_CONFIG.keys()),
                        help='Track name (uses track defaults for data_dir and output_dir)')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Dataset directory (optional if --track is specified)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for models (optional if --track is specified)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--backbone', type=str, default='resnet50',
                        choices=['resnet18', 'resnet34', 'resnet50'],
                        help='Backbone architecture')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    
    args = parser.parse_args()
    
    # Resolve data_dir and output_dir from track or direct args
    if args.track:
        track_cfg = TRACK_CONFIG[args.track]
        data_dir = args.data_dir or track_cfg["data_dir"]
        output_dir = args.output_dir or track_cfg["output_dir"]
    else:
        data_dir = args.data_dir or "npe_datasets/circle"
        output_dir = args.output_dir or "npe_models/circle"
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"NPE Training")
    print(f"{'='*60}")
    print(f"Track: {args.track or 'custom'}")
    print(f"Data dir: {data_dir}")
    print(f"Output dir: {output_dir}")
    print(f"{'='*60}\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data transforms (minimal augmentation for sim-only deployment)
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # ResNet input size
        transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Datasets
    train_dataset = NPEDataset(data_dir, transform=train_transform, split='train')
    val_dataset = NPEDataset(data_dir, transform=val_transform, split='val')
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)
    
    # Model
    model = NPEModel(pretrained=True, backbone=args.backbone).to(device)
    print(f"Backbone: {args.backbone}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'train_pos_err': [], 'val_pos_err': [],
        'train_yaw_err': [], 'val_yaw_err': []
    }
    
    best_val_pos_err = float('inf')
    
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Target: Position error < 38cm (gate radius)")
    print("-" * 60)
    
    for epoch in range(args.epochs):
        # Train
        train_loss, train_pos_err, train_yaw_err = train_epoch(
            model, train_loader, optimizer, device)
        
        # Validate
        val_loss, val_pos_err, val_yaw_err = validate(model, val_loader, device)
        
        scheduler.step()
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_pos_err'].append(train_pos_err)
        history['val_pos_err'].append(val_pos_err)
        history['train_yaw_err'].append(train_yaw_err)
        history['val_yaw_err'].append(val_yaw_err)
        
        # Print progress
        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"Loss: {train_loss:.4f}/{val_loss:.4f} | "
              f"Pos Err: {train_pos_err:.1f}/{val_pos_err:.1f} cm | "
              f"Yaw Err: {train_yaw_err:.1f}/{val_yaw_err:.1f}°")
        
        # Save best model
        if val_pos_err < best_val_pos_err:
            best_val_pos_err = val_pos_err
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_pos_err': val_pos_err,
                'val_yaw_err': val_yaw_err,
            }, os.path.join(output_dir, 'best_npe.pth'))
            print(f"  ✓ New best model saved (pos_err: {val_pos_err:.1f} cm)")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(output_dir, f'checkpoint_epoch_{epoch+1}.pth'))
    
    # Save final model
    torch.save({
        'epoch': args.epochs - 1,
        'model_state_dict': model.state_dict(),
    }, os.path.join(output_dir, 'final_npe.pth'))
    
    # Save training history
    with open(os.path.join(output_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    # Plot training curves
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Val')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(history['train_pos_err'], label='Train')
    axes[1].plot(history['val_pos_err'], label='Val')
    axes[1].axhline(y=38, color='r', linestyle='--', label='Gate radius (38cm)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Position Error (cm)')
    axes[1].set_title('Position Error')
    axes[1].legend()
    axes[1].grid(True)
    
    axes[2].plot(history['train_yaw_err'], label='Train')
    axes[2].plot(history['val_yaw_err'], label='Val')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Yaw Error (°)')
    axes[2].set_title('Yaw Error')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=150)
    print(f"\nTraining curves saved to {output_dir}/training_curves.png")
    
    # Summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best validation position error: {best_val_pos_err:.1f} cm")
    print(f"Gate radius: 38 cm")
    if best_val_pos_err < 38:
        print("✓ Model error is within gate radius - promising!")
    else:
        print("✗ Model error exceeds gate radius - may need more data or tuning")
    print(f"\nModels saved to: {output_dir}/")


if __name__ == '__main__':
    main()
