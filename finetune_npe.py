"""
Fine-tune NPE model with additional gate-focused data.
Loads existing model and continues training with combined dataset.
"""
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
import argparse


class NPEDataset(Dataset):
    """Dataset for Neural Pose Estimator training."""
    
    def __init__(self, data_dir, transform=None, split='train', train_ratio=0.9):
        self.data_dir = data_dir
        self.images_dir = os.path.join(data_dir, 'images')
        self.transform = transform
        
        with open(os.path.join(data_dir, 'metadata.json'), 'r') as f:
            metadata = json.load(f)
        
        self.poses = metadata['poses']
        n_samples = len(self.poses)
        
        indices = np.arange(n_samples)
        np.random.seed(42)
        np.random.shuffle(indices)
        
        split_idx = int(n_samples * train_ratio)
        if split == 'train':
            self.indices = indices[:split_idx]
        else:
            self.indices = indices[split_idx:]
        
        print(f"  {data_dir}: {len(self.indices)} {split} samples")
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        
        img_path = os.path.join(self.images_dir, f'frame_{real_idx:05d}.png')
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        pose = self.poses[real_idx]
        x, y, z, roll, pitch, yaw = pose
        
        target = torch.tensor([
            x, y, z,
            np.sin(yaw), np.cos(yaw)
        ], dtype=torch.float32)
        
        return image, target


class NPEModel(nn.Module):
    """Neural Pose Estimator using ResNet backbone."""
    
    def __init__(self, pretrained=True, backbone='resnet50'):
        super().__init__()
        
        if backbone == 'resnet50':
            base = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
            feature_dim = 2048
        elif backbone == 'resnet34':
            base = models.resnet34(weights='IMAGENET1K_V1' if pretrained else None)
            feature_dim = 512
        else:
            base = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
            feature_dim = 512
        
        self.features = nn.Sequential(*list(base.children())[:-1])
        
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 5)
        )
    
    def forward(self, x):
        features = self.features(x)
        return self.regressor(features)


def compute_metrics(pred, target):
    """Compute position and yaw errors."""
    pred_pos = pred[:, :3]
    target_pos = target[:, :3]
    pos_err = torch.norm(pred_pos - target_pos, dim=1).mean().item() * 100  # cm
    
    pred_yaw = torch.atan2(pred[:, 3], pred[:, 4])
    target_yaw = torch.atan2(target[:, 3], target[:, 4])
    yaw_diff = torch.abs(pred_yaw - target_yaw)
    yaw_diff = torch.min(yaw_diff, 2*np.pi - yaw_diff)
    yaw_err = torch.rad2deg(yaw_diff).mean().item()
    
    return pos_err, yaw_err


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    total_pos_err = 0
    total_yaw_err = 0
    
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        
        optimizer.zero_grad()
        pred = model(images)
        loss = criterion(pred, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pos_err, yaw_err = compute_metrics(pred, targets)
        total_pos_err += pos_err
        total_yaw_err += yaw_err
    
    n = len(loader)
    return total_loss / n, total_pos_err / n, total_yaw_err / n


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_pos_err = 0
    total_yaw_err = 0
    
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        pred = model(images)
        loss = criterion(pred, targets)
        
        total_loss += loss.item()
        pos_err, yaw_err = compute_metrics(pred, targets)
        total_pos_err += pos_err
        total_yaw_err += yaw_err
    
    n = len(loader)
    return total_loss / n, total_pos_err / n, total_yaw_err / n


def main():
    parser = argparse.ArgumentParser(description='Fine-tune NPE model with gate-focused data')
    parser.add_argument('--track', type=str, default='lemniscate',
                        help='Track name')
    parser.add_argument('--base_model', type=str, default=None,
                        help='Path to base model checkpoint (default: npe_models/<track>/best_npe.pth)')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of fine-tuning epochs')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='Learning rate (lower for fine-tuning)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    
    args = parser.parse_args()
    
    # Paths
    base_data_dir = f"npe_datasets/{args.track}"
    gate_data_dir = f"npe_datasets/{args.track}_gate_focused"
    output_dir = f"npe_models/{args.track}"
    base_model_path = args.base_model or f"npe_models/{args.track}/best_npe.pth"
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"NPE Fine-Tuning with Gate-Focused Data")
    print(f"{'='*60}")
    print(f"Track: {args.track}")
    print(f"Base model: {base_model_path}")
    print(f"Base data: {base_data_dir}")
    print(f"Gate data: {gate_data_dir}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Combine datasets
    print("\nLoading datasets:")
    base_train = NPEDataset(base_data_dir, transform=train_transform, split='train')
    base_val = NPEDataset(base_data_dir, transform=val_transform, split='val')
    gate_train = NPEDataset(gate_data_dir, transform=train_transform, split='train')
    gate_val = NPEDataset(gate_data_dir, transform=val_transform, split='val')
    
    # Combine train and val datasets
    train_dataset = ConcatDataset([base_train, gate_train])
    val_dataset = ConcatDataset([base_val, gate_val])
    
    print(f"\nCombined: {len(train_dataset)} train, {len(val_dataset)} val samples")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)
    
    # Load base model
    model = NPEModel(pretrained=False, backbone='resnet50').to(device)
    checkpoint = torch.load(base_model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"\nLoaded base model (val_pos_err: {checkpoint.get('val_pos_err', 'N/A'):.1f} cm)")
    
    # Fine-tuning: lower learning rate
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.MSELoss()
    
    best_val_err = float('inf')
    
    print(f"\nFine-tuning for {args.epochs} epochs (lr={args.lr})...")
    print("-" * 60)
    
    for epoch in range(1, args.epochs + 1):
        train_loss, train_pos, train_yaw = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_pos, val_yaw = validate(model, val_loader, criterion, device)
        scheduler.step()
        
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Loss: {train_loss:.4f}/{val_loss:.4f} | "
              f"Pos Err: {train_pos:.1f}/{val_pos:.1f} cm | "
              f"Yaw Err: {train_yaw:.1f}/{val_yaw:.1f}°")
        
        if val_pos < best_val_err:
            best_val_err = val_pos
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_pos_err': val_pos,
                'val_yaw_err': val_yaw,
            }, os.path.join(output_dir, 'best_npe.pth'))
            print(f"  ✓ New best model saved (pos_err: {val_pos:.1f} cm)")
    
    # Save final model
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'val_pos_err': val_pos,
        'val_yaw_err': val_yaw,
    }, os.path.join(output_dir, 'finetuned_npe.pth'))
    
    print(f"\n{'='*60}")
    print(f"FINE-TUNING COMPLETE")
    print(f"{'='*60}")
    print(f"Best validation position error: {best_val_err:.1f} cm")
    print(f"Models saved to: {output_dir}/")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
