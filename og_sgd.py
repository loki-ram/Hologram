# ============================================================
# ENHANCED INTERFEROGRAM DENOISING MODEL - SGD OPTIMIZER VARIANT
# ============================================================

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from skimage.metrics import structural_similarity as ssim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ------------------------------------------------------------
# 1. DATASET (FIXED PATH)
# ------------------------------------------------------------

class InterferogramDataset(Dataset):
    def __init__(self, root, augment=False):
        self.root = root
        self.augment = augment
        self.files = sorted([f for f in os.listdir(root) if f.startswith("interf")])
        
        if len(self.files) == 0:
            raise ValueError(f"No interferogram files found in {root}")
        
        print(f"✓ Found {len(self.files)} interferogram files")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        noisy_path = os.path.join(self.root, self.files[idx])
        noisy = plt.imread(noisy_path).astype(np.float32)
        if noisy.max() > 1.0:
            noisy /= 255.0

        clean_path = noisy_path.replace("interf_", "Phi_").replace(".png", ".npy")
        clean = np.load(clean_path).astype(np.float32)

        if self.augment:
            if np.random.rand() > 0.5:
                noisy = np.fliplr(noisy)
                clean = np.fliplr(clean)
            if np.random.rand() > 0.5:
                noisy = np.flipud(noisy)
                clean = np.flipud(clean)
            k = np.random.randint(0, 4)
            if k > 0:
                noisy = np.rot90(noisy, k)
                clean = np.rot90(clean, k)

        noisy = torch.from_numpy(noisy.copy()).unsqueeze(0)
        clean = torch.from_numpy(clean.copy()).unsqueeze(0)

        return noisy, clean


class AugmentedSubset(Subset):
    def __init__(self, dataset, indices, augment=False):
        super().__init__(dataset, indices)
        self.augment = augment
        
    def __getitem__(self, idx):
        old_aug = self.dataset.augment
        self.dataset.augment = self.augment
        result = super().__getitem__(idx)
        self.dataset.augment = old_aug
        return result

# ------------------------------------------------------------
# 2. CBAM ATTENTION
# ------------------------------------------------------------

class AttentionBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.SiLU(),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(channels, 1, 7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x * self.channel_attn(x)
        x = x * self.spatial_attn(x)
        return x

# ------------------------------------------------------------
# 3. CONV BLOCK
# ------------------------------------------------------------

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.gn1 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.gn2 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.res = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        r = self.res(x)
        x = F.silu(self.gn1(self.conv1(x)))
        x = self.gn2(self.conv2(x))
        return F.silu(x + r)

# ------------------------------------------------------------
# 4. ENHANCED UNET
# ------------------------------------------------------------

class EnhancedUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = ConvBlock(1, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        self.enc4 = ConvBlock(256, 512)
        self.pool = nn.MaxPool2d(2)
        
        self.mid = ConvBlock(512, 1024)
        self.attn = AttentionBlock(1024)
        
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.dec4 = ConvBlock(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.dec3 = ConvBlock(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec2 = ConvBlock(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec1 = ConvBlock(128, 64)
        
        self.out = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        m = self.attn(self.mid(self.pool(e4)))
        
        d4 = self.dec4(torch.cat([self.up4(m), e4], 1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        
        return self.out(d1)

# ------------------------------------------------------------
# 5. COMBINED LOSS
# ------------------------------------------------------------

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.6, beta=0.3, gamma=0.1):
        super().__init__()
        self.mse = nn.MSELoss()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def ssim_loss(self, pred, target):
        pred_min, pred_max = pred.min(), pred.max()
        target_min, target_max = target.min(), target.max()
        pred_norm = (pred - pred_min) / (pred_max - pred_min + 1e-8)
        target_norm = (target - target_min) / (target_max - target_min + 1e-8)
        
        ssim_val = 0
        for i in range(pred.shape[0]):
            p = pred_norm[i, 0].detach().cpu().numpy()
            t = target_norm[i, 0].detach().cpu().numpy()
            ssim_val += ssim(p, t, data_range=1.0, win_size=7)
        return 1 - (ssim_val / pred.shape[0])

    def gradient_loss(self, pred, target):
        dx_p = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        dx_t = target[:, :, :, 1:] - target[:, :, :, :-1]
        dy_p = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        dy_t = target[:, :, 1:, :] - target[:, :, :-1, :]
        grad_loss = F.l1_loss(dx_p, dx_t) + F.l1_loss(dy_p, dy_t)
        target_grad_mag = (dx_t.abs().mean() + dy_t.abs().mean()) / 2 + 1e-8
        return grad_loss / target_grad_mag

    def forward(self, pred, target):
        mse = self.mse(pred, target)
        ssim_l = self.ssim_loss(pred, target)
        grad = self.gradient_loss(pred, target)
        total = self.alpha * mse + self.beta * ssim_l + self.gamma * grad
        return total, {'mse': mse.item(), 'ssim': 1-ssim_l, 'grad': grad.item()}

# ------------------------------------------------------------
# 6. TRAINING (SGD optimizer)
# ------------------------------------------------------------

def train(model, loader, val_loader, epochs=15, lr=1e-3, patience=10):
    # Use SGD with momentum + small weight decay
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    criterion = CombinedLoss().to(device)
    scaler = torch.cuda.amp.GradScaler() if device == 'cuda' else None

    best_val = float("inf")
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'train_metrics': [], 'val_metrics': []}

    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        total_metrics = {'mse': 0, 'ssim': 0, 'grad': 0}

        pbar = tqdm(loader, desc=f"Epoch {ep+1}/{epochs}")
        for noisy, clean in pbar:
            noisy, clean = noisy.to(device), clean.to(device)

            with torch.cuda.amp.autocast(enabled=(device == 'cuda')):
                pred = model(noisy)
                loss, metrics = criterion(pred, clean)

            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item()
            for k in metrics:
                total_metrics[k] += metrics[k]
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'ssim': f'{metrics["ssim"]:.3f}'})

        avg_train_loss = total_loss / len(loader)
        avg_train_metrics = {k: v/len(loader) for k, v in total_metrics.items()}
        history['train_loss'].append(avg_train_loss)
        history['train_metrics'].append(avg_train_metrics)

        model.eval()
        val_loss = 0.0
        val_met = {'mse': 0, 'ssim': 0, 'grad': 0}
        with torch.no_grad():
            for noisy, clean in val_loader:
                noisy, clean = noisy.to(device), clean.to(device)
                pred = model(noisy)
                loss, metrics = criterion(pred, clean)
                val_loss += loss.item()
                for k in metrics:
                    val_met[k] += metrics[k]

        avg_val_loss = val_loss / len(val_loader)
        avg_val_metrics = {k: v/len(val_loader) for k, v in val_met.items()}
        history['val_loss'].append(avg_val_loss)
        history['val_metrics'].append(avg_val_metrics)

        scheduler.step()
        lr_current = scheduler.get_last_lr()[0]

        print(f"\nEpoch {ep+1}/{epochs}")
        print(f"  Train: Loss={avg_train_loss:.6f} | MSE={avg_train_metrics['mse']:.6f} | SSIM={avg_train_metrics['ssim']:.4f}")
        print(f"  Val:   Loss={avg_val_loss:.6f} | MSE={avg_val_metrics['mse']:.6f} | SSIM={avg_val_metrics['ssim']:.4f}")
        print(f"  LR: {lr_current:.6f}")

        if avg_val_loss < best_val:
            best_val = avg_val_loss
            patience_counter = 0
            torch.save({
                'epoch': ep,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val,
                'val_metrics': avg_val_metrics
            }, "best_model_sgd.pth")
            print(f"  ✓ Best model saved! (Val Loss: {best_val:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⚠ Early stopping triggered at epoch {ep+1}")
                break

    return history

# ------------------------------------------------------------
# 7. VISUALIZATION
# ------------------------------------------------------------

@torch.no_grad()
def visualize_results(model, loader, n=3, save_path="results.png"):
    model.eval()
    noisy, clean = next(iter(loader))
    noisy, clean = noisy.to(device), clean.to(device)
    pred = model(noisy)

    fig, axes = plt.subplots(n, 4, figsize=(16, 4*n))
    for i in range(min(n, noisy.shape[0])):
        axes[i,0].imshow(noisy[i,0].cpu(), cmap="gray")
        axes[i,0].set_title("Noisy Interferogram", fontsize=12)
        axes[i,0].axis('off')
        
        axes[i,1].imshow(pred[i,0].cpu(), cmap="jet")
        axes[i,1].set_title("Predicted Phase", fontsize=12)
        axes[i,1].axis('off')
        
        axes[i,2].imshow(clean[i,0].cpu(), cmap="jet")
        axes[i,2].set_title("Ground Truth Phase", fontsize=12)
        axes[i,2].axis('off')
        
        error = torch.abs(pred[i,0] - clean[i,0]).cpu().numpy()
        im = axes[i,3].imshow(error, cmap="hot")
        axes[i,3].set_title(f"Abs Error (MAE: {error.mean():.4f})", fontsize=12)
        axes[i,3].axis('off')
        plt.colorbar(im, ax=axes[i,3], fraction=0.046)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Visualization saved to {save_path}")

def plot_training_history(history, save_path="training_curves.png"):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0,0].plot(history['train_loss'], label='Train', linewidth=2)
    axes[0,0].plot(history['val_loss'], label='Val', linewidth=2)
    axes[0,0].set_title("Total Loss", fontsize=14, fontweight='bold')
    axes[0,0].set_xlabel("Epoch")
    axes[0,0].set_ylabel("Loss")
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    axes[0,1].plot([m['mse'] for m in history['train_metrics']], label='Train', linewidth=2)
    axes[0,1].plot([m['mse'] for m in history['val_metrics']], label='Val', linewidth=2)
    axes[0,1].set_title("MSE", fontsize=14, fontweight='bold')
    axes[0,1].set_xlabel("Epoch")
    axes[0,1].set_ylabel("MSE")
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    axes[1,0].plot([m['ssim'] for m in history['train_metrics']], label='Train', linewidth=2)
    axes[1,0].plot([m['ssim'] for m in history['val_metrics']], label='Val', linewidth=2)
    axes[1,0].set_title("SSIM", fontsize=14, fontweight='bold')
    axes[1,0].set_xlabel("Epoch")
    axes[1,0].set_ylabel("SSIM")
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    axes[1,0].set_ylim([0, 1])
    
    axes[1,1].plot([m['grad'] for m in history['train_metrics']], label='Train', linewidth=2)
    axes[1,1].plot([m['grad'] for m in history['val_metrics']], label='Val', linewidth=2)
    axes[1,1].set_title("Gradient Loss", fontsize=14, fontweight='bold')
    axes[1,1].set_xlabel("Epoch")
    axes[1,1].set_ylabel("Gradient Loss")
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Training curves saved to {save_path}")

# ------------------------------------------------------------
# 8. MAIN
# ------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train Enhanced Interferogram Denoiser (SGD)")
    p.add_argument("--data_path", default="data/open_fringes", help="Path to dataset (interf_*.png + Phi_*.npy)")
    p.add_argument("--visuals_dir", default=None, help="Directory to save visualizations (defaults to <data_path>/visuals)")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--num_workers", type=int, default=2)
    args = p.parse_args()

    data_path = args.data_path
    visuals_dir = args.visuals_dir or os.path.join(data_path, "visuals")
    os.makedirs(visuals_dir, exist_ok=True)

    print("="*70)
    print("INTERFEROGRAM DENOISING - TRAINING (SGD)")
    print("="*70)

    full_ds = InterferogramDataset(data_path, augment=False)

    train_len = int(0.8 * len(full_ds))
    val_len = len(full_ds) - train_len
    train_indices = list(range(train_len))
    val_indices = list(range(train_len, len(full_ds)))

    train_ds = AugmentedSubset(full_ds, train_indices, augment=True)
    val_ds = AugmentedSubset(full_ds, val_indices, augment=False)

    pin_memory = True if device == 'cuda' else False
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                             num_workers=args.num_workers, pin_memory=pin_memory, persistent_workers=(args.num_workers>0))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                           num_workers=args.num_workers, pin_memory=pin_memory, persistent_workers=(args.num_workers>0))

    print(f"\nDataset Statistics:")
    print(f"  Total samples: {len(full_ds)}")
    print(f"  Training: {train_len} samples")
    print(f"  Validation: {val_len} samples")
    print(f"  Image size: 256x256")
    print(f"  Batch size: {args.batch_size}")
    print("="*70)

    model = EnhancedUNet().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: Enhanced U-Net with CBAM Attention")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: ~{total_params * 4 / 1024**2:.1f} MB")
    print("="*70)

    print("\nStarting training...")
    history = train(model, train_loader, val_loader, epochs=args.epochs, lr=args.lr, patience=args.patience)

    checkpoint = torch.load("best_model_sgd.pth", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Best model from epoch: {checkpoint['epoch']+1}")
    print(f"Best validation loss: {checkpoint['val_loss']:.6f}")
    print(f"Best validation SSIM: {checkpoint['val_metrics']['ssim']:.4f}")
    print("="*70)

    plot_training_history(history, save_path=os.path.join(visuals_dir, "training_curves_sgd.png"))

    print("\nGenerating visualizations...")
    visualize_results(model, val_loader, n=4, save_path=os.path.join(visuals_dir, "results_sgd.png"))

    torch.save(model.state_dict(), "enhanced_interferogram_denoiser_final_sgd.pth")
    print("\n✓ All done! Model saved as 'enhanced_interferogram_denoiser_final_sgd.pth'")
    # Final training summary plots (saved to visuals_dir)
    try:
        save_path = os.path.join(visuals_dir, "training_summary_sgd.png")
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        axes[0,0].plot(history['train_loss'], label='Train', linewidth=2)
        axes[0,0].plot(history['val_loss'], label='Val', linewidth=2)
        axes[0,0].set_title("Total Loss", fontsize=14, fontweight='bold')
        axes[0,0].set_xlabel("Epoch")
        axes[0,0].set_ylabel("Loss")
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)

        axes[0,1].plot([m['mse'] for m in history['train_metrics']], label='Train', linewidth=2)
        axes[0,1].plot([m['mse'] for m in history['val_metrics']], label='Val', linewidth=2)
        axes[0,1].set_title("MSE", fontsize=14, fontweight='bold')
        axes[0,1].set_xlabel("Epoch")
        axes[0,1].set_ylabel("MSE")
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)

        axes[1,0].plot([m['ssim'] for m in history['train_metrics']], label='Train', linewidth=2)
        axes[1,0].plot([m['ssim'] for m in history['val_metrics']], label='Val', linewidth=2)
        axes[1,0].set_title("SSIM", fontsize=14, fontweight='bold')
        axes[1,0].set_xlabel("Epoch")
        axes[1,0].set_ylabel("SSIM")
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        axes[1,0].set_ylim([0, 1])

        axes[1,1].plot([m['grad'] for m in history['train_metrics']], label='Train', linewidth=2)
        axes[1,1].plot([m['grad'] for m in history['val_metrics']], label='Val', linewidth=2)
        axes[1,1].set_title("Gradient Loss", fontsize=14, fontweight='bold')
        axes[1,1].set_xlabel("Epoch")
        axes[1,1].set_ylabel("Gradient Loss")
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)

        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Training summary saved to {save_path}")
    except Exception as e:
        print(f"Warning: failed to save training summary plot: {e}")

