# ============================================================
# BASELINE U-NET FOR INTERFEROGRAM DENOISING (MSE ONLY)
# ============================================================

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ------------------------------------------------------------
# 1. DATASET (SAME PATHS)
# ------------------------------------------------------------

class InterferogramDataset(Dataset):
    def __init__(self, root, augment=False):
        self.root = root
        self.augment = augment
        self.files = sorted([f for f in os.listdir(root) if f.startswith("interf")])

        if len(self.files) == 0:
            raise RuntimeError(f"No interferogram files found in {root}")

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
        old = self.dataset.augment
        self.dataset.augment = self.augment
        sample = super().__getitem__(idx)
        self.dataset.augment = old
        return sample

# ------------------------------------------------------------
# 2. BASELINE U-NET
# ------------------------------------------------------------

class UNetConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class BaselineUNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc1 = UNetConvBlock(1, 64)
        self.enc2 = UNetConvBlock(64, 128)
        self.enc3 = UNetConvBlock(128, 256)
        self.enc4 = UNetConvBlock(256, 512)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = UNetConvBlock(512, 1024)

        self.up4 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.dec4 = UNetConvBlock(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.dec3 = UNetConvBlock(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec2 = UNetConvBlock(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec1 = UNetConvBlock(128, 64)

        self.out = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], 1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))

        return self.out(d1)

# ------------------------------------------------------------
# 3. METRICS AND VISUALIZATION
# ------------------------------------------------------------

def calculate_metrics(pred, clean):
    """Calculate MSE and SSIM"""
    mse = F.mse_loss(pred, clean).item()
    
    # SSIM calculation
    pred_np = pred.detach().cpu().numpy()
    clean_np = clean.detach().cpu().numpy()
    
    ssim_val = 0
    for i in range(pred.shape[0]):
        p = pred_np[i, 0]
        c = clean_np[i, 0]
        # Normalize to [0, 1] for SSIM
        p_norm = (p - p.min()) / (p.max() - p.min() + 1e-8)
        c_norm = (c - c.min()) / (c.max() - c.min() + 1e-8)
        ssim_val += ssim(p_norm, c_norm, data_range=1.0, win_size=7)
    
    ssim_val /= pred.shape[0]
    return mse, ssim_val


@torch.no_grad()
def visualize_results(model, loader, n=3, save_path="baseline_results.png"):
    """Visualize denoising results"""
    model.eval()
    noisy, clean = next(iter(loader))
    noisy, clean = noisy.to(device), clean.to(device)
    pred = model(noisy)

    fig, axes = plt.subplots(n, 4, figsize=(16, 4*n))
    if n == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(min(n, noisy.shape[0])):
        # Noisy Interferogram
        axes[i, 0].imshow(noisy[i, 0].cpu(), cmap="gray")
        axes[i, 0].set_title("Noisy Interferogram", fontsize=12, fontweight='bold')
        axes[i, 0].axis('off')
        
        # Predicted Phase
        axes[i, 1].imshow(pred[i, 0].cpu(), cmap="jet")
        axes[i, 1].set_title("Predicted Phase", fontsize=12, fontweight='bold')
        axes[i, 1].axis('off')
        
        # Ground Truth Phase
        axes[i, 2].imshow(clean[i, 0].cpu(), cmap="jet")
        axes[i, 2].set_title("Ground Truth Phase", fontsize=12, fontweight='bold')
        axes[i, 2].axis('off')
        
        # Absolute Error
        error = torch.abs(pred[i, 0] - clean[i, 0]).cpu().numpy()
        im = axes[i, 3].imshow(error, cmap="hot")
        mae = error.mean()
        axes[i, 3].set_title(f"Abs Error (MAE: {mae:.4f})", fontsize=12, fontweight='bold')
        axes[i, 3].axis('off')
        plt.colorbar(im, ax=axes[i, 3], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Visualization saved to {save_path}")


def plot_training_history(history, save_path="baseline_training_curves.png"):
    """Plot training and validation loss curves"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    ax.plot(history['train'], label='Train Loss', linewidth=2.5, marker='o', markersize=4)
    ax.plot(history['val'], label='Val Loss', linewidth=2.5, marker='s', markersize=4)
    ax.set_title("Baseline U-Net: Training History", fontsize=14, fontweight='bold')
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("MSE Loss", fontsize=12)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Training curves saved to {save_path}")


def plot_metrics_history(history, save_path="baseline_metrics.png"):
    """Plot SSIM and MSE metrics over epochs"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # MSE plot
    axes[0].plot(history['train_mse'], label='Train MSE', linewidth=2.5, marker='o')
    axes[0].plot(history['val_mse'], label='Val MSE', linewidth=2.5, marker='s')
    axes[0].set_title("MSE Metrics", fontsize=13, fontweight='bold')
    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("MSE", fontsize=11)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # SSIM plot
    axes[1].plot(history['train_ssim'], label='Train SSIM', linewidth=2.5, marker='o')
    axes[1].plot(history['val_ssim'], label='Val SSIM', linewidth=2.5, marker='s')
    axes[1].set_title("SSIM Metrics", fontsize=13, fontweight='bold')
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("SSIM", fontsize=11)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Metrics plot saved to {save_path}")

# ------------------------------------------------------------
# 4. TRAINING LOOP (WITH METRICS)
# ------------------------------------------------------------

def train(model, train_loader, val_loader, epochs, lr, patience):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val = float("inf")
    wait = 0
    history = {"train": [], "val": [], "train_mse": [], "val_mse": [], "train_ssim": [], "val_ssim": []}

    for ep in range(epochs):
        model.train()
        train_loss = 0.0
        train_mse = 0.0
        train_ssim = 0.0
        num_batches = 0

        for noisy, clean in tqdm(train_loader, desc=f"Epoch {ep+1}/{epochs}", leave=False):
            noisy, clean = noisy.to(device), clean.to(device)
            pred = model(noisy)
            loss = criterion(pred, clean)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            mse_val, ssim_val = calculate_metrics(pred, clean)
            train_mse += mse_val
            train_ssim += ssim_val
            num_batches += 1

        train_loss /= len(train_loader)
        train_mse /= num_batches
        train_ssim /= num_batches
        history["train"].append(train_loss)
        history["train_mse"].append(train_mse)
        history["train_ssim"].append(train_ssim)

        model.eval()
        val_loss = 0.0
        val_mse = 0.0
        val_ssim = 0.0
        num_val_batches = 0
        
        with torch.no_grad():
            for noisy, clean in val_loader:
                noisy, clean = noisy.to(device), clean.to(device)
                pred = model(noisy)
                val_loss += criterion(pred, clean).item()
                mse_val, ssim_val = calculate_metrics(pred, clean)
                val_mse += mse_val
                val_ssim += ssim_val
                num_val_batches += 1

        val_loss /= len(val_loader)
        val_mse /= num_val_batches
        val_ssim /= num_val_batches
        history["val"].append(val_loss)
        history["val_mse"].append(val_mse)
        history["val_ssim"].append(val_ssim)

        print(f"Epoch {ep+1}/{epochs}: Train Loss={train_loss:.6f} | Val Loss={val_loss:.6f}")
        print(f"           Train SSIM={train_ssim:.4f} | Val SSIM={val_ssim:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            wait = 0
            torch.save(model.state_dict(), "baseline_unet_best.pth")
            print("✓ Best model saved")
        else:
            wait += 1
            if wait >= patience:
                print("⚠ Early stopping")
                break

    return history

# ------------------------------------------------------------
# 5. MAIN
# ------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="data/open_fringes")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--visuals_dir", default=None, help="Directory to save visualizations")
    args = parser.parse_args()

    visuals_dir = args.visuals_dir or os.path.join(args.data_path, "baseline_visuals")
    os.makedirs(visuals_dir, exist_ok=True)

    full_ds = InterferogramDataset(args.data_path)
    n_train = int(0.8 * len(full_ds))
    train_idx = list(range(n_train))
    val_idx = list(range(n_train, len(full_ds)))

    train_ds = AugmentedSubset(full_ds, train_idx, augment=True)
    val_ds = AugmentedSubset(full_ds, val_idx, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            num_workers=args.num_workers)

    print("="*70)
    print("BASELINE U-NET FOR INTERFEROGRAM DENOISING")
    print("="*70)
    
    model = BaselineUNet().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: Baseline U-Net")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Data path: {args.data_path}")
    print(f"  Epochs: {args.epochs} | Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr} | Patience: {args.patience}")
    print("="*70)

    history = train(model, train_loader, val_loader,
                    args.epochs, args.lr, args.patience)

    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Final Train Loss: {history['train'][-1]:.6f}")
    print(f"Final Val Loss: {history['val'][-1]:.6f}")
    print(f"Best Val Loss: {min(history['val']):.6f}")
    print(f"Final Train SSIM: {history['train_ssim'][-1]:.4f}")
    print(f"Final Val SSIM: {history['val_ssim'][-1]:.4f}")
    print("="*70)

    # Save visualizations
    print("\nGenerating visualizations...")
    plot_training_history(history, save_path=os.path.join(visuals_dir, "training_curves.png"))
    plot_metrics_history(history, save_path=os.path.join(visuals_dir, "metrics.png"))
    visualize_results(model, val_loader, n=4, save_path=os.path.join(visuals_dir, "results.png"))
    
    torch.save(model.state_dict(), os.path.join(visuals_dir, "baseline_unet_final.pth"))
    print(f"\n✓ All results saved to: {visuals_dir}")
    print("✓ Baseline training complete!")
