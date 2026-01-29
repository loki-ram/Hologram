import argparse
import numpy as np
import os
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from scipy.special import factorial


# ------------------------------------------------------------
# ZERNIKE UTILITIES
# ------------------------------------------------------------

def make_mesh(H, W):
    y = np.linspace(-1, 1, H)
    x = np.linspace(-1, 1, W)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    TH = np.arctan2(Y, X)
    return X, Y, R, TH


def zernike_radial(n, m, rho):
    R = np.zeros_like(rho)
    for s in range((n - abs(m)) // 2 + 1):
        c = ((-1)**s * factorial(n - s)) / (
            factorial(s)
            * factorial((n + abs(m)) // 2 - s)
            * factorial((n - abs(m)) // 2 - s)
        )
        R += c * rho**(n - 2 * s)
    return R


def zernike(n, m, rho, theta):
    R = zernike_radial(n, m, rho)
    Z = R * (np.cos(m * theta) if m >= 0 else np.sin(-m * theta))
    Z[rho > 1] = 0.0
    return Z


def generate_zernike_coeff_list(n_modes=18):
    pairs = [
        (0,0),(1,-1),(1,1),
        (2,-2),(2,0),(2,2),
        (3,-3),(3,-1),(3,1),(3,3),
        (4,-4),(4,-2),(4,0),(4,2),(4,4),
        (5,-5),(5,-3),(5,-1)
    ]
    return pairs[:n_modes]


def zernike_basis_stack(H, W, n_modes=18):
    X, Y, R, TH = make_mesh(H, W)
    pairs = generate_zernike_coeff_list(n_modes)
    B = []
    for (n, m) in pairs:
        B.append(zernike(n, m, R, TH))
    return np.stack(B, axis=0), pairs


# ------------------------------------------------------------
# INTERFEROGRAM GENERATOR
# ------------------------------------------------------------

def generate_sample_open_fringes(
    H=256, W=256,
    fx=0.05, fy=0.02,
    background=0.5, contrast=0.5,
    snr_db=20,
    add_zernike=False, zernike_coeffs=None,
    seed=None
):

    if seed is not None:
        np.random.seed(seed)

    # Coordinate grid
    y = np.linspace(-1, 1, H)
    x = np.linspace(-1, 1, W)
    X, Y = np.meshgrid(x, y)

    # Carrier (open fringes)
    carrier = 2 * np.pi * (fx * X + fy * Y)

    # Phase deformation
    Phi = np.zeros((H, W))
    Phi += 2.0 * X
    Phi += 3.0 * np.exp(-((X - 0.3)**2 + (Y + 0.1)**2) / 0.3)

    # Zernike phase distortion
    if add_zernike and zernike_coeffs is not None:
        B, _ = zernike_basis_stack(H, W, len(zernike_coeffs))
        Phi += np.tensordot(zernike_coeffs, B, axes=(0, 0))

    # Ideal interferogram
    I_clean = background + contrast * np.cos(carrier + Phi)

    # Add Gaussian noise based on SNR
    signal_power = np.mean(I_clean**2)
    noise_power = signal_power / (10**(snr_db / 10))
    noise_sigma = np.sqrt(noise_power)

    I_noisy = I_clean + noise_sigma * np.random.randn(H, W)

    # Normalize to [0,1]
    I_noisy -= I_noisy.min()
    I_noisy /= (I_noisy.max() + 1e-12)

    return {
        "I_noisy": I_noisy.astype(np.float32),
        "Phi": Phi.astype(np.float32)
    }


# ------------------------------------------------------------
# DATASET CREATOR
# ------------------------------------------------------------

def create_dataset(out_dir="data/open_fringes", n=500, H=256, W=256):
    os.makedirs(out_dir, exist_ok=True)

    for i in range(n):
        fx = np.random.uniform(0.01, 0.15)
        fy = np.random.uniform(0.0, 0.1)
        snr_db = np.random.uniform(-5, 25)

        add_z = np.random.rand() < 0.5
        zcoeffs = None
        if add_z:
            zcoeffs = 0.05 * np.random.randn(18)

        sample = generate_sample_open_fringes(
            H=H, W=W,
            fx=fx, fy=fy,
            snr_db=snr_db,
            add_zernike=add_z,
            zernike_coeffs=zcoeffs
        )

        imageio.imwrite(
            os.path.join(out_dir, f"interf_{i:05d}.png"),
            (sample["I_noisy"] * 255).astype(np.uint8)
        )

        np.save(
            os.path.join(out_dir, f"Phi_{i:05d}.npy"),
            sample["Phi"]
        )

    print(f"Dataset created: {n} samples in '{out_dir}'")


# ------------------------------------------------------------
# VISUALIZATION
# ------------------------------------------------------------

def show_one_sample(folder="data/open_fringes", idx=0):
    img_path = os.path.join(folder, f"interf_{idx:05d}.png")
    phi_path = os.path.join(folder, f"Phi_{idx:05d}.npy")

    img = imageio.imread(img_path)
    Phi = np.load(phi_path)

    fig = plt.figure(figsize=(10, 4))

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(img, cmap="gray")
    ax1.set_title("Noisy Interferogram")
    ax1.axis("off")

    ax2 = fig.add_subplot(1, 2, 2)
    im = ax2.imshow(Phi, cmap="jet")
    ax2.set_title("Unwrapped Phase (Ground Truth)")
    ax2.axis("off")
    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

    plt.tight_layout()
    return fig


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Generate interferogram dataset and save visuals")
    p.add_argument("--out_dir", default="data/open_fringes", help="Output dataset directory")
    p.add_argument("--n", type=int, default=20000, help="Number of samples to generate")
    p.add_argument("--H", type=int, default=256, help="Image height")
    p.add_argument("--W", type=int, default=256, help="Image width")
    p.add_argument("--visuals_all", action="store_true", help="Save visualization for all samples")
    p.add_argument("--visuals_dir", default=None, help="Directory to save visuals (defaults to <out_dir>/visuals)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    create_dataset(out_dir=args.out_dir, n=args.n, H=args.H, W=args.W)

    visuals_dir = args.visuals_dir or os.path.join(args.out_dir, "visuals")
    os.makedirs(visuals_dir, exist_ok=True)

    if args.visuals_all:
        for i in range(args.n):
            fig = show_one_sample(folder=args.out_dir, idx=i)
            out_path = os.path.join(visuals_dir, f"visual_{i:05d}.png")
            fig.savefig(out_path, bbox_inches="tight")
            plt.close(fig)
    else:
        fig = show_one_sample(folder=args.out_dir, idx=0)
        out_path = os.path.join(visuals_dir, f"visual_00000.png")
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)

    print(f"Dataset created: {args.n} samples in '{args.out_dir}'")
    print(f"Visualizations saved to '{visuals_dir}'")