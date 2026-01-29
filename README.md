**Interferogram Denoising — Repo README**

Purpose
- Small workspace to generate synthetic interferograms and train an enhanced U-Net denoiser.

Files
- [dataset.py](dataset.py): dataset generator. CLI available; saves interferograms (`interf_*.png`) and ground-truth phases (`Phi_*.npy`).
- [og.py](og.py): training script for the Enhanced U-Net (CBAM + CombinedLoss). Uses CUDA AMP when available and saves visuals to a `visuals` directory.
- [requirements.txt](requirements.txt): Python dependencies (see GPU notes below).

Quick setup (A100 / Linux)
1. Create & activate venv:
```bash
python -m venv venv
source venv/bin/activate
```
2. Install PyTorch for your CUDA version (example for CUDA 11.8):
```bash
pip install --upgrade pip
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
```
3. Install the rest:
```bash
pip install -r requirements.txt
```

Generate dataset
- Example (write to absolute path):
```bash
python dataset.py --out_dir /path/to/data/open_fringes --n 500 --visuals_all
```
- Output:
  - Interferograms: `/path/to/data/open_fringes/interf_00000.png` ...
  - Ground-truth phases: `/path/to/data/open_fringes/Phi_00000.npy` ...
  - Visuals: `/path/to/data/open_fringes/visuals/visual_00000.png` (if `--visuals_all` used)

Train on A100
- Example command:
```bash
python og.py --data_path /path/to/data/open_fringes --visuals_dir /path/to/visuals --epochs 25 --batch_size 8 --num_workers 4
```
- Notes:
  - `og.py` auto-detects CUDA and enables AMP + GradScaler when available.
  - Visual outputs (training curves and results) are saved to the `--visuals_dir`.
  - Saved checkpoints: `best_model.pth` and final model `enhanced_interferogram_denoiser_final.pth`.

Headless server tips
- Matplotlib backend is set to `Agg` so scripts can run without a display.
- Adjust `--num_workers` to match server CPUs for better throughput.

If you want, I can:
- Pin package versions in `requirements.txt`.
- Convert dataset generation to use GPU (CuPy/PyTorch) for faster synthetic-data generation on A100.
- Add a small `run.sh` utility to run dataset generation and training with one command.
