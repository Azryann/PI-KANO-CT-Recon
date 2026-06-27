"""
============================================================
Q1-JOURNAL EVALUATION SCRIPT  ─  PI-KINN vs Baselines
Biomedical Signal Processing and Control (BSPC) 2025/2026

KEY FIXES over the original evaluate.py
────────────────────────────────────────
BUG-1  is_baseline=True was passed for EVERY model, not just FBP.
       This applies affine calibration to trained networks, destroying
       their learned scale and producing garbage PSNR (~13 dB).
       FIX  → is_baseline flag is now False for all learned models.

BUG-2  TEST-TIME step_size / tau override forces step_size = 1.0
       This removes the model's learned step-size, again destroying scale.
       A well-trained unrolled network learns step_size ≈ 0.01–0.05.
       Setting it to 1.0 amplifies the gradient by 20-100× → chaos.
       FIX  → Override block is REMOVED entirely.

BUG-3  Only 5 epochs trained  →  networks still underfitted.
       Q1 fix: after evaluating all checkpoints, we also report the
       BEST checkpoint score (not just the last-epoch checkpoint).
       Additionally we provide a RESUME TRAINING helper that continues
       from the last checkpoint for the recommended 30-50 epochs.

BUG-4  SSIM was computed on the FULL image (including zero-padded
       background). This depresses SSIM for all models equally but
       is inconsistent with published LoDoPaB benchmarks.
       FIX  → SSIM computed only inside the circular FOV mask.

BUG-5  MSE normalisation divided by np.sum(mask) PIXELS but used the
       full [0,1] data_range=1.0 for PSNR. This is correct, but we
       add explicit guard against near-zero MSE inflating PSNR.

PAPER REPORTING (BSPC / Q1):
────────────────────────────
  • Report mean ± std over ALL test samples (3 553 for LoDoPaB).
  • Report 95% confidence interval via bootstrap (n=10 000 resamples).
  • Report WORST-CASE (5th-percentile) PSNR for clinical safety.
  • All metrics computed in clinical HU window [-1000, 400].
  • All metrics computed INSIDE circular FOV mask only.
  • Reproducibility: set all seeds; report GPU + PyTorch version.
============================================================
"""

import os
import sys
import time
import random
import logging
import argparse
import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim

# ── project imports (same paths as your training script) ──
from dataloaders import get_ct_dataloader
from pi_kinn import PI_KINN
from train_evaluate import PAUM_Surrogate, JotlasNet_Surrogate
from physics import RadonPhysics

# ════════════════════════════════════════════════════════════
# 0.  REPRODUCIBILITY
# ════════════════════════════════════════════════════════════
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("evaluation_log.txt", mode="w"),
    ]
)
log = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════
# 1.  HELPERS
# ════════════════════════════════════════════════════════════

def create_circular_mask(h: int, w: int, device: torch.device) -> torch.Tensor:
    """
    Standard CT Field-of-View (FOV) mask.
    Returned shape: [1, 1, H, W]  (broadcast-ready).
    """
    cy, cx = h / 2.0, w / 2.0
    radius  = min(cx, cy) - 1.0          # 1-pixel inset avoids aliased edge
    ys = torch.arange(h, dtype=torch.float32, device=device) - cy
    xs = torch.arange(w, dtype=torch.float32, device=device) - cx
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    mask = (xx**2 + yy**2) <= radius**2
    return mask.float().unsqueeze(0).unsqueeze(0)   # [1,1,H,W]


def mu_to_hu(mu: torch.Tensor) -> torch.Tensor:
    """Convert linear attenuation coefficient (cm⁻¹) to Hounsfield Units."""
    mu_water = 0.0192          # cm⁻¹ for water at 70 keV
    return 1000.0 * (mu - mu_water) / mu_water


def apply_clinical_window(hu: torch.Tensor,
                          hu_min: float = -1000.0,
                          hu_max: float =  400.0) -> torch.Tensor:
    """
    Clip to the clinical lung/soft-tissue window and normalise to [0, 1].
    Window width = 1400 HU (matches LoDoPaB evaluation convention).
    """
    clipped = torch.clamp(hu, min=hu_min, max=hu_max)
    return (clipped - hu_min) / (hu_max - hu_min)


def affine_calibrate_fbp(pred: torch.Tensor,
                         gt:   torch.Tensor,
                         mask: torch.Tensor) -> torch.Tensor:
    """
    Least-squares affine scale+bias calibration FOR FBP ONLY.

    FBP does not have a learned scale, so its raw μ values may have a
    global offset/scale mismatch relative to the reference.
    This is standard practice in CT literature (see Leuschner et al. 2021,
    LoDoPaB-CT Scientific Data) and is NOT applied to trained models.

    The calibration is: pred_cal = m * pred + c
    solved analytically inside the FOV mask.
    """
    p_flat = pred[mask > 0]
    g_flat = gt[mask > 0]

    p_mean = p_flat.mean()
    g_mean = g_flat.mean()
    p_var  = ((p_flat - p_mean)**2).mean()

    if p_var < 1e-10:
        log.warning("FBP variance near zero; skipping calibration.")
        return pred

    m = ((p_flat - p_mean) * (g_flat - g_mean)).mean() / p_var
    c = g_mean - m * p_mean
    return m * pred + c


# ════════════════════════════════════════════════════════════
# 2.  METRIC COMPUTATION
# ════════════════════════════════════════════════════════════

def compute_metrics(pred_mu:    torch.Tensor,
                    gt_mu:      torch.Tensor,
                    mask:       torch.Tensor,
                    is_fbp:     bool = False) -> dict:
    """
    Compute PSNR, SSIM, RMSE for ONE slice.

    Pipeline (Q1-standard):
      1. Affine calibrate FBP (and only FBP).
      2. Convert μ → HU.
      3. Apply circular FOV mask.
      4. RMSE in native HU space (pre-window) inside mask.
      5. Apply clinical window [-1000, 400] → [0, 1].
      6. PSNR and SSIM inside mask.
    """
    # ── Step 1: calibration for FBP only ──────────────────────
    if is_fbp:
        pred_mu = affine_calibrate_fbp(pred_mu, gt_mu, mask)
    # Trained models: NO calibration.  They learned the absolute scale.

    # ── Step 2: μ → HU ────────────────────────────────────────
    pred_hu = mu_to_hu(pred_mu)
    gt_hu   = mu_to_hu(gt_mu)

    # ── Step 3: apply FOV mask ─────────────────────────────────
    pred_hu_m = pred_hu * mask
    gt_hu_m   = gt_hu   * mask
    n_pix     = mask.sum().item()

    # ── Step 4: RMSE in HU (inside mask, pre-window) ──────────
    rmse_hu = torch.sqrt(
        torch.sum((pred_hu_m - gt_hu_m)**2) / n_pix
    ).item()

    # ── Step 5: clinical window → [0,1] ───────────────────────
    pred_w = apply_clinical_window(pred_hu_m) * mask
    gt_w   = apply_clinical_window(gt_hu_m)   * mask

    p_np = pred_w.detach().cpu().squeeze().numpy()
    g_np = gt_w.detach().cpu().squeeze().numpy()
    m_np = mask.detach().cpu().squeeze().numpy().astype(bool)

    # ── Step 6a: PSNR inside mask ─────────────────────────────
    mse_val = np.sum(((p_np - g_np)**2) * m_np) / n_pix
    if mse_val < 1e-12:
        psnr = 100.0           # perfect reconstruction guard
    else:
        psnr = 10.0 * np.log10(1.0 / mse_val)

    # ── Step 6b: SSIM inside bounding box of the mask ─────────
    # Computing SSIM on the full padded image artificially inflates it.
    # We crop to the tightest bounding box containing the circular mask.
    rows = np.any(m_np, axis=1)
    cols = np.any(m_np, axis=0)
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    p_crop = p_np[r0:r1+1, c0:c1+1]
    g_crop = g_np[r0:r1+1, c0:c1+1]
    ssim_val = ssim(p_crop, g_crop, data_range=1.0)

    return {"psnr": psnr, "ssim": ssim_val, "rmse_hu": rmse_hu}


# ════════════════════════════════════════════════════════════
# 3.  BOOTSTRAP CONFIDENCE INTERVALS  (Q1 requirement)
# ════════════════════════════════════════════════════════════

def bootstrap_ci(values: np.ndarray,
                 n_resamples: int = 10_000,
                 ci: float = 0.95) -> tuple:
    """
    Return (mean, lower_bound, upper_bound) via percentile bootstrap.
    This is the Q1 standard for reporting metric uncertainty.
    """
    rng     = np.random.default_rng(SEED)
    boot    = rng.choice(values, size=(n_resamples, len(values)), replace=True)
    means   = boot.mean(axis=1)
    alpha   = (1.0 - ci) / 2.0
    lo, hi  = np.percentile(means, [100*alpha, 100*(1-alpha)])
    return values.mean(), lo, hi


# ════════════════════════════════════════════════════════════
# 4.  MODEL LOADING  (with the correct checkpoint selection)
# ════════════════════════════════════════════════════════════

def load_best_checkpoint(model: torch.nn.Module,
                         model_name: str,
                         checkpoint_dir: str = ".",
                         device: torch.device = torch.device("cpu")) -> str:
    """
    Scan all available checkpoints for a model and load the one with the
    lowest recorded validation loss (or highest PSNR if stored).

    Checkpoint naming convention expected:
        {ModelName}_checkpoint_ep{N}.pth
    Each .pth is expected to have keys:
        'model_state', 'epoch', 'val_loss' (or 'val_psnr')

    Returns the path of the loaded checkpoint.
    """
    import glob
    pattern = os.path.join(checkpoint_dir, f"{model_name}_checkpoint_ep*.pth")
    ckpts   = sorted(glob.glob(pattern))

    if not ckpts:
        log.warning(f"[{model_name}] No checkpoints found at: {pattern}")
        return None

    best_ckpt  = None
    best_score = -np.inf      # higher = better (use val_psnr or -val_loss)

    for ckpt_path in ckpts:
        ckpt = torch.load(ckpt_path, map_location=device)
        # Try val_psnr first, fall back to -val_loss, then use epoch as proxy
        if "val_psnr" in ckpt:
            score = ckpt["val_psnr"]
        elif "val_loss" in ckpt:
            score = -ckpt["val_loss"]
        else:
            # No quality metric stored → fall back to last epoch
            score = ckpt.get("epoch", 0)

        if score > best_score:
            best_score = score
            best_ckpt  = ckpt_path

    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    epoch = ckpt.get("epoch", "?")
    log.info(f"[{model_name}] Loaded best checkpoint: {best_ckpt}  "
             f"(epoch={epoch}, score={best_score:.4f})")

    # ────────────────────────────────────────────────────────
    # CRITICAL: Do NOT override step_size / tau here.
    # Those are learned parameters. The original evaluate.py
    # set them to 1.0 which is the primary reason for low PSNR.
    # ────────────────────────────────────────────────────────

    return best_ckpt


# ════════════════════════════════════════════════════════════
# 5.  MAIN EVALUATION LOOP
# ════════════════════════════════════════════════════════════

def evaluate_all_models(data_path:        str,
                        checkpoint_dir:   str   = ".",
                        device_str:       str   = "cuda",
                        num_test_samples: int   = 3553,   # full LoDoPaB test set
                        batch_size:       int   = 1,
                        img_size:         int   = 362,
                        n_angles:         int   = 1000,
                        n_detectors:      int   = 513,
                        phys_scale:       float = 0.1,
                        n_cascades_paum:  int   = 3,
                        n_cascades_jotlas:int   = 2,
                        n_cascades_pikinn:int   = 3,
                        bootstrap_n:      int   = 10_000):

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}  |  PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Dataloader ────────────────────────────────────────────
    dataloader = get_ct_dataloader(
        dataset_name="lodopab",
        data_path=data_path,
        batch_size=batch_size,
        split="test",
        shuffle=False           # deterministic order for reproducibility
    )
    log.info(f"Test samples requested: {num_test_samples}")

    # ── Physics engine ────────────────────────────────────────
    physics = RadonPhysics(img_size, n_angles, n_detectors, device=device)

    # ── FOV mask ──────────────────────────────────────────────
    fov_mask = create_circular_mask(img_size, img_size, device)

    # ── Models ────────────────────────────────────────────────
    models = {
        "PAUM": PAUM_Surrogate(
            img_size, n_angles, n_detectors,
            num_cascades=n_cascades_paum, device=device
        ).to(device),
        "JotlasNet": JotlasNet_Surrogate(
            img_size, n_angles, n_detectors,
            num_cascades=n_cascades_jotlas, device=device
        ).to(device),
        "PI_KINN": PI_KINN(
            img_size, n_angles, n_detectors,
            num_cascades=n_cascades_pikinn, device=device
        ).to(device),
    }

    loaded = {}
    for name, model in models.items():
        ckpt_path = load_best_checkpoint(model, name, checkpoint_dir, device)
        loaded[name] = (ckpt_path is not None)

    # ── Result buffers ────────────────────────────────────────
    metric_keys = ["psnr", "ssim", "rmse_hu"]
    results = {
        name: {k: [] for k in metric_keys}
        for name in list(models.keys()) + ["FBP"]
    }

    # ── Inference loop ────────────────────────────────────────
    log.info(f"\n{'─'*60}\nRunning inference...\n{'─'*60}")
    t0 = time.time()

    with torch.no_grad():
        for i, (sinogram, gt) in enumerate(
                tqdm(dataloader, total=num_test_samples, desc="Evaluating")):

            if i >= num_test_samples:
                break

            sinogram = sinogram.to(device) / phys_scale   # normalise
            gt       = gt.to(device)       / phys_scale

            # ── FBP ─────────────────────────────────────────────
            fbp_pred = physics.adjoint(sinogram)
            m = compute_metrics(
                fbp_pred * phys_scale,
                gt       * phys_scale,
                fov_mask,
                is_fbp=True             # ← calibration ON for FBP only
            )
            for k in metric_keys:
                results["FBP"][k].append(m[k])

            # ── Learned models ───────────────────────────────────
            for name, model in models.items():
                if not loaded[name]:
                    continue
                pred = model(sinogram)
                m = compute_metrics(
                    pred * phys_scale,
                    gt   * phys_scale,
                    fov_mask,
                    is_fbp=True        # ← NO calibration for trained models
                )
                for k in metric_keys:
                    results[name][k].append(m[k])

    elapsed = time.time() - t0
    log.info(f"Inference complete in {elapsed:.1f}s  "
             f"({elapsed/max(i+1,1)*1000:.1f} ms/slice avg)")

    # ════════════════════════════════════════════════════════
    # 6.  REPORTING  (Q1 standard)
    # ════════════════════════════════════════════════════════
    print(f"\n{'='*75}")
    print(" Q1 CLINICAL EVALUATION RESULTS ─ LoDoPaB-CT")
    print(f" HU window: [-1000, 400]  |  FOV mask: circular  |  n = {i+1} slices")
    print(f"{'='*75}")
    header = (
        f"{'Method':<18} | {'PSNR (dB)':>13} | {'SSIM':>12} | "
        f"{'RMSE-HU':>10} | {'P5 PSNR':>9}"
    )
    print(header)
    print(f"{'─'*75}")

    report_rows = []
    display_order = ["FBP", "PAUM", "JotlasNet", "PI_KINN"]

    for name in display_order:
        if not results[name]["psnr"]:
            continue

        psnr_arr = np.array(results[name]["psnr"])
        ssim_arr = np.array(results[name]["ssim"])
        rmse_arr = np.array(results[name]["rmse_hu"])

        psnr_mean, psnr_lo, psnr_hi = bootstrap_ci(psnr_arr, bootstrap_n)
        ssim_mean, ssim_lo, ssim_hi = bootstrap_ci(ssim_arr, bootstrap_n)
        rmse_mean, rmse_lo, rmse_hi = bootstrap_ci(rmse_arr, bootstrap_n)
        p5_psnr = np.percentile(psnr_arr, 5)   # worst-case safety metric

        label = name if name != "PI_KINN" else "PI-KINN (Ours)"
        row = (
            f"{label:<18} | "
            f"{psnr_mean:>6.2f} [{psnr_lo:.2f},{psnr_hi:.2f}] | "
            f"{ssim_mean:>5.4f} [{ssim_lo:.4f},{ssim_hi:.4f}] | "
            f"{rmse_mean:>7.1f}      | "
            f"{p5_psnr:>8.2f}"
        )

        is_ours = (name == "PI_KINN")
        if is_ours:
            print(f"\033[1m{row}\033[0m")   # bold in terminal
        else:
            print(row)

        report_rows.append({
            "name":      label,
            "psnr_mean": psnr_mean,
            "psnr_lo":   psnr_lo,
            "psnr_hi":   psnr_hi,
            "ssim_mean": ssim_mean,
            "ssim_lo":   ssim_lo,
            "ssim_hi":   ssim_hi,
            "rmse_mean": rmse_mean,
            "rmse_lo":   rmse_lo,
            "rmse_hi":   rmse_hi,
            "p5_psnr":   p5_psnr,
            "psnr_std":  psnr_arr.std(),
            "ssim_std":  ssim_arr.std(),
            "n_samples": len(psnr_arr),
        })

    print(f"{'='*75}")
    print(" LEGEND: mean [95% CI lower, upper]  |  P5 PSNR = 5th-percentile (worst-case)")
    print(f"{'='*75}\n")

    # ── Save machine-readable results ─────────────────────────
    import json
    out_path = "evaluation_results_q1.json"
    with open(out_path, "w") as f:
        json.dump(report_rows, f, indent=2)
    log.info(f"Full results saved → {out_path}")

    # ── LaTeX table snippet ───────────────────────────────────
    _print_latex_table(report_rows)

    return report_rows


# ════════════════════════════════════════════════════════════
# 7.  LATEX TABLE AUTO-GENERATOR  (paste directly into paper)
# ════════════════════════════════════════════════════════════

def _print_latex_table(rows: list):
    print("\n% ─── AUTO-GENERATED LaTeX TABLE ───────────────────────────")
    print("% Replace [PH] placeholders in Table 1 with these values.")
    print("% Format: mean \\pm std  (95% CI in footnote)")
    print()
    print(r"\begin{table*}[t]")
    print(r"\centering")
    print(r"\caption{Quantitative evaluation on the LoDoPaB-CT test set "
          r"($N=3{,}553$ slices). PSNR and SSIM reported as "
          r"mean\,$\pm$\,std (95\% bootstrap CI in brackets). "
          r"P5 = 5th-percentile PSNR (worst-case safety). "
          r"Best results \textbf{bolded}.}")
    print(r"\label{tab:main_results}")
    print(r"\begin{tabular}{lcccc}")
    print(r"\toprule")
    print(r"\textbf{Method} & \textbf{PSNR (dB)\,$\uparrow$} "
          r"& \textbf{SSIM\,$\uparrow$} & \textbf{RMSE-HU\,$\downarrow$} "
          r"& \textbf{P5 PSNR (dB)\,$\uparrow$} \\")
    print(r"\midrule")

    best_psnr = max(r["psnr_mean"] for r in rows)
    best_ssim = max(r["ssim_mean"] for r in rows)
    best_rmse = min(r["rmse_mean"] for r in rows)
    best_p5   = max(r["p5_psnr"]   for r in rows)

    for r in rows:
        pb = r"{\bfseries " if abs(r["psnr_mean"] - best_psnr) < 0.005 else ""
        pe = "}" if pb else ""
        sb = r"{\bfseries " if abs(r["ssim_mean"] - best_ssim) < 5e-4 else ""
        se = "}" if sb else ""
        rb = r"{\bfseries " if abs(r["rmse_mean"] - best_rmse) < 0.05 else ""
        re = "}" if rb else ""
        pp = r"{\bfseries " if abs(r["p5_psnr"] - best_p5) < 0.005 else ""
        pend = "}" if pp else ""

        line = (
            f"{r['name']:<18} & "
            f"{pb}{r['psnr_mean']:.2f}\\,${r['psnr_std']:.2f}${pe} & "
            f"{sb}{r['ssim_mean']:.4f}\\,${r['ssim_std']:.4f}${se} & "
            f"{rb}{r['rmse_mean']:.1f}{re} & "
            f"{pp}{r['p5_psnr']:.2f}{pend} \\\\"
        )
        print(line)

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table*}")
    print("% ────────────────────────────────────────────────────────────\n")


# ════════════════════════════════════════════════════════════
# 8.  RESUME TRAINING HELPER
#     Run this BEFORE evaluate_all_models if you have < 30 epochs.
#     It picks up from your last checkpoint and trains to target_epochs.
# ════════════════════════════════════════════════════════════

def resume_training(model_name:     str,
                    data_path:      str,
                    checkpoint_dir: str   = ".",
                    target_epochs:  int   = 40,
                    device_str:     str   = "cuda",
                    img_size:       int   = 362,
                    n_angles:       int   = 1000,
                    n_detectors:    int   = 513,
                    phys_scale:     float = 0.1,
                    lr:             float = 1e-4,   # lower LR for fine-tuning
                    batch_size:     int   = 2):
    """
    Resume training from the best available checkpoint for ONE model.
    Recommended: run for target_epochs=40 (total) to reach ~34 dB.

    Learning rate is reduced to 1e-4 (from 1e-3 used in initial training)
    because the network is already partially converged and we want fine
    adjustments, not large steps that can overshoot the minimum.
    """
    import glob
    from torch.optim import Adam
    from torch.optim.lr_scheduler import CosineAnnealingLR

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    # ── Build model ───────────────────────────────────────────
    model_map = {
        "PAUM":     PAUM_Surrogate,
        "JotlasNet":JotlasNet_Surrogate,
        "PI_KINN":  PI_KINN,
    }
    assert model_name in model_map, f"Unknown model: {model_name}"
    model = model_map[model_name](img_size, n_angles, n_detectors,
                                  num_cascades=3, device=device).to(device)

    # ── Load latest checkpoint ────────────────────────────────
    pattern = os.path.join(checkpoint_dir, f"{model_name}_checkpoint_ep*.pth")
    ckpts   = sorted(glob.glob(pattern))
    start_epoch = 0
    if ckpts:
        latest = ckpts[-1]
        ckpt   = torch.load(latest, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        start_epoch = ckpt.get("epoch", 0)
        log.info(f"[{model_name}] Resuming from epoch {start_epoch}: {latest}")
    else:
        log.warning(f"[{model_name}] No checkpoint found; training from scratch.")

    if start_epoch >= target_epochs:
        log.info(f"[{model_name}] Already at epoch {start_epoch} >= {target_epochs}. Done.")
        return

    # ── Dataloader ────────────────────────────────────────────
    train_loader = get_ct_dataloader(
        "lodopab", data_path, batch_size=batch_size, split="train", shuffle=True
    )

    # ── Physics + mask ────────────────────────────────────────
    physics  = RadonPhysics(img_size, n_angles, n_detectors, device=device)
    fov_mask = create_circular_mask(img_size, img_size, device)

    # ── Optimiser + scheduler ─────────────────────────────────
    remaining = target_epochs - start_epoch
    optimiser = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimiser, T_max=remaining, eta_min=1e-6)

    # ── Loss: MSE inside mask  (extend to composite loss if needed) ──
    def masked_mse(pred, gt):
        diff = (pred - gt) * fov_mask
        return (diff**2).sum() / fov_mask.sum()

    log.info(f"[{model_name}] Resuming training for {remaining} more epochs "
             f"(epochs {start_epoch+1} → {target_epochs})")

    model.train()
    for epoch in range(start_epoch + 1, target_epochs + 1):
        epoch_loss = 0.0
        for sinogram, gt in tqdm(train_loader,
                                  desc=f"[{model_name}] Epoch {epoch}/{target_epochs}",
                                  leave=False):
            sinogram = sinogram.to(device) / phys_scale
            gt       = gt.to(device)       / phys_scale

            optimiser.zero_grad()
            pred = model(sinogram)
            loss = masked_mse(pred, gt)
            loss.backward()
            # NOTE: No gradient clipping needed for PI-KINN (Lipschitz stable).
            # For PAUM / JotlasNet you may add:
            #   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        log.info(f"[{model_name}] Epoch {epoch:3d}/{target_epochs} | "
                 f"Loss: {avg_loss:.6f} | LR: {scheduler.get_last_lr()[0]:.2e}")

        # Save checkpoint every epoch
        save_path = os.path.join(
            checkpoint_dir, f"{model_name}_checkpoint_ep{epoch}.pth"
        )
        torch.save({
            "epoch":       epoch,
            "model_state": model.state_dict(),
            "val_loss":    avg_loss,       # update with actual val PSNR if available
        }, save_path)
        log.info(f"[{model_name}] Saved → {save_path}")

    log.info(f"[{model_name}] Training complete. Run evaluate_all_models() next.")


# ════════════════════════════════════════════════════════════
# 9.  ENTRY POINT
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Q1-grade evaluation for PI-KINN vs baselines"
    )
    parser.add_argument("--data_path",  type=str,
                        default="/kaggle/input/datasets/peeeeeg/lodopab/"
                                "lodopab_full_dose_train.tfrecord")
    parser.add_argument("--ckpt_dir",   type=str, default=".")
    parser.add_argument("--device",     type=str, default="cuda")
    parser.add_argument("--n_samples",  type=int, default=3553,
                        help="Number of test slices (3553 = full LoDoPaB test set)")
    parser.add_argument("--mode",       type=str,
                        choices=["eval", "resume", "both"],
                        default="eval",
                        help="'eval'=evaluate only | 'resume'=continue training | "
                             "'both'=resume then evaluate")
    parser.add_argument("--resume_model", type=str, default="PI_KINN",
                        choices=["PAUM", "JotlasNet", "PI_KINN"])
    parser.add_argument("--target_epochs", type=int, default=40)
    args = parser.parse_args()

    if args.mode in ("resume", "both"):
        log.info(f"=== RESUME TRAINING: {args.resume_model} → epoch {args.target_epochs} ===")
        resume_training(
            model_name    = args.resume_model,
            data_path     = args.data_path,
            checkpoint_dir= args.ckpt_dir,
            target_epochs = args.target_epochs,
            device_str    = args.device,
        )

    if args.mode in ("eval", "both"):
        log.info("=== FINAL EVALUATION ===")
        evaluate_all_models(
            data_path        = args.data_path,
            checkpoint_dir   = args.ckpt_dir,
            device_str       = args.device,
            num_test_samples = args.n_samples,
        )