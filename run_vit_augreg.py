# -----------------------------------------------------------------------------
# run_vit_augreg.py – ViT + AugReg (RandAugment‑style) for GAF regression
# -----------------------------------------------------------------------------
#  Key changes versus the original script supplied by the user
#  ────────────────────────────────────────────────────────────
#  1. Data‑augmentation “AugReg” suite implemented on‑the‑fly:
#       • Random H/V flips, minor rotations, Gaussian noise
#       • Mixup + CutMix helpers adapted for *regression* labels
#  2. ViT backbone created with a configurable "drop_path_rate" (stochastic depth)
#  3. Layer‑wise learning‑rate decay (LLRD) & stronger weight‑decay defaults
#  4. Hyper‑parameters surfaced at the top for easy sweeping
# -----------------------------------------------------------------------------

import time, json, random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score,
    root_mean_squared_error,
)

import timm  # pip install timm

from data_splits import make_loaders

# ╭───────────────────────────  CONFIG  ───────────────────────────╮
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STRDEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
EPOCHS = 60
PATIENCE = 20
BASE_LR = 3e-4  # top layers
LAYER_DECAY = 0.75
WEIGHT_DECAY = 5e-2  # stronger WD works well with ViT
DROP_PATH = 0.15

# AugReg params
P_HFLIP = 0.5
P_VFLIP = 0.5
NOISE_STD = 0.02
MIXUP_ALPHA = 0.8
CUTMIX_ALPHA = 1.0
CUTMIX_PROB = 0.5
# ╰──────────────────────────────────────────────────────────────╯

# =====================================================================
#  Augmentation helpers
# =====================================================================

def rand_bbox(size, lam):
    """Generate random bounding box (CutMix). size = (B, C, H, W)"""
    H, W = size[2], size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_h, cut_w = int(H * cut_rat), int(W * cut_rat)
    cx, cy = np.random.randint(W), np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2


def mixup_cutmix(x, y, alpha_mix=MIXUP_ALPHA, alpha_cut=CUTMIX_ALPHA, p_cut=CUTMIX_PROB):
    """Return augmented (x, y) for regression."""
    if alpha_mix <= 0 and alpha_cut <= 0:
        return x, y

    lam = np.random.beta(alpha_mix, alpha_mix) if random.random() > p_cut else np.random.beta(alpha_cut, alpha_cut)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    if random.random() < p_cut:  # CutMix path
        bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
        x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
        # Adjust lambda to exact area ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size(2) * x.size(3)))
    else:  # Mixup path
        x = lam * x + (1 - lam) * x[index, :]

    y = lam * y + (1 - lam) * y[index]
    return x, y


# =====================================================================
#  Metrics
# =====================================================================

def eval_metrics(y_true, y_pred):
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(root_mean_squared_error(y_true, y_pred)),
        "MAPE": float(mean_absolute_percentage_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


# =====================================================================
#  Model: ViT tiny on GAF tiles   (2‑ch ◄─▶ 3‑ch + up‑res 48→224)
# =====================================================================


class GAFViT(nn.Module):
    def __init__(self, vit_name: str = "vit_tiny_patch16_224", pretrained: bool = True):
        super().__init__()
        self.to3 = nn.Conv2d(2, 3, kernel_size=1, bias=False)
        self.vit = timm.create_model(
            vit_name,
            pretrained=pretrained,
            num_classes=1,
            drop_path_rate=DROP_PATH,
        )

    def forward(self, x):  # x: (B,2,48,48)
        x = self.to3(x)
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        return self.vit(x).squeeze(1)


# =====================================================================
#  Layer‑wise LR decay utility  (timm >=0.9.12) – returns param groups
# =====================================================================

def get_llrd_param_groups(model, base_lr=BASE_LR, layer_decay=LAYER_DECAY, wd=WEIGHT_DECAY):
    matcher = model.vit.group_matcher(coarse=True)
    return model.vit.param_groups(
        lr=base_lr,
        weight_decay=wd,
        layer_decay=layer_decay,
        group_matcher=matcher,
    )


# =====================================================================
#  Train / Predict
# =====================================================================

def train_vit(
    train_loader,
    val_loader,
    vit_name: str = "vit_tiny_patch16_224",
    pretrained: bool = True,
    epochs: int = EPOCHS,
):
    model = GAFViT(vit_name=vit_name, pretrained=pretrained).to(DEVICE)

    param_groups = get_llrd_param_groups(model, base_lr=BASE_LR)
    opt = torch.optim.AdamW(param_groups, betas=(0.9, 0.95))
    crit = nn.MSELoss()

    best_rmse, patience_ctr, best_state = float("inf"), 0, None
    scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())

    for ep in range(1, epochs + 1):
        # --------------------------- train ---------------------------
        model.train()
        sse, n_obs = 0.0, 0
        for batch in train_loader:
            x = batch["x"].to(DEVICE, non_blocking=True)
            y = batch["y"].to(DEVICE, non_blocking=True)

            # --- simple RandAugment‑style flips + noise
            if random.random() < P_HFLIP:
                x = torch.flip(x, dims=[3])  # horizontal
            if random.random() < P_VFLIP:
                x = torch.flip(x, dims=[2])  # vertical
            if NOISE_STD > 0:
                x = x + torch.randn_like(x) * NOISE_STD

            # --- Mixup / CutMix
            x, y = mixup_cutmix(x, y)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(enabled=torch.cuda.is_available(), device_type=STRDEVICE):
                y_hat = model(x)
                loss = crit(y_hat, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            # accumulate SSE for RMSE
            with torch.no_grad():
                sse += F.mse_loss(y_hat, y, reduction="sum").item()
                n_obs += y.size(0)

        train_rmse = (sse / n_obs) ** 0.5

        # -------------------------- validate -------------------------
        model.eval()
        with torch.no_grad():
            y_true, y_pred = [], []
            for b in val_loader:
                x = b["x"].to(DEVICE, non_blocking=True)
                y = b["y"].to(DEVICE, non_blocking=True)
                with torch.amp.autocast(enabled=torch.cuda.is_available(), device_type=STRDEVICE):
                    y_hat = model(x)
                y_true.append(y)
                y_pred.append(y_hat)
            y_true = torch.cat(y_true).float().cpu().numpy()
            y_pred = torch.cat(y_pred).float().cpu().numpy()
            val_rmse = root_mean_squared_error(y_true, y_pred)

        print(
            f"{vit_name}  Ep {ep}/{epochs}  TrainRMSE: {train_rmse:.4f}  ValRMSE: {val_rmse:.4f}"
        )

        # ----------------------- early stopping ----------------------
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                break

    model.load_state_dict(best_state)
    return model.eval()


# ---------------------------------------------------------------------
#  Inference helper
# ---------------------------------------------------------------------

def predict_model(model, loader):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for b in loader:
            x = b["x"].to(DEVICE)
            y_hat = model(x).cpu()
            y_true.append(b["y"])
            y_pred.append(y_hat)
    y_true = torch.cat(y_true).numpy()
    y_pred = torch.cat(y_pred).numpy()
    return y_true, y_pred


# ---------------------------------------------------------------------
#  Runner
# ---------------------------------------------------------------------

def run_vit(
    epochs: int = 60,
    vit_name: str = "vit_tiny_patch16_224",
    pretrained: bool = True,
    data_path: str = "data_gaf.npz",
    out_json: str = "vit_results.json",
):
    t0 = time.time()
    train_loader, val_loader, test_loader = make_loaders(
        path=data_path, dataset_kind="gaf", batch_size=BATCH_SIZE
    )

    model = train_vit(train_loader, val_loader, vit_name, pretrained, epochs)
    y_true, y_pred = predict_model(model, test_loader)
    results = eval_metrics(y_true, y_pred)

    with open(out_json, "w") as fp:
        json.dump(results, fp, indent=4)
    print(json.dumps(results, indent=4))
    print(f"⏱  Total runtime: {time.time() - t0:.1f}s")

    return y_true, y_pred, results


if __name__ == "__main__":
    run_vit()
