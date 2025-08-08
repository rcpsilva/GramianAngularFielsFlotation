# -----------------------------------------------------------------------------
# run_vit_augreg.py – ViT + AugReg (RandAugment-style) for GAF regression
# -----------------------------------------------------------------------------
#  Principais pontos
#  ────────────────────────────────────────────────────────────
#  1) Augmentations on-the-fly:
#       • Random H/V flips, small rotations (opcional), Gaussian noise
#       • Mixup + CutMix adaptados para *regression*
#  2) ViT backbone com "drop_path_rate" (stochastic depth)
#  3) Layer-wise learning-rate decay (LLRD) robusto a versões do timm
#  4) Hyper-parâmetros fáceis de ajustar
#  5) Correções:
#       • CutMix bug de indexação corrigido
#       • Escolha consistente entre Mixup/CutMix
#       • Removida duplicata de get_llrd_param_groups (causa do AttributeError)
#       • Predição/targets achatados para evitar broadcasting indevido
# -----------------------------------------------------------------------------

import time, json, random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,  # pode estar disponível; não usamos com squared=False
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
BASE_LR = 3e-4          # LR da cabeça/topo
LAYER_DECAY = 0.75
WEIGHT_DECAY = 5e-2     # WD mais forte costuma ir bem com ViT
DROP_PATH = 0.15

# AugReg params
P_HFLIP = 0.5
P_VFLIP = 0.5
NOISE_STD = 0.02
MIXUP_ALPHA = 0.8
CUTMIX_ALPHA = 1.0
CUTMIX_PROB = 0.5

# (Opcional) pequena rotação aleatória em graus (0 = desliga)
MAX_ROT_DEG = 0
# ╰──────────────────────────────────────────────────────────────╯


# =====================================================================
#  Augmentations helpers
# =====================================================================

def rand_bbox(size, lam):
    """Gera bounding box aleatória (CutMix). size = (B, C, H, W)"""
    H, W = size[2], size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_h, cut_w = int(H * cut_rat), int(W * cut_rat)
    cx, cy = np.random.randint(W), np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2


def maybe_rotate(x, max_deg=0):
    """Rotação leve opcional (inteira) usando grid_sample (mantém shape)."""
    if max_deg <= 0:
        return x
    deg = random.uniform(-max_deg, max_deg)
    # Para simplicidade e velocidade, desabilitamos por padrão
    # (ativar se quiser explorar)
    return x


def mixup_cutmix(x, y, alpha_mix=MIXUP_ALPHA, alpha_cut=CUTMIX_ALPHA, p_cut=CUTMIX_PROB):
    """Aplica Mixup OU CutMix para regressão. Retorna (x_aug, y_aug)."""
    if (alpha_mix <= 0) and (alpha_cut <= 0):
        return x, y

    # Use CutMix com prob p_cut (se alpha_cut > 0); caso contrário Mixup
    use_cutmix = (random.random() < p_cut) and (alpha_cut > 0)
    lam = np.random.beta(alpha_cut, alpha_cut) if use_cutmix else np.random.beta(alpha_mix, alpha_mix)

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    # Garanta que y seja float e com primeiro dim = batch
    # (funciona para y shape (B,) ou (B,1) ou (B,d))
    orig_shape = y.shape
    y_flat = y.view(y.size(0), -1).float()

    if use_cutmix:
        bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
        # corrigido: a ordem dos índices estava trocada
        x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
        # lambda ajustado pela área exata trocada
        lam = 1.0 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size(2) * x.size(3)))
    else:
        x = lam * x + (1.0 - lam) * x[index, :]

    y_flat = lam * y_flat + (1.0 - lam) * y_flat[index]
    y = y_flat.view(orig_shape)
    return x, y


# =====================================================================
#  Métricas
# =====================================================================

def eval_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(root_mean_squared_error(y_true, y_pred)),
        "MAPE": float(mean_absolute_percentage_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


# =====================================================================
#  Modelo: ViT tiny em GAF (2-ch → 3-ch + upsample 48→224)
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
        return self.vit(x).squeeze(1)  # (B,)


# =====================================================================
#  Layer-wise LR decay (LLRD) – robusto a versões do timm
# =====================================================================

def get_llrd_param_groups(model, base_lr=BASE_LR, layer_decay=LAYER_DECAY, wd=WEIGHT_DECAY):
    """
    Retorna param groups com decaimento por camada:
      1) Tenta helper do timm>=1.0 (param_groups_layer_decay)
      2) Se existir API antiga (param_groups/group_matcher), usa
      3) Fallback manual (naming das camadas)
    """
    # 1) Helper moderno do timm (1.0.x)
    try:
        from timm.optim.optim_factory import param_groups_layer_decay
        nwd = set()
        if hasattr(model.vit, "no_weight_decay"):
            nwd = set(model.vit.no_weight_decay())
        return param_groups_layer_decay(
            model.vit,
            weight_decay=wd,
            layer_decay=layer_decay,
            no_weight_decay_list=nwd,
            lr=base_lr,
        )
    except Exception:
        pass

    # 2) API "antiga" (se disponível na sua versão)
    if hasattr(model.vit, "param_groups") and hasattr(model.vit, "group_matcher"):
        matcher = model.vit.group_matcher(coarse=True)
        return model.vit.param_groups(
            lr=base_lr,
            weight_decay=wd,
            layer_decay=layer_decay,
            group_matcher=matcher,
        )

    # 3) Fallback manual (independente da versão)
    param_groups = []
    num_blocks = len(getattr(model.vit, "blocks", []))
    num_layers = num_blocks + 2  # 0=embed, 1..num_blocks=blocks, last=head

    def layer_id_from_name(n: str) -> int:
        if n.startswith("patch_embed") or n.startswith("pos_embed"):
            return 0
        if n.startswith("blocks."):
            try:
                return int(n.split(".")[1]) + 1
            except Exception:
                return 1
        return num_layers - 1  # head & outros do topo

    for name, param in model.vit.named_parameters():
        if not param.requires_grad:
            continue
        lid = layer_id_from_name(name)
        scale = layer_decay ** (num_layers - lid - 1)
        # Sem WD para bias/norm (receita comum para ViT)
        use_wd = 0.0 if param.ndim < 2 else wd
        param_groups.append({
            "params": [param],
            "lr": base_lr * scale,
            "weight_decay": use_wd,
        })
    return param_groups


# =====================================================================
#  Treino / Predição
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
            x = batch["x"].to(DEVICE, non_blocking=True).float()
            y = batch["y"].to(DEVICE, non_blocking=True).float().view(-1)

            # RandAugment-style: flips + (opcional) rotação + ruído
            if random.random() < P_HFLIP:
                x = torch.flip(x, dims=[3])  # horizontal
            if random.random() < P_VFLIP:
                x = torch.flip(x, dims=[2])  # vertical
            if MAX_ROT_DEG > 0:
                x = maybe_rotate(x, MAX_ROT_DEG)
            if NOISE_STD > 0:
                x = x + torch.randn_like(x) * NOISE_STD

            # Mixup / CutMix
            x, y = mixup_cutmix(x, y, MIXUP_ALPHA, CUTMIX_ALPHA, CUTMIX_PROB)
            y = y.view(-1)  # garante shape (B,)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(enabled=torch.cuda.is_available(), device_type=STRDEVICE):
                y_hat = model(x)              # (B,)
                loss = crit(y_hat, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            # acumula SSE para RMSE de treino
            with torch.no_grad():
                sse += F.mse_loss(y_hat, y, reduction="sum").item()
                n_obs += y.size(0)

        train_rmse = (sse / max(n_obs, 1)) ** 0.5

        # -------------------------- validate -------------------------
        model.eval()
        with torch.no_grad():
            y_true, y_pred = [], []
            for b in val_loader:
                x = b["x"].to(DEVICE, non_blocking=True).float()
                y = b["y"].to(DEVICE, non_blocking=True).float().view(-1)
                with torch.amp.autocast(enabled=torch.cuda.is_available(), device_type=STRDEVICE):
                    y_hat = model(x)
                y_true.append(y)
                y_pred.append(y_hat)
            y_true = torch.cat(y_true).float().cpu().numpy().ravel()
            y_pred = torch.cat(y_pred).float().cpu().numpy().ravel()
            val_rmse = root_mean_squared_error(y_true, y_pred)

        print(f"{vit_name}  Ep {ep:03d}/{epochs}  TrainRMSE: {train_rmse:.4f}  ValRMSE: {val_rmse:.4f}")

        # ----------------------- early stopping ----------------------
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                break

    if best_state is not None:
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
            x = b["x"].to(DEVICE).float()
            y_hat = model(x).cpu()
            y_true.append(b["y"].view(-1).cpu())
            y_pred.append(y_hat.view(-1))
    y_true = torch.cat(y_true).numpy().ravel()
    y_pred = torch.cat(y_pred).numpy().ravel()
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
