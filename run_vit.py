import time, json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score, root_mean_squared_error

import timm  # pip install timm

from data_splits import make_loaders

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STRDEVICE= "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
EPOCHS = 60
PATIENCE = 10
LR = 3e-4
WEIGHT_DECAY = 1e-4

# ------------------------------
# Metrics
# ------------------------------
def eval_metrics(y_true, y_pred):
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(root_mean_squared_error(y_true, y_pred)),
        "MAPE": float(mean_absolute_percentage_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }

# ------------------------------
# Model: ViT tiny on GAF tiles
#   - 2ch -> (1x1 conv) -> 3ch
#   - 48x48 -> (bilinear) -> 224x224
#   - ViT (timm) with num_classes=1 (regression)
# ------------------------------
class GAFViT(nn.Module):
    def __init__(self, vit_name: str = "vit_tiny_patch16_224", pretrained: bool = True):
        super().__init__()
        self.to3 = nn.Conv2d(2, 3, kernel_size=1, bias=False)
        self.vit = timm.create_model(
            vit_name, pretrained=pretrained, num_classes=1
        )  # outputs (B,1)

    def forward(self, x):  # x: (B,2,48,48)
        x = self.to3(x)
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        return self.vit(x).squeeze(1)

# ------------------------------
# Train / Predict
# ------------------------------
def train_vit(train_loader, val_loader,vit_name:str = "vit_tiny_patch16_224",
              pretrained:bool = True, epochs=EPOCHS, lr=LR, wd=WEIGHT_DECAY):
    model = GAFViT(vit_name=vit_name, pretrained=pretrained).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    crit = nn.MSELoss()

    best_rmse = float("inf")
    best_state = None
    patience_ctr = 0

    scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())

    for ep in range(1, epochs + 1):
        sse, n_obs = 0.0, 0 
        # ---- train ----
        model.train()
        for batch in train_loader:
            x = batch["x"].to(DEVICE, non_blocking=True)
            y = batch["y"].to(DEVICE, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(enabled=torch.cuda.is_available(),device_type=STRDEVICE):
                y_hat = model(x)
                loss = crit(y_hat, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

             # ---- accumulate squared error ----
            with torch.no_grad():                         # no gradients needed
                sse   += F.mse_loss(y_hat, y, reduction='sum').item()
                n_obs += y.size(0)

        train_rmse = (sse / n_obs) ** 0.5  

        # ---- validate ----
        model.eval()
        with torch.no_grad():
            y_true, y_pred = [], []
            for b in val_loader:
                x = b["x"].to(DEVICE)
                y = b["y"].to(DEVICE)
                with torch.amp.autocast(enabled=torch.cuda.is_available(),device_type=STRDEVICE):
                    y_hat = model(x)
                y_true.append(y)
                y_pred.append(y_hat)
            y_true = torch.cat(y_true).float().cpu().numpy()
            y_pred = torch.cat(y_pred).float().cpu().numpy()
            val_rmse = root_mean_squared_error(y_true, y_pred)
            print(f'{vit_name} {epochs}/{ep} TrainRMSE: {train_rmse} ValRMSE: {val_rmse}')

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

# ------------------------------
# Runner
# ------------------------------
def run_vit(epochs:int=20, vit_name:str = "vit_tiny_patch16_224", pretrained:bool = True, data_path="data_gaf.npz", out_json="vit_results.json"):
    t0 = time.time()
    train_loader, val_loader, test_loader = make_loaders(
        path=data_path, dataset_kind="gaf", batch_size=BATCH_SIZE
    )

    model = train_vit(train_loader, val_loader, vit_name,
              pretrained, epochs=epochs, lr=LR, wd=WEIGHT_DECAY)
    y_true, y_pred = predict_model(model, test_loader)
    results = eval_metrics(y_true, y_pred)

    with open(out_json, "w") as fp:
        json.dump(results, fp, indent=4)
    print(json.dumps(results, indent=4))
    print(f"⏱  Total runtime: {time.time() - t0:.1f}s")

    return y_true, y_pred, results

if __name__ == "__main__":
    run_vit()
