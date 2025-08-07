#!/usr/bin/env python
"""
run_baselines.py
Train classical & lightweight-DL baselines for %-silica forecasting.

Assumes:
    ▸ data_raw.npz       – produced by prepare_datasets.py
    ▸ data_splits.py     – make_loaders() already implemented

Baselines implemented here
    0) “Last value”              (no training)
    1) 48-step moving average    (no training)
    2) LightGBM regressor        (lag-features)
    3) 1-D CNN (very small)

Metrics reported
    MAE · RMSE · MAPE · R²
"""
# ──────────────────────────────────────────────────────────────
import argparse, json, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
import lightgbm as lgb

from data_splits import make_loaders   # Step 2 util

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 256
WINDOW = 48          # keep in sync with prepare_datasets.py
# ──────────────────────────────────────────────────────────────
# keep all imports as-is …

def eval_metrics(y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    # ← Cast to native Python float
    return {k: float(v) for k, v in
            dict(MAE=mae, RMSE=rmse, MAPE=mape, R2=r2).items()}


# ╭─────────────────────────────────────────────────────────────╮
# │ 0 & 1 – Naïve baselines (no training)                      │
# ╰─────────────────────────────────────────────────────────────╯
def predict_last_value(loader):
    y_hat, y_true = [], []
    for batch in loader:
        X = batch["x"]          # (B,1,W)
        y_hat.append(X[:, 0, -1])  # last observed value
        y_true.append(batch["y"])
    return torch.cat(y_true).numpy(), torch.cat(y_hat).numpy()

def predict_mavg(loader):
    y_hat, y_true = [], []
    for batch in loader:
        X = batch["x"]
        y_hat.append(X.mean(dim=2).squeeze(1))
        y_true.append(batch["y"])
    return torch.cat(y_true).numpy(), torch.cat(y_hat).numpy()

# ╭─────────────────────────────────────────────────────────────╮
# │ 2 – LightGBM regressor                                     │
# ╰─────────────────────────────────────────────────────────────╯
# ──────────────────────────────────────────────────────────────
# 2) LightGBM (robust to old versions)
# ──────────────────────────────────────────────────────────────
from lightgbm import LGBMRegressor          # instead of lightgbm.train()

def train_lgb(train_loader, val_loader):
    X_tr, y_tr = _flatten(train_loader)
    X_va, y_va = _flatten(val_loader)

    model = LGBMRegressor(
        objective="regression",
        n_estimators=5000,       # high upper-bound; early-stop will cut it
        learning_rate=0.05,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="rmse",
        early_stopping_rounds=50,
        verbose=False,
    )
    return model            # predict() works the same way

def predict_lgb(model, loader):
    X_te, y_te = _flatten(loader)
    return y_te, model.predict(X_te, num_iteration=model.best_iteration_)


# ╭─────────────────────────────────────────────────────────────╮
# │ 3 – Tiny 1-D CNN                                           │
# ╰─────────────────────────────────────────────────────────────╯
class TinyCNN(nn.Module):
    """3-layer causal 1-D CNN with global average pool."""
    def __init__(self, in_len=WINDOW):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, 5, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 32, 5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 32, 5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        # x: (B,1,W)
        y = self.net(x).squeeze(-1)
        return self.fc(y).squeeze(1)

def train_cnn(train_loader, val_loader, epochs=50, lr=1e-3):
    model  = TinyCNN().to(DEVICE)
    opt    = torch.optim.Adam(model.parameters(), lr=lr)
    crit   = nn.MSELoss()

    best_val = 1e9; patience, counter = 10, 0
    for ep in range(epochs):
        # ---- train ----
        model.train()
        for batch in train_loader:
            x, y = batch["x"].to(DEVICE), batch["y"].to(DEVICE)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
        # ---- val ----
        model.eval()
        with torch.no_grad():
            y_hat, y_true = [], []
            for b in val_loader:
                x, y = b["x"].to(DEVICE), b["y"].to(DEVICE)
                y_hat.append(model(x))
                y_true.append(y)
        val_rmse = mean_squared_error(
            torch.cat(y_true).cpu(), torch.cat(y_hat).cpu(), squared=False)
        if val_rmse < best_val:
            best_val, counter = val_rmse, 0
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            counter += 1
            if counter >= patience:
                break
    model.load_state_dict(best_state)
    return model.eval()

def predict_cnn(model, loader):
    y_hat, y_true = [], []
    with torch.no_grad():
        for b in loader:
            x = b["x"].to(DEVICE)
            y_hat.append(model(x).cpu())
            y_true.append(b["y"])
    return torch.cat(y_true).numpy(), torch.cat(y_hat).numpy()

# ──────────────────────────────────────────────────────────────
# NEW - Random-Forest & XGBoost regressors
# ──────────────────────────────────────────────────────────────
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb          # pip install xgboost

def _flatten(loader):
    """Utility: (DataLoader) ➜ X np.ndarray (N,W), y np.ndarray (N,)"""
    Xs, ys = [], []
    for b in loader:
        Xs.append(b["x"].squeeze(1).numpy())
        ys.append(b["y"].numpy())
    return np.vstack(Xs), np.hstack(ys)

# 4) Random-Forest
def train_rf(train_loader):
    X_tr, y_tr = _flatten(train_loader)
    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42,
    ).fit(X_tr, y_tr)
    return model

def predict_rf(model, loader):
    X_te, y_te = _flatten(loader)
    return y_te, model.predict(X_te)

# 5) XGBoost
# ──────────────────────────────────────────────────────────────
# 5) XGBoost (API ≥ 2.0 compatible)
# ──────────────────────────────────────────────────────────────
import xgboost as xgb

def train_xgb(train_loader, val_loader):
    X_tr, y_tr = _flatten(train_loader)
    X_va, y_va = _flatten(val_loader)

    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=2000,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        eval_metric="rmse",          # ← now in the constructor
        early_stopping_rounds=100,   # ← also moved here
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        verbose=False,
    )
    return model

def predict_xgb(model, loader):
    X_te, y_te = _flatten(loader)
    return y_te, model.predict(X_te, iteration_range=(0, model.best_iteration + 1))


# ──────────────────────────────────────────────────────────────
# MAIN – extend run_all() with the two new baselines
# ──────────────────────────────────────────────────────────────
def run_all(data_path="data_raw.npz", out_json="baseline_results.json"):
    train_loader, val_loader, test_loader = make_loaders(
        path=data_path, dataset_kind="raw", batch_size=BATCH_SIZE)

    t0 = time.time()
    results = {}

    # 0) Last value
    y, yhat = predict_last_value(test_loader)
    results["LastValue"] = eval_metrics(y, yhat)

    # 1) Moving average
    y, yhat = predict_mavg(test_loader)
    results["MovingAvg48"] = eval_metrics(y, yhat)

    # 2) LightGBM
    #booster = train_lgb(train_loader, val_loader)
    #y, yhat = predict_lgb(booster, test_loader)
    #results["LightGBM"] = eval_metrics(y, yhat)

    # 3) Tiny CNN
    cnn = train_cnn(train_loader, val_loader)
    y, yhat = predict_cnn(cnn, test_loader)
    results["TinyCNN"] = eval_metrics(y, yhat)

    # 4) Random-Forest
    rf = train_rf(train_loader)
    y, yhat = predict_rf(rf, test_loader)
    results["RandomForest"] = eval_metrics(y, yhat)

    # 5) XGBoost
    xgb_model = train_xgb(train_loader, val_loader)
    y, yhat = predict_xgb(xgb_model, test_loader)
    results["XGBoost"] = eval_metrics(y, yhat)

    with open(out_json, "w") as fp:
        json.dump(results, fp, indent=4)
    print(json.dumps(results, indent=4))
    print(f"⏱  Total runtime: {time.time()-t0:.1f}s")

# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data_raw.npz")
    parser.add_argument("--out",  default="baseline_results.json")
    args = parser.parse_args()
    run_all(args.data, args.out)

#python run_baselines.py --data data_raw.npz --out baseline_results.json
