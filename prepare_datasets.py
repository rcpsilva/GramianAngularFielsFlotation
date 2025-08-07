"""
Prepare GAF and raw-series forecasting datasets for % Silica.
Outputs:
    data_gaf.npz     – X_gaf (N,2,W,W), y, starts
    data_raw.npz     – X_raw (N,1,W),   y, starts
"""
from pathlib import Path
import numpy as np
import pandas as pd
from gaf import build_forecast_gaf_dataset          # already implemented:contentReference[oaicite:0]{index=0}

# -------------------------
# configurable parameters
# -------------------------
CSV_PATH   = Path(r"C:\Users\rcpsi\OneDrive\Documents\GitHub\flotation_forecast\RSPlayground\GAF\data_cleaned.csv")
TARGET_COL = "% Silica Concentrate"
WINDOW     = 48          # 48 hours
HORIZON    = 1           # one-step-ahead
STRIDE     = 1
TRAIN_VAL_TEST = (0.7, 0.15, 0.15)   # chronological split
SEED = 42
# -------------------------

def build_raw_window_dataset(series, window, horizon=1, stride=1):
    series = pd.Series(series).interpolate(
        limit_direction="both").ffill().bfill().to_numpy()
    X, y, starts = [], [], []
    for s in range(0, len(series) - window - horizon + 1, stride):
        X.append(series[s:s+window].astype(np.float32))
        y.append(np.float32(series[s+window+horizon-1]))
        starts.append(s)
    X = np.expand_dims(np.stack(X, axis=0), 1)   # (N,1,W)
    y = np.asarray(y, dtype=np.float32)
    return X, y, starts

def main():
    df = pd.read_csv(CSV_PATH)
    series = pd.to_numeric(df[TARGET_COL], errors="coerce")

    # --- GAF tiles ---
    Xg, yg, starts = build_forecast_gaf_dataset(
        series, window=WINDOW, horizon=HORIZON, stride=STRIDE, mode="both")
    np.savez_compressed("data_gaf.npz", X=Xg, y=yg, starts=starts)

    # --- raw 1-D windows ---
    Xr, yr, _ = build_raw_window_dataset(
        series, window=WINDOW, horizon=HORIZON, stride=STRIDE)
    np.savez_compressed("data_raw.npz", X=Xr, y=yr, starts=starts)  # same starts!

    print(f"GAF dataset  : {Xg.shape} samples")
    print(f"Raw dataset  : {Xr.shape} samples")

if __name__ == "__main__":
    np.random.seed(SEED)
    main()
