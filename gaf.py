import numpy as np
import pandas as pd

def _scale_to_unit_interval(x: np.ndarray, eps: float = 1e-12):
    """Min–max scale to [-1, 1]; constant series → zeros."""
    x = np.asarray(x, dtype=float)
    xmin, xmax = np.nanmin(x), np.nanmax(x)
    if abs(xmax - xmin) < eps:
        return np.zeros_like(x)
    x01 = (x - xmin) / (xmax - xmin)
    return np.clip(2.0 * x01 - 1.0, -1.0, 1.0)

def gaf_matrix(x_scaled: np.ndarray, mode: str = "summation") -> np.ndarray:
    """Build a GAF matrix from a series scaled to [-1, 1]."""
    phi = np.arccos(np.clip(x_scaled, -1.0, 1.0))
    if mode == "summation":
        return np.cos(phi[:, None] + phi[None, :])  # GASF
    else:
        return np.sin(phi[:, None] - phi[None, :])  # GADF

def build_gaf(x: np.ndarray, mode: str = "both"):
    """
    Returns:
        (GASF, GADF) if mode='both'
        or a single-channel tuple otherwise
    """
    x = np.asarray(x, dtype=float)
    x = pd.Series(x).interpolate(limit_direction="both").ffill().bfill().to_numpy()
    x_scaled = _scale_to_unit_interval(x)
    if mode == "both":
        return gaf_matrix(x_scaled, "summation"), gaf_matrix(x_scaled, "difference")
    elif mode == "summation":
        return (gaf_matrix(x_scaled, "summation"),)
    else:
        return (gaf_matrix(x_scaled, "difference"),)

def build_forecast_gaf_dataset(series: np.ndarray, window: int, horizon: int = 1, stride: int = 1, mode: str = "both"):
    """
    Build dataset for GAF-based time series forecasting.
    
    Inputs:
        series  : 1D array of time series values (e.g. % Silica)
        window  : number of past steps to use for input (e.g., 24 for 24 hours)
        horizon : prediction offset (usually 1 for next-step forecast)
        stride  : window sliding stride
        mode    : "both", "summation", or "difference"
    
    Returns:
        X       : array of GAF inputs (N, C, window, window)
        y       : array of target values (N,)
        starts  : list of starting indices for each input window
    """
    series = pd.Series(series).interpolate(limit_direction="both").ffill().bfill().to_numpy()
    T = len(series)
    X_list = []
    y_list = []
    starts = []

    for start in range(0, T - window - horizon + 1, stride):
        segment = series[start:start + window]
        target = series[start + window + horizon - 1]  # future target

        gaf = build_gaf(segment, mode=mode)
        if mode == "both":
            x = np.stack(gaf, axis=0)  # (2, window, window)
        else:
            x = np.expand_dims(gaf[0], axis=0)  # (1, window, window)

        X_list.append(x.astype(np.float32))
        y_list.append(np.float32(target))
        starts.append(start)

    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.float32)
    return X, y, starts

# -----------------------
# PyTorch Dataset Wrapper
# -----------------------
try:
    import torch
    from torch.utils.data import Dataset
except ImportError:
    torch = None
    Dataset = object

class GAFForecastDataset(Dataset):
    def __init__(self, X, y, starts=None, transform=None):
        self.X = torch.from_numpy(X) if torch else X
        self.y = torch.from_numpy(y) if torch else y
        self.starts = starts
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        sample = {
            "x": self.X[idx],     # shape: (C, W, W)
            "y": self.y[idx],     # scalar
            "start": self.starts[idx] if self.starts is not None else None
        }
        if self.transform:
            sample = self.transform(sample)
        return sample

# -----------------------
# Example usage
# -----------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Load your CSV
    df = pd.read_csv(r"RSPlayground\data_cleaned.csv")  # Replace with the correct path
    col = "% Silica Concentrate"
    series = pd.to_numeric(df[col], errors='coerce').interpolate(limit_direction="both").ffill().bfill().to_numpy()

    plt.plot(series)
    plt.show()

    # Build dataset
    X, y, starts = build_forecast_gaf_dataset(series, window=48, horizon=1, stride=1, mode="both")
    print("X shape:", X.shape)  
    print("y shape:", y.shape)
    print(f"Example target: {y[0]:.3f} from window starting at t={starts[0]}")


    np.savez_compressed("forecast_dataset.npz", X=X, y=y, starts=starts)
    data = np.load("forecast_dataset.npz")
    X = data["X"]
    y = data["y"]
    starts = data["starts"]


    # Wrap into PyTorch Dataset
    dataset = GAFForecastDataset(X, y, starts)

    # Optional: visualize
    #for i in range(len(dataset)):
    #    sample = dataset[i]
    #    print(sample['y'])
        #if sample['y'] >= 5:
        #    plt.imshow(sample["x"][0], cmap="viridis")
        #    plt.title(f"GASF (target: {sample['y'].item():.3f})")
        #    plt.axis("off")
        #    plt.show()
