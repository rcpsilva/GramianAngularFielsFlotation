import torch
import torch.nn as nn
import numpy as np
import time, json
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
from data_splits import make_loaders

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS = 50
PATIENCE = 10

# -----------------------------
# Small Custom CNN for GAF
# -----------------------------
class SmallGAFNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.GroupNorm(4, 32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.GroupNorm(4, 64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.GroupNorm(4, 128),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = (x - x.mean(dim=(2, 3), keepdim=True)) / (x.std(dim=(2, 3), keepdim=True) + 1e-6)
        return self.net(x).squeeze(1)

# -----------------------------
# Metrics
# -----------------------------
def eval_metrics(y_true, y_pred):
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(mean_squared_error(y_true, y_pred, squared=False)),
        "MAPE": float(mean_absolute_percentage_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }

# -----------------------------
# Training Function
# -----------------------------
def train_cnn(train_loader, val_loader, epochs=EPOCHS, lr=1e-3):
    model = SmallGAFNet().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.MSELoss()

    best_val = float('inf')
    counter = 0

    for ep in range(epochs):
        model.train()
        train_true, train_pred = [], []
        for batch in train_loader:
            x = batch["x"].to(DEVICE)
            y = batch["y"].to(DEVICE)
            opt.zero_grad()
            output = model(x)
            loss = crit(output, y)
            loss.backward()
            opt.step()
            train_true.append(y.detach().cpu())
            train_pred.append(output.detach().cpu())

        train_true = torch.cat(train_true)
        train_pred = torch.cat(train_pred)
        train_rmse = mean_squared_error(train_true, train_pred, squared=False)

        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for b in val_loader:
                x = b["x"].to(DEVICE)
                y = b["y"].to(DEVICE)
                y_hat = model(x)
                y_true.append(y)
                y_pred.append(y_hat)

        y_true = torch.cat(y_true).cpu()
        y_pred = torch.cat(y_pred).cpu()
        val_rmse = mean_squared_error(y_true, y_pred, squared=False)
        print(f"Epoch {ep} - Train RMSE: {train_rmse:.4f} - Val RMSE: {val_rmse:.4f}")

        if val_rmse < best_val:
            best_val = val_rmse
            counter = 0
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            counter += 1
            if counter >= PATIENCE:
                break

    model.load_state_dict(best_state)
    return model.eval()

# -----------------------------
# Prediction
# -----------------------------
def predict_model(model, loader):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for b in loader:
            x = b["x"].to(DEVICE)
            y_hat = model(x).cpu()
            y_true.append(b["y"])
            y_pred.append(y_hat)
    return torch.cat(y_true).numpy(), torch.cat(y_pred).numpy()

# -----------------------------
# Runner
# -----------------------------
def run_small_cnn(data_path="data_gaf.npz", out_json="smallcnn_results.json"):
    t0 = time.time()
    train_loader, val_loader, test_loader = make_loaders(
        path=data_path, dataset_kind="gaf", batch_size=BATCH_SIZE)

    model = train_cnn(train_loader, val_loader)
    y_true, y_pred = predict_model(model, test_loader)
    results = eval_metrics(y_true, y_pred)

    with open(out_json, "w") as fp:
        json.dump(results, fp, indent=4)
    print(json.dumps(results, indent=4))
    print(f"\u23f1  Total runtime: {time.time() - t0:.1f}s")

    return y_true, y_pred

if __name__ == "__main__":
    run_small_cnn()
