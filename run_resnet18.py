import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import time, json
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score, root_mean_squared_error

from data_splits import make_loaders
from gaf import GAFForecastDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS = 50
PATIENCE = 10
WINDOW = 48  # for 48x48 GAF

# ResNet-18 for 2-channel GAF input
class GAFResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet18(weights=None)
        base.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.features = nn.Sequential(*list(base.children())[:-1])  # remove FC
        self.fc = nn.Linear(512, 1)

    def forward(self, x):  # (B, 2, 48, 48)
        x = self.features(x).squeeze(-1).squeeze(-1)  # -> (B, 512)
        return self.fc(x).squeeze(1)

def eval_metrics(y_true, y_pred):
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(root_mean_squared_error(y_true, y_pred, squared=True)),
        "MAPE": float(mean_absolute_percentage_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }

def train_resnet(train_loader, val_loader, epochs=EPOCHS, lr=1e-3):
    model = GAFResNet18().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()

    best_val = 1e9
    patience = PATIENCE
    counter = 0

    for ep in range(epochs):
        model.train()
        for batch in train_loader:
            x = batch["x"].to(DEVICE)
            y = batch["y"].to(DEVICE)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            y_true, y_pred = [], []
            for b in val_loader:
                x = b["x"].to(DEVICE)
                y = b["y"].to(DEVICE)
                y_hat = model(x)
                y_true.append(y)
                y_pred.append(y_hat)

        y_true = torch.cat(y_true).cpu()
        y_pred = torch.cat(y_pred).cpu()
        val_rmse = root_mean_squared_error(y_true, y_pred)

        if val_rmse < best_val:
            best_val = val_rmse
            counter = 0
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            counter += 1
            if counter >= patience:
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
    return torch.cat(y_true).numpy(), torch.cat(y_pred).numpy()

def run_resnet18(data_path="data_gaf.npz", out_json="resnet18_results.json"):
    t0 = time.time()
    train_loader, val_loader, test_loader = make_loaders(
        path=data_path, dataset_kind="gaf", batch_size=BATCH_SIZE)

    model = train_resnet(train_loader, val_loader)
    y_true, y_pred = predict_model(model, test_loader)
    results = eval_metrics(y_true, y_pred)

    with open(out_json, "w") as fp:
        json.dump(results, fp, indent=4)
    print(json.dumps(results, indent=4))
    print(f"⏱  Total runtime: {time.time() - t0:.1f}s")

    return y_true, y_pred

if __name__ == "__main__":
    run_resnet18()
