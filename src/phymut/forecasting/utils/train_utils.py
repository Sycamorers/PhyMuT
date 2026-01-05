# import numpy as np
# import torch
# import torch.nn as nn
# import copy
# import torch.optim as optim
# from sklearn.model_selection import KFold
# from typing import Tuple, Dict, List, Any
# from models.build import build_model
# from torch.utils.data import DataLoader, TensorDataset

# # -----------------------------------------------------------------------------
# # K-fold training & validation with regularization and LR warmup/decay
# # -----------------------------------------------------------------------------
# def train_and_validate_kfold(
#     X_all: np.ndarray,
#     Y_yield_all: np.ndarray,
#     Y_red_all: np.ndarray,
#     forecast_len: int,
#     model_name: str,
#     hid_dim: int,
#     lr: float,
#     num_epochs: int = 100,
#     n_splits: int = 5,
#     num_layers: int = 1,
#     dropout_p: float = 0.2,
#     weight_decay: float = 1e-5,
#     warmup_epochs: int = 5,
#     **kwargs
# ) -> Tuple[float, float]:
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Prepare splits
#     n_samples = X_all.shape[0]
#     n_splits = min(n_splits, n_samples)
#     if n_splits < 2:
#         return float('inf'), float('inf')

#     kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
#     fold_losses_yield, fold_losses_red = [], []

#     for train_idx, val_idx in kf.split(X_all):
#         # Split data
#         X_train, X_val = X_all[train_idx], X_all[val_idx]
#         y_yield_tr, y_yield_val = Y_yield_all[train_idx], Y_yield_all[val_idx]
#         y_red_tr, y_red_val = Y_red_all[train_idx], Y_red_all[val_idx]

#         # Tensors
#         X_tr_t = torch.tensor(X_train, dtype=torch.float32, device=device)
#         X_val_t = torch.tensor(X_val,   dtype=torch.float32, device=device)
#         y_y_tr = torch.tensor(y_yield_tr, dtype=torch.float32, device=device)
#         y_y_val = torch.tensor(y_yield_val, dtype=torch.float32, device=device)
#         y_r_tr = torch.tensor(y_red_tr,   dtype=torch.float32, device=device)
#         y_r_val = torch.tensor(y_red_val, dtype=torch.float32, device=device)

#         # Build models
#         def make_model():
#             if model_name == "nbeats":
#                 return build_model(
#                     model_name=model_name,
#                     hid_dim=hid_dim,
#                     forecast_len=forecast_len,
#                     num_layers=num_layers,
#                     **kwargs
#                 ).to(device)
#             return build_model(
#                 model_name=model_name,
#                 in_dim=X_train.shape[2],
#                 hid_dim=hid_dim,
#                 forecast_len=forecast_len,
#                 num_layers=num_layers,
#                 **kwargs
#             ).to(device)

#         model_yield = make_model()
#         model_red   = make_model()

#         # Regularization layers
#         dropout_layer = nn.Dropout(dropout_p).to(device)

#         # Optimizers with weight decay
#         opt_y = optim.Adam(model_yield.parameters(), lr=lr, weight_decay=weight_decay)
#         opt_r = optim.Adam(model_red.parameters(),   lr=lr, weight_decay=weight_decay)

#         # LR schedulers: linear warmup then step decay
#         sched_y = optim.lr_scheduler.SequentialLR(
#             opt_y,
#             schedulers=[
#                 optim.lr_scheduler.LinearLR(opt_y, start_factor=0.1, total_iters=warmup_epochs),
#                 optim.lr_scheduler.StepLR(opt_y, step_size=num_epochs // 2, gamma=0.5)
#             ],
#             milestones=[warmup_epochs]
#         )
#         sched_r = optim.lr_scheduler.SequentialLR(
#             opt_r,
#             schedulers=[
#                 optim.lr_scheduler.LinearLR(opt_r, start_factor=0.1, total_iters=warmup_epochs),
#                 optim.lr_scheduler.StepLR(opt_r, step_size=num_epochs // 2, gamma=0.5)
#             ],
#             milestones=[warmup_epochs]
#         )

#         criterion = nn.SmoothL1Loss()

#         # Training loop
#         for epoch in range(1, num_epochs + 1):
#             # Yield model
#             model_yield.train()
#             opt_y.zero_grad()
#             y_pred = model_yield(dropout_layer(X_tr_t))
#             loss_y = criterion(y_pred, y_y_tr)
#             loss_y.backward()
#             torch.nn.utils.clip_grad_norm_(model_yield.parameters(), max_norm=1.0)
#             opt_y.step()
#             sched_y.step()

#             # Red model
#             model_red.train()
#             opt_r.zero_grad()
#             r_pred = model_red(dropout_layer(X_tr_t))
#             loss_r = criterion(r_pred, y_r_tr)
#             loss_r.backward()
#             torch.nn.utils.clip_grad_norm_(model_red.parameters(), max_norm=1.0)
#             opt_r.step()
#             sched_r.step()

#         # Validation
#         model_yield.eval(); model_red.eval()
#         with torch.no_grad():
#             val_loss_y = criterion(model_yield(X_val_t), y_y_val).item()
#             val_loss_r = criterion(model_red(X_val_t),   y_r_val).item()

#         fold_losses_yield.append(val_loss_y)
#         fold_losses_red.append(val_loss_r)

#     return float(np.mean(fold_losses_yield)), float(np.mean(fold_losses_red))

# # -----------------------------------------------------------------------------
# # Grid Search (unchanged)
# # -----------------------------------------------------------------------------
# def run_grid_search(
#     X_all: np.ndarray,
#     Y_yield_all: np.ndarray,
#     Y_red_all: np.ndarray,
#     forecast_len: int,
#     grid: Dict[str, List[Any]],
#     model_name: str,
#     num_epochs: int = 100,
#     n_splits: int = 5,
#     **kwargs
# ) -> Tuple[Dict[str, Any], float]:
#     best_total_loss = float('inf')
#     best_params: Dict[str, Any] = {}

#     for lr in grid.get("lr", []):
#         for hid_dim in grid.get("hid_dim", []):
#             for num_layer in grid.get("num_layers", []):
#                 y_loss, r_loss = train_and_validate_kfold(
#                     X_all, Y_yield_all, Y_red_all,
#                     forecast_len=forecast_len,
#                     model_name=model_name,
#                     hid_dim=hid_dim,
#                     lr=lr,
#                     num_epochs=num_epochs,
#                     n_splits=n_splits,
#                     num_layers=num_layer,
#                     **kwargs
#                 )
#                 total = y_loss + r_loss
#                 print(f"Testing: lr={lr}, hid_dim={hid_dim}, num_layers={num_layer}")
#                 print(f"yield={y_loss:.4f}, red={r_loss:.4f}, total={total:.4f}")
#                 if total < best_total_loss:
#                     best_total_loss = total
#                     best_params = {"lr": lr, "hid_dim": hid_dim, "num_layers": num_layer}
#                     print(f"New best: {best_params} with total loss {best_total_loss:.4f}")

#     return best_params, best_total_loss

# # -----------------------------------------------------------------------------
# # EarlyStopping (unchanged)
# # -----------------------------------------------------------------------------
# class EarlyStopping:
#     def __init__(self, patience=5, min_delta=1e-4):
#         self.patience = patience
#         self.min_delta = min_delta
#         self.best_loss = float('inf')
#         self.counter = 0
#         self.best_state = None

#     def __call__(self, current_loss, model):
#         if current_loss + self.min_delta < self.best_loss:
#             self.best_loss = current_loss
#             self.best_state = copy.deepcopy(model.state_dict())
#             self.counter = 0
#             return False
#         else:
#             self.counter += 1
#             return self.counter >= self.patience

# # -----------------------------------------------------------------------------
# # Final training with extended epochs, regularization, and LR warmup/decay
# # -----------------------------------------------------------------------------
# def train_final_models(
#     X_all: torch.Tensor,
#     Y_yield_all: torch.Tensor,
#     Y_red_all: torch.Tensor,
#     best_params: Dict[str, Any],
#     forecast_len: int,
#     model_name: str,
#     num_epochs_final: int = 100,
#     val_split: float = 0.1,
#     patience: int = 5,
#     dropout_p: float = 0.2,
#     weight_decay: float = 1e-5,
#     warmup_epochs: int = 10,
#     noise_std: float = 0.01,
#     **kwargs
# ):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Convert inputs
#     X = X_all.float().to(device) if isinstance(X_all, torch.Tensor) else torch.tensor(X_all, dtype=torch.float32, device=device)
#     Yy = Y_yield_all.float().to(device) if isinstance(Y_yield_all, torch.Tensor) else torch.tensor(Y_yield_all, dtype=torch.float32, device=device)
#     Yr = Y_red_all.float().to(device)   if isinstance(Y_red_all, torch.Tensor)   else torch.tensor(Y_red_all,   dtype=torch.float32, device=device)

#     # Split train/val
#     N = X.size(0)
#     split = int(N * (1 - val_split))
#     X_tr, X_val = X[:split], X[split:]
#     yY_tr, yY_val = Yy[:split], Yy[split:]
#     yR_tr, yR_val = Yr[:split], Yr[split:]

#     # DataLoaders
#     train_y = DataLoader(TensorDataset(X_tr, yY_tr), batch_size=32, shuffle=True)
#     val_y   = DataLoader(TensorDataset(X_val, yY_val), batch_size=32)
#     train_r = DataLoader(TensorDataset(X_tr, yR_tr), batch_size=32, shuffle=True)
#     val_r   = DataLoader(TensorDataset(X_val, yR_val), batch_size=32)

#     lr        = best_params["lr"]
#     hid_dim   = best_params["hid_dim"]
#     num_layer = best_params["num_layers"]

#     def build_with_regularization():
#         if model_name == "nbeats":
#             return build_model(
#                 model_name=model_name,
#                 hid_dim=hid_dim,
#                 forecast_len=forecast_len,
#                 num_layers=num_layer,
#                 **kwargs
#             ).to(device)
#         return build_model(
#             model_name=model_name,
#             in_dim=X_tr.shape[2],
#             hid_dim=hid_dim,
#             forecast_len=forecast_len,
#             num_layers=num_layer,
#             **kwargs
#         ).to(device)

#     def train_one(loader, val_loader):
#         model = build_with_regularization()
#         opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
#         # warmup then decay
#         sched = optim.lr_scheduler.SequentialLR(
#             opt,
#             schedulers=[
#                 optim.lr_scheduler.LinearLR(opt, start_factor=0.1, total_iters=warmup_epochs),
#                 optim.lr_scheduler.StepLR(opt, step_size=num_epochs_final // 2, gamma=0.5)
#             ],
#             milestones=[warmup_epochs]
#         )
#         stopper = EarlyStopping(patience=patience)
#         dropout_layer = nn.Dropout(dropout_p).to(device)

#         for epoch in range(1, num_epochs_final + 1):
#             model.train()
#             for Xb, yb in loader:
#                 Xb, yb = Xb.to(device), yb.to(device)
#                 xb = dropout_layer(Xb + torch.randn_like(Xb) * noise_std)
#                 opt.zero_grad()
#                 pred = model(xb)
#                 loss = nn.MSELoss()(pred, yb)
#                 loss.backward()
#                 nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#                 opt.step()
#                 sched.step()

#             # Validation
#             model.eval()
#             val_losses = []
#             with torch.no_grad():
#                 for Xv, yv in val_loader:
#                     Xv, yv = Xv.to(device), yv.to(device)
#                     val_losses.append(nn.MSELoss()(model(Xv), yv).item())
#             val_loss = sum(val_losses) / len(val_losses)
#             if stopper(val_loss, model):
#                 print(f"Early stopping at epoch {epoch}")
#                 break
#             if epoch % 10 == 0:
#                 print(f"Epoch {epoch}/{num_epochs_final} \t val_loss={val_loss:.4f}")
#         model.load_state_dict(stopper.best_state)
#         return model

#     model_yield = train_one(train_y, val_y)
#     model_red   = train_one(train_r, val_r)

#     return model_yield, model_red



import numpy as np
import torch
import torch.nn as nn
import copy
import torch.optim as optim
from sklearn.model_selection import KFold
from typing import Tuple, Dict, List, Any
from ..models.build import build_model
from torch.utils.data import DataLoader, TensorDataset

# -----------------------------------------------------------------------------
# K-fold training & validation with regularization and LR warmup/decay
# -----------------------------------------------------------------------------
def train_and_validate_kfold(
    X_all: np.ndarray,
    Y_yield_all: np.ndarray,
    Y_red_all: np.ndarray,
    forecast_len: int,
    model_name: str,
    hid_dim: int,
    lr: float,
    num_epochs: int = 100,
    n_splits: int = 5,
    num_layers: int = 1,
    dropout_p: float = 0.2,
    weight_decay: float = 1e-5,
    warmup_epochs: int = 5,
    patience: int = 5,
    noise_std: float = 0.01,
    **kwargs
) -> Tuple[float, float]:
    """
    Original CV loop for grid search: trains separate yield and red models on each fold and returns mean losses.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_samples = X_all.shape[0]
    n_splits = min(n_splits, n_samples)
    if n_splits < 2:
        return float('inf'), float('inf')

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    losses_y, losses_r = [], []

    for train_idx, val_idx in kf.split(X_all):
        X_tr, X_val = X_all[train_idx], X_all[val_idx]
        yY_tr, yY_val = Y_yield_all[train_idx], Y_yield_all[val_idx]
        yR_tr, yR_val = Y_red_all[train_idx], Y_red_all[val_idx]

        X_tr_t = torch.tensor(X_tr, dtype=torch.float32, device=device)
        X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
        yY_tr_t = torch.tensor(yY_tr, dtype=torch.float32, device=device)
        yY_val_t = torch.tensor(yY_val, dtype=torch.float32, device=device)
        yR_tr_t = torch.tensor(yR_tr, dtype=torch.float32, device=device)
        yR_val_t = torch.tensor(yR_val, dtype=torch.float32, device=device)

        # Build models
        def make_model():
            if model_name == "nbeats":
                return build_model(
                    model_name=model_name,
                    hid_dim=hid_dim,
                    forecast_len=forecast_len,
                    num_layers=num_layers,
                    **kwargs
                ).to(device)
            return build_model(
                model_name=model_name,
                in_dim=X_tr.shape[2],
                hid_dim=hid_dim,
                forecast_len=forecast_len,
                num_layers=num_layers,
                **kwargs
            ).to(device)

        model_y = make_model()
        model_r = make_model()

        criterion = nn.SmoothL1Loss()
        opt_y = optim.Adam(model_y.parameters(), lr=lr, weight_decay=weight_decay)
        opt_r = optim.Adam(model_r.parameters(), lr=lr, weight_decay=weight_decay)

        sched_y = optim.lr_scheduler.SequentialLR(
            opt_y,
            schedulers=[
                optim.lr_scheduler.LinearLR(opt_y, start_factor=0.1, total_iters=warmup_epochs),
                optim.lr_scheduler.StepLR(opt_y, step_size=num_epochs//2, gamma=0.5)
            ],
            milestones=[warmup_epochs]
        )
        sched_r = optim.lr_scheduler.SequentialLR(
            opt_r,
            schedulers=[
                optim.lr_scheduler.LinearLR(opt_r, start_factor=0.1, total_iters=warmup_epochs),
                optim.lr_scheduler.StepLR(opt_r, step_size=num_epochs//2, gamma=0.5)
            ],
            milestones=[warmup_epochs]
        )

        dropout = nn.Dropout(dropout_p)

        # Train
        for _ in range(num_epochs):
            # yield model training
            model_y.train()
            xb = dropout(X_tr_t + torch.randn_like(X_tr_t) * noise_std)
            opt_y.zero_grad()
            loss_y = criterion(model_y(xb), yY_tr_t)
            loss_y.backward()
            nn.utils.clip_grad_norm_(model_y.parameters(), 1.0)
            opt_y.step()
            sched_y.step()

            # red model training
            model_r.train()
            xb = dropout(X_tr_t + torch.randn_like(X_tr_t) * noise_std)
            opt_r.zero_grad()
            loss_r = criterion(model_r(xb), yR_tr_t)
            loss_r.backward()
            nn.utils.clip_grad_norm_(model_r.parameters(), 1.0)
            opt_r.step()
            sched_r.step()

        # Validate
        model_y.eval(); model_r.eval()
        with torch.no_grad():
            val_y = criterion(model_y(X_val_t), yY_val_t).item()
            val_r = criterion(model_r(X_val_t), yR_val_t).item()

        losses_y.append(val_y)
        losses_r.append(val_r)

    return float(np.mean(losses_y)), float(np.mean(losses_r))

# -----------------------------------------------------------------------------
# Hyperparameter search (unchanged)
# -----------------------------------------------------------------------------
def run_grid_search(
    X_all: np.ndarray,
    Y_yield_all: np.ndarray,
    Y_red_all: np.ndarray,
    forecast_len: int,
    grid: Dict[str, List[Any]],
    model_name: str,
    num_epochs: int = 100,
    n_splits: int = 5,
    **kwargs
) -> Tuple[Dict[str, Any], float]:
    best_total, best_params = float('inf'), {}
    for lr in grid.get("lr", []):
        for hid_dim in grid.get("hid_dim", []):
            for num_layers in grid.get("num_layers", []):
                y_loss, r_loss = train_and_validate_kfold(
                    X_all,
                    Y_yield_all,
                    Y_red_all,
                    forecast_len,
                    model_name,
                    hid_dim,
                    lr,
                    num_epochs=num_epochs,
                    n_splits=n_splits,
                    num_layers=num_layers,
                    **kwargs
                )
                total = y_loss + r_loss
                print(f"lr={lr}, hid={hid_dim}, layers={num_layers} -> y={y_loss:.4f}, r={r_loss:.4f}")
                if total < best_total:
                    best_total = total
                    best_params = {"lr": lr, "hid_dim": hid_dim, "num_layers": num_layers}
    return best_params, best_total

# -----------------------------------------------------------------------------
# Early stopping helper (unchanged)
# -----------------------------------------------------------------------------
class EarlyStopping:
    def __init__(self, patience=5, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.best_state = None

    def __call__(self, current_loss, model):
        if current_loss + self.min_delta < self.best_loss:
            self.best_loss = current_loss
            self.best_state = copy.deepcopy(model.state_dict())
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience

# -----------------------------------------------------------------------------
# Final training with embedded K-fold CV (signature unchanged)
# -----------------------------------------------------------------------------
def train_final_models(
    X_all: torch.Tensor,
    Y_yield_all: torch.Tensor,
    Y_red_all: torch.Tensor,
    best_params: dict,
    forecast_len: int,
    model_name: str,
    num_epochs_final: int = 50,
    val_split: float = 0.1,
    patience: int = 5,
    noise_std: float = 0.01,
    **kwargs
) -> Tuple[Any, Any]:
    """
    Performs K-fold cross-validation on the full dataset,
    selecting the best-performing model instance for both
    yield and red targets, and returns those two models.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare tensors on device
    X = X_all.to(device) if isinstance(X_all, torch.Tensor) else torch.tensor(X_all, dtype=torch.float32, device=device)
    Yy = Y_yield_all.to(device) if isinstance(Y_yield_all, torch.Tensor) else torch.tensor(Y_yield_all, dtype=torch.float32, device=device)
    Yr = Y_red_all.to(device) if isinstance(Y_red_all, torch.Tensor) else torch.tensor(Y_red_all, dtype=torch.float32, device=device)

    # K-fold
    n_samples = X.size(0)
    n_splits = min(5, n_samples)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    best_y_loss = float('inf')
    best_r_loss = float('inf')
    best_model_y, best_model_r = None, None

    for train_idx, val_idx in kf.split(X):
        X_tr, X_val = X[train_idx], X[val_idx]
        yY_tr, yY_val = Yy[train_idx], Yy[val_idx]
        yR_tr, yR_val = Yr[train_idx], Yr[val_idx]

        # DataLoaders
        train_loader_y = DataLoader(TensorDataset(X_tr, yY_tr), batch_size=32, shuffle=True)
        val_loader_y   = DataLoader(TensorDataset(X_val, yY_val), batch_size=32)
        train_loader_r = DataLoader(TensorDataset(X_tr, yR_tr), batch_size=32, shuffle=True)
        val_loader_r   = DataLoader(TensorDataset(X_val, yR_val), batch_size=32)

        # Build fresh models
        if model_name == "nbeats":
            model_y = build_model(
                model_name=model_name,
                hid_dim=best_params['hid_dim'],
                forecast_len=forecast_len,
                num_layers=best_params['num_layers'],
                **kwargs
            ).to(device)
            model_r = build_model(
                model_name=model_name,
                hid_dim=best_params['hid_dim'],
                forecast_len=forecast_len,
                num_layers=best_params['num_layers'],
                **kwargs
            ).to(device)
        else:
            model_y = build_model(
                model_name=model_name,
                in_dim=X.shape[2],
                hid_dim=best_params['hid_dim'],
                forecast_len=forecast_len,
                num_layers=best_params['num_layers'],
                **kwargs
            ).to(device)
            model_r = build_model(
                model_name=model_name,
                in_dim=X.shape[2],
                hid_dim=best_params['hid_dim'],
                forecast_len=forecast_len,
                num_layers=best_params['num_layers'],
                **kwargs
            ).to(device)

        # Opts and schedulers
        # opt_y = optim.Adam(model_y.parameters(), lr=best_params['lr'], weight_decay=best_params.get('weight_decay', 1e-5))
        # opt_r = optim.Adam(model_r.parameters(), lr=best_params['lr'], weight_decay=best_params.get('weight_decay', 1e-5))
        # used AdamW
        opt_r = optim.AdamW(model_r.parameters(), lr=best_params['lr'], weight_decay=best_params.get('weight_decay', 1e-5))
        opt_y = optim.AdamW(model_y.parameters(), lr=best_params['lr'], weight_decay=best_params.get('weight_decay', 1e-5))
        
        sched_y = optim.lr_scheduler.SequentialLR(
            opt_y,
            schedulers=[
                optim.lr_scheduler.LinearLR(opt_y, start_factor=0.1, total_iters=best_params.get('warmup_epochs', 5)),
                optim.lr_scheduler.StepLR(opt_y, step_size=num_epochs_final//2, gamma=0.5)
            ],
            milestones=[best_params.get('warmup_epochs', 5)]
        )
        sched_r = optim.lr_scheduler.SequentialLR(
            opt_r,
            schedulers=[
                optim.lr_scheduler.LinearLR(opt_r, start_factor=0.1, total_iters=best_params.get('warmup_epochs', 5)),
                optim.lr_scheduler.StepLR(opt_r, step_size=num_epochs_final//2, gamma=0.5)
            ],
            milestones=[best_params.get('warmup_epochs', 5)]
        )

        dropout = nn.Dropout(best_params.get('dropout_p', 0.2)).to(device)
        
        # criterion = nn.MSELoss()
        criterion = nn.SmoothL1Loss()

        stopper_y = EarlyStopping(patience=patience)
        stopper_r = EarlyStopping(patience=patience)

        # Train & validate per fold
        for epoch in range(1, num_epochs_final+1):
            # yield training
            model_y.train()
            for Xb, yb in train_loader_y:
                Xb, yb = Xb.to(device), yb.to(device)
                xb = dropout(Xb + torch.randn_like(Xb) * noise_std)
                opt_y.zero_grad()
                loss = criterion(model_y(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model_y.parameters(), 1.0)
                opt_y.step()
                sched_y.step()
            # red training
            model_r.train()
            for Xb, yb in train_loader_r:
                Xb, yb = Xb.to(device), yb.to(device)
                xb = dropout(Xb + torch.randn_like(Xb) * noise_std)
                opt_r.zero_grad()
                loss = criterion(model_r(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model_r.parameters(), 1.0)
                opt_r.step()
                sched_r.step()

            # validation
            model_y.eval(); model_r.eval()
            with torch.no_grad():
                val_y_losses = [criterion(model_y(Xv.to(device)), yv.to(device)).item() for Xv, yv in val_loader_y]
                val_r_losses = [criterion(model_r(Xv.to(device)), yr.to(device)).item() for Xv, yr in val_loader_r]
            val_y = float(np.mean(val_y_losses))
            val_r = float(np.mean(val_r_losses))
            # early stop both
            stop_y = stopper_y(val_y, model_y)
            stop_r = stopper_r(val_r, model_r)
            if stop_y and stop_r:
                break

        # load best states
        model_y.load_state_dict(stopper_y.best_state)
        model_r.load_state_dict(stopper_r.best_state)

        # select best
        if stopper_y.best_loss < best_y_loss:
            best_y_loss = stopper_y.best_loss
            best_model_y = model_y
        if stopper_r.best_loss < best_r_loss:
            best_r_loss = stopper_r.best_loss
            best_model_r = model_r

    print(f"Selected CV models -> yield loss: {best_y_loss:.4f}, red loss: {best_r_loss:.4f}")
    return best_model_y, best_model_r

# NOTE: bugfix for val iteration
# ensure iteration over dataloaders, not over val metrics
# End of script
