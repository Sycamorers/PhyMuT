# #!/usr/bin/env python
# """
# batch_weather_encoder_multi.py
# Train a self‐supervised CNN weather encoder to predict all targets jointly.
# All outputs land under ./weather_allout/<season>/:
#     - model.pth
#     - embed.npy
#     - loss_curve.npy
#     - loss_curve.png
# """

# import os, re
# from pathlib import Path
# from datetime import datetime, timedelta, date
# from typing import List

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader, TensorDataset

# # ───────────────────────── Batch grid ─────────────────────────
# SEASONS     = ["2324", "2425"]  # add/remove seasons here
# FEATURE_COLS = [
#     "Soil Temp (C)", "Temp @ 60cm (C)", "Relative Humidity (%)",
#     "Dew Point Temp (C)", "Rainfall Amount (in)",
#     "Wind Speed (mph)", "Wind Direction (deg)", "Solar Radiation (w/m2)"
# ]
# F = len(FEATURE_COLS)
# STEP, SEQ_LEN = "15min", 7 * 24 * 4   # 672 steps per week

# # ───────────────── Hyper‐params (shared) ────────────────────
# BATCH, EPOCHS, LR, DATE_FMT = 32, 2000, 3e-4, "%y%m%d"
# DATA_ROOT = Path("../yield_data")    # edit if paths differ
# OUT_ROOT  = Path("weather_allout")   # new output root

# # ═══════════════════ helper functions ════════════════════════
# def collect_week_dates(yield_dir: Path) -> List[date]:
#     days = []
#     for p in sorted(yield_dir.iterdir()):
#         if p.is_dir() and re.fullmatch(r"\d{6}", p.name):
#             try:
#                 days.append(datetime.strptime(p.name, DATE_FMT).date())
#             except ValueError:
#                 pass
#     return days

# def build_weather_df(csv_path: Path) -> pd.DataFrame:
#     df = pd.read_csv(csv_path)
#     df["Date Time"] = pd.to_datetime(df["Date Time"])
#     return df.set_index("Date Time").sort_index()

# def week_tensor(end_date, wdf) -> np.ndarray:
#     start = end_date - timedelta(days=7)
#     idx   = pd.date_range(start, end_date, freq=STEP, inclusive="left")
#     block = (wdf.reindex(idx)[FEATURE_COLS]
#              .ffill().bfill()
#              .interpolate(limit_direction="both"))
#     if block.shape[0] != SEQ_LEN:
#         block = block.reindex(idx).interpolate(limit_direction="both")
#     out = block.values.astype(np.float32)
#     assert out.shape == (SEQ_LEN, F)
#     return out

# # ═══════════════════════ Model ══════════════════════════════
# class WeatherCNN(nn.Module):
#     def __init__(self, in_feat=F, n_ch=16, ks=(6, 24, 96)):
#         super().__init__()
#         self.branches = nn.ModuleList([
#             nn.Sequential(
#                 nn.Conv1d(in_feat, n_ch, k, padding=k//2),
#                 nn.ReLU(),
#                 nn.AdaptiveAvgPool1d(1)
#             )
#             for k in ks
#         ])
#         # project to F outputs (one per feature)
#         self.proj = nn.Linear(len(ks)*n_ch, F*8)

#     def forward(self, x):
#         # x: (batch, time, feat) -> (batch, feat, time)
#         x = x.permute(0, 2, 1)
#         outs = [b(x).squeeze(-1) for b in self.branches]  # each (batch, n_ch)
#         h = torch.cat(outs, dim=1)                       # (batch, n_ch*len(ks))
#         return self.proj(h)                              # (batch, F)

# # ═════════════════════ Training ════════════════════════════
# def train_encoder(x_weather: np.ndarray):
#     """
#     x_weather: (N, SEQ_LEN, F)
#     Returns trained model and loss history.
#     """
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     net = WeatherCNN().to(device)
#     opt = torch.optim.Adam(net.parameters(), lr=LR)
#     loss_fn = nn.MSELoss()  # averages over all output dims
#     ds = TensorDataset(torch.tensor(x_weather))
#     dl = DataLoader(ds, batch_size=BATCH, shuffle=True)
#     losses = np.zeros(EPOCHS, np.float32)

#     for ep in range(EPOCHS):
#         net.train()
#         total_loss = 0.0
#         for (seq,) in dl:
#             seq = seq.to(device)  # (batch, 672, F)
#             x_in = seq[:, :-4, :]              # remove last 4 steps
#             tgt  = seq[:, -4:, :].mean(dim=1)  # (batch, F)
#             pred = net(x_in)                   # (batch, F)
#             loss = loss_fn(pred, tgt)
#             opt.zero_grad()
#             loss.backward()
#             opt.step()
#             total_loss += loss.item() * seq.size(0)
#         losses[ep] = total_loss / len(ds)

#     net.eval()
#     return net.cpu(), losses

# # ═══════════════════════ Batch loop ════════════════════════
# for season in SEASONS:
#     yield_dir   = DATA_ROOT / f"{season}_GNV_processed"
#     weather_csv = yield_dir / f"{season}_weather.csv"

#     # prepare data
#     week_dates    = collect_week_dates(yield_dir)
#     wdf           = build_weather_df(weather_csv)
#     week_tensors  = [week_tensor(pd.Timestamp(d), wdf) for d in week_dates]
#     x_weather     = np.stack(week_tensors)  # (N, 672, F)

#     # train once for all targets
#     out_dir = OUT_ROOT / season
#     out_dir.mkdir(parents=True, exist_ok=True)

#     print(f"\n▶ Training Season {season}  Targets: all {F} features")
#     net, losses = train_encoder(x_weather)

#     # save model & outputs
#     torch.save(net.state_dict(), out_dir/"model.pth")
#     with torch.no_grad():
#         embeds = net(torch.tensor(x_weather)).numpy()  # (N, F)
#     np.save(out_dir/"embed.npy", embeds)
#     np.save(out_dir/"loss_curve.npy", losses)

#     # plot training loss
#     plt.figure()
#     plt.plot(losses)
#     plt.title(f"{season} – all_targets")
#     plt.xlabel("Epoch")
#     plt.ylabel("MSE")
#     plt.tight_layout()
#     plt.savefig(out_dir/"loss_curve.png", dpi=300)
#     plt.close()

#     print(f"✓ Saved to {out_dir}")

# print("\nAll runs complete. Results are under ./weather_allout/") 


#!/usr/bin/env python
"""
batch_weather_encoder_multi.py
Train a self‐supervised CNN weather encoder to predict all targets jointly,
but retain a 64-dim embedding from the penultimate layer.

All outputs land under data/weather_allout/<season>/:
    - model.pth
    - embed.npy      # shape (N_weeks, 64)
    - loss_curve.npy
    - loss_curve.png
"""

import re
from pathlib import Path
from datetime import datetime, timedelta, date
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from phymut.paths import data_dir
# ───────────────────────── Batch grid ─────────────────────────
SEASONS      = ["2324", "2425"]  # add/remove seasons here
FEATURE_COLS = [
    "Temp @ 60cm (C)",
    "Humidity @ 60cm (%)",
    "SR @ 60cm (W/m²)",
    "Soil Temp @ 5cm (C)",
    "Precipitation (mm)",
    "Wind Speed @ 60cm (m/s)",
    "Wind Direction @ 60cm (°)",
    "Temp @ 250cm (C)",
]
F         = len(FEATURE_COLS)
STEP      = "15min"
SEQ_LEN   = 7 * 24 * 4   # 672 steps / week
EMB_DIM   = 64          # desired embedding size

# ───────────────── Hyper‐params (shared) ────────────────────
BATCH     = 32
EPOCHS    = 1000
LR        = 3e-4
DATE_FMT  = "%y%m%d"
DATA_ROOT = data_dir()
OUT_ROOT = data_dir("weather_allout")

# ═══════════════════ helper functions ════════════════════════
def collect_week_dates(yield_dir: Path) -> List[date]:
    days = []
    for p in sorted(yield_dir.iterdir()):
        if p.is_dir() and re.fullmatch(r"\d{6}", p.name):
            try:
                days.append(datetime.strptime(p.name, DATE_FMT).date())
            except ValueError:
                pass
    return days

def build_weather_df(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["Date Time"] = pd.to_datetime(df["Date Time"])
    return df.set_index("Date Time").sort_index()

def week_tensor(end_date: date, wdf: pd.DataFrame) -> np.ndarray:
    start = end_date - timedelta(days=7)
    idx   = pd.date_range(start, end_date, freq=STEP, inclusive="left")
    block = wdf.reindex(idx)[FEATURE_COLS].ffill().bfill().interpolate(limit_direction="both")
    if block.shape[0] != SEQ_LEN:
        block = block.reindex(idx).interpolate(limit_direction="both")
    out = block.values.astype(np.float32)
    assert out.shape == (SEQ_LEN, F)
    return out

# ═══════════════════════ Model ══════════════════════════════
class WeatherCNN(nn.Module):
    def __init__(self, in_feat: int = F, n_ch: int = 16, ks=(6, 24, 96)):
        super().__init__()
        # multi-scale conv branches
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_feat, n_ch, k, padding=k//2),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1)
            )
            for k in ks
        ])
        # penultimate embedding layer
        self.proj64   = nn.Linear(len(ks)*n_ch, EMB_DIM)
        # final prediction head (F targets)
        self.head     = nn.Linear(EMB_DIM, F)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, time, feat) -> (batch, feat, time)
        x = x.permute(0, 2, 1)
        # collect branches
        outs = [b(x).squeeze(-1) for b in self.branches]  # each (batch, n_ch)
        h   = torch.cat(outs, dim=1)                     # (batch, n_ch*len(ks))
        z   = self.proj64(h)                             # (batch, EMB_DIM)
        return self.head(z), z                           # returns (pred, embed)

# ═════════════════════ Training ════════════════════════════
def train_encoder(x_weather: np.ndarray):
    """
    Train WeatherCNN on x_weather and return the trained model plus loss history.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net    = WeatherCNN().to(device)
    opt    = torch.optim.Adam(net.parameters(), lr=LR)
    loss_fn= nn.MSELoss()  # averages over output dims

    ds = TensorDataset(torch.tensor(x_weather))
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True)
    losses = np.zeros(EPOCHS, dtype=np.float32)

    for ep in range(EPOCHS):
        net.train()
        total = 0.0
        for (seq,) in dl:
            seq = seq.to(device)                     # (batch, 672, F)
            x_in= seq[:, :-4, :]                     # drop last 4
            tgt = seq[:, -4:, :].mean(dim=1)         # (batch, F)
            pred, _ = net(x_in)                      # (batch, F), (batch, EMB_DIM)
            loss = loss_fn(pred, tgt)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * seq.size(0)
        losses[ep] = total / len(ds)

    net.eval()
    return net.cpu(), losses

# ═══════════════════════ Batch loop ════════════════════════
if __name__ == "__main__":
    for season in SEASONS:
        yield_dir   = DATA_ROOT / f"{season}_GNV_processed"
        weather_csv = yield_dir / f"{season}_weather.csv"

        # prepare data
        dates        = collect_week_dates(yield_dir)
        wdf          = build_weather_df(weather_csv)
        tensors      = [week_tensor(d, wdf) for d in dates]
        x_weather    = np.stack(tensors)  # (N_weeks, 672, F)

        # train
        out_dir = OUT_ROOT / season
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n▶ Training season {season} on all {F} features")
        model, losses = train_encoder(x_weather)

        # save weights
        torch.save(model.state_dict(), out_dir / "model.pth")

        # compute & save embeddings (second-last layer output)
        with torch.no_grad():
            _, embeds = model(torch.tensor(x_weather))
            embeds = embeds.numpy()  # (N_weeks, EMB_DIM)
        np.save(out_dir / "embed.npy", embeds)

        # save loss curve
        np.save(out_dir / "loss_curve.npy", losses)
        plt.figure()
        plt.plot(losses)
        plt.title(f"{season} – embed dim {EMB_DIM}")
        plt.xlabel("Epoch")
        plt.ylabel("MSE")
        plt.tight_layout()
        plt.savefig(out_dir / "loss_curve.png", dpi=300)
        plt.close()

        print(f"✓ Saved to {out_dir}")

    print("\nAll runs complete. Results are under data/weather_allout/") 
