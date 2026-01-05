#!/usr/bin/env python3
import os
from dataclasses import dataclass
from datetime import date
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from phymut.paths import output_dir
from phymut.forecasting.models.build import build_model
from phymut.forecasting.utils.dataloader import load_dataset, load_and_clean_gt
from phymut.forecasting.utils.dir_utils import build_paths
from phymut.forecasting.utils.weather_embed_utils import load_weather_embedding_multi


@dataclass(frozen=True)
class InterpretabilityConfig:
    input_rows: List[str]
    cutoff_date: date
    num_plots: int = 40
    n_weather_pcs: int = 8


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0 or y.size == 0:
        return np.nan
    x_std = np.std(x)
    y_std = np.std(y)
    if x_std == 0 or y_std == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def compute_weather_pcs(season: str, cutoff_date: date, n_components: int) -> pd.DataFrame:
    weather_df = load_weather_embedding_multi(season)
    weather_df = weather_df.loc[weather_df.index > cutoff_date]
    if weather_df.empty:
        raise RuntimeError(f"No weather embeddings after cutoff for season {season}")
    pca = PCA(n_components=n_components, random_state=42)
    pcs = pca.fit_transform(weather_df.values)
    pc_cols = [f"weather_pc{i+1}" for i in range(n_components)]
    return pd.DataFrame(pcs, index=weather_df.index, columns=pc_cols)


def build_feature_yield_arrays(
    season: str,
    cfg: InterpretabilityConfig,
    *,
    use_gt_inputs: bool = False,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[date]]:
    paths = build_paths(season)
    yield_dfs = load_dataset(
        yield_base_dir=str(paths["yield_dir"]),
        yield_csv_name=paths["yield_csv"],
        ground_truth_dir=str(paths["gt_dir"]),
        input_rows=cfg.input_rows,
        cutoff_date=cfg.cutoff_date,
        use_gt_inputs=use_gt_inputs,
    )
    weather_pc_df = compute_weather_pcs(
        season,
        cutoff_date=cfg.cutoff_date,
        n_components=cfg.n_weather_pcs,
    )

    dates = sorted(set(yield_dfs.keys()) & set(weather_pc_df.index))
    if not dates:
        raise RuntimeError(f"No overlapping dates for season {season}")

    feature_names = cfg.input_rows + list(weather_pc_df.columns)
    features_by_date: List[np.ndarray] = []
    yields_by_date: List[np.ndarray] = []
    valid_dates: List[date] = []

    for d in dates:
        df = yield_dfs[d]
        plot_cols = df.columns.tolist()
        agr_rows = []
        for feat in cfg.input_rows:
            if feat not in df.index:
                agr_rows.append(np.zeros(len(plot_cols), dtype=np.float32))
            else:
                agr_rows.append(df.loc[feat].reindex(plot_cols).fillna(0).values.astype(np.float32))
        agr_mat = np.stack(agr_rows, axis=1)

        pc_vals = weather_pc_df.loc[d].values.astype(np.float32)
        weather_mat = np.repeat(pc_vals[None, :], agr_mat.shape[0], axis=0)
        feat_mat = np.concatenate([agr_mat, weather_mat], axis=1)

        gt = load_and_clean_gt(d, str(paths["gt_dir"]))
        y_keys = [idx for idx in gt.index if "yield" in idx]
        if not y_keys:
            continue
        y_vals = gt.loc[y_keys[0]].reindex(plot_cols).fillna(0).values.astype(np.float32)

        features_by_date.append(feat_mat)
        yields_by_date.append(y_vals)
        valid_dates.append(d)

    if not features_by_date:
        raise RuntimeError(f"No valid feature/yield pairs found for season {season}")

    features = np.stack(features_by_date, axis=0)
    yields = np.stack(yields_by_date, axis=0)
    return features, yields, feature_names, valid_dates


def compute_lagged_correlations(
    features: np.ndarray,
    yields: np.ndarray,
    feature_names: List[str],
    horizons: Iterable[int],
    season: str,
) -> pd.DataFrame:
    records = []
    n_dates = features.shape[0]
    for h in horizons:
        if h <= 0 or h >= n_dates:
            continue
        x = features[:-h].reshape(-1, features.shape[-1])
        y = yields[h:].reshape(-1)
        for i, name in enumerate(feature_names):
            corr = _safe_corr(x[:, i], y)
            records.append({
                "season": season,
                "horizon": h,
                "feature": name,
                "corr": corr,
                "n_samples": y.size,
            })
    return pd.DataFrame.from_records(records)


def plot_lagged_correlation_heatmap(
    df: pd.DataFrame,
    out_path: str,
    *,
    title: str,
) -> None:
    if df.empty:
        return
    pivot = df.pivot(index="feature", columns="horizon", values="corr")
    fig, ax = plt.subplots(figsize=(8, max(4, 0.45 * len(pivot.index))))
    im = ax.imshow(pivot.values, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(c) for c in pivot.columns], fontsize=12)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=10)
    ax.set_xlabel("Forecast horizon (weeks)", fontsize=12)
    ax.set_ylabel("Feature", fontsize=12)
    ax.set_title(title, fontsize=14)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _train_yield_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    model_name: str,
    forecast_len: int,
    hid_dim: int,
    num_layers: int,
    lr: float,
    num_epochs: int,
) -> torch.nn.Module:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        model_name=model_name,
        in_dim=X_train.shape[2],
        hid_dim=hid_dim,
        forecast_len=forecast_len,
        num_layers=num_layers,
    ).to(device)

    X_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_train, dtype=torch.float32, device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = torch.nn.SmoothL1Loss()

    model.train()
    for _ in range(num_epochs):
        optimizer.zero_grad()
        pred = model(X_t)
        loss = criterion(pred, y_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    return model


def _rmse_per_horizon(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    errs = (y_pred - y_true) ** 2
    return np.sqrt(np.mean(errs, axis=0))


def compute_permutation_importance(
    season: str,
    cfg: InterpretabilityConfig,
    *,
    model_name: str,
    seq_len: int,
    forecast_len: int,
    hid_dim: int,
    num_layers: int,
    lr: float,
    num_epochs: int,
    seed: int = 42,
) -> Tuple[pd.DataFrame, List[str]]:
    from phymut.forecasting.utils.build_sequences import create_multistep_sequences

    np.random.seed(seed)
    torch.manual_seed(seed)

    paths = build_paths(season)
    yield_dfs = load_dataset(
        yield_base_dir=str(paths["yield_dir"]),
        yield_csv_name=paths["yield_csv"],
        ground_truth_dir=str(paths["gt_dir"]),
        input_rows=cfg.input_rows,
        cutoff_date=cfg.cutoff_date,
        use_gt_inputs=False,
    )
    X_pre, _, _, _, Y_post_yield, _ = create_multistep_sequences(
        yield_dfs,
        season=season,
        gt_dir=str(paths["gt_dir"]),
        seq_len=seq_len,
        input_rows=cfg.input_rows,
        num_plots=cfg.num_plots,
    )
    X_all = np.nan_to_num(X_pre)
    Y_target = np.nan_to_num(Y_post_yield)

    X_train, X_test, y_train, y_test = train_test_split(
        X_all,
        Y_target,
        test_size=0.2,
        random_state=seed,
        shuffle=True,
    )

    y_mean = y_train.mean()
    y_std = y_train.std() if y_train.std() > 0 else 1.0
    y_train_std = (y_train - y_mean) / y_std

    model = _train_yield_model(
        X_train,
        y_train_std,
        model_name=model_name,
        forecast_len=forecast_len,
        hid_dim=hid_dim,
        num_layers=num_layers,
        lr=lr,
        num_epochs=num_epochs,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        base_pred = model(torch.tensor(X_test, dtype=torch.float32, device=device)).cpu().numpy()
    base_pred = (base_pred * y_std) + y_mean
    base_rmse = _rmse_per_horizon(y_test, base_pred)
    base_rmse_overall = float(np.sqrt(np.mean((base_pred - y_test) ** 2)))

    feature_names = cfg.input_rows + [f"weather_pc{i+1}" for i in range(cfg.n_weather_pcs)]
    records = []
    n_samples = X_test.shape[0]

    for feat_idx, feat_name in enumerate(feature_names):
        X_perm = X_test.copy()
        perm_idx = np.random.permutation(n_samples)
        X_perm[:, :, feat_idx] = X_perm[perm_idx, :, feat_idx]
        with torch.no_grad():
            perm_pred = model(torch.tensor(X_perm, dtype=torch.float32, device=device)).cpu().numpy()
        perm_pred = (perm_pred * y_std) + y_mean
        perm_rmse = _rmse_per_horizon(y_test, perm_pred)
        perm_rmse_overall = float(np.sqrt(np.mean((perm_pred - y_test) ** 2)))

        for h, delta in enumerate(perm_rmse - base_rmse, start=1):
            records.append({
                "season": season,
                "seq_len": seq_len,
                "horizon": h,
                "feature": feat_name,
                "rmse_delta": float(delta),
            })
        records.append({
            "season": season,
            "seq_len": seq_len,
            "horizon": "overall",
            "feature": feat_name,
            "rmse_delta": float(perm_rmse_overall - base_rmse_overall),
        })

    return pd.DataFrame.from_records(records), feature_names


def summarize_topk_importance(
    df: pd.DataFrame,
    *,
    k: int = 5,
) -> pd.DataFrame:
    records = []
    for (season, seq_len, horizon), sub in df.groupby(["season", "seq_len", "horizon"]):
        top = sub.sort_values("rmse_delta", ascending=False).head(k)
        for _, row in top.iterrows():
            records.append({
                "season": season,
                "seq_len": seq_len,
                "horizon": horizon,
                "feature": row["feature"],
                "rmse_delta": row["rmse_delta"],
            })
    return pd.DataFrame.from_records(records)


def plot_importance_heatmap(
    df: pd.DataFrame,
    out_path: str,
    *,
    title: str,
) -> None:
    sub = df[df["horizon"] != "overall"].copy()
    if sub.empty:
        return
    pivot = sub.pivot(index="feature", columns="horizon", values="rmse_delta")
    fig, ax = plt.subplots(figsize=(8, max(4, 0.45 * len(pivot.index))))
    im = ax.imshow(pivot.values, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(c) for c in pivot.columns], fontsize=12)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=10)
    ax.set_xlabel("Forecast horizon (weeks)", fontsize=12)
    ax.set_ylabel("Feature", fontsize=12)
    ax.set_title(title, fontsize=14)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def run_lagged_correlation_analysis(
    seasons: Iterable[str],
    cfg: InterpretabilityConfig,
    horizons_by_season: Dict[str, List[int]],
    *,
    out_root: str,
    use_gt_inputs: bool = False,
    cfg_by_season: Dict[str, InterpretabilityConfig] | None = None,
) -> pd.DataFrame:
    all_rows = []
    for season in seasons:
        season_cfg = cfg_by_season.get(season, cfg) if cfg_by_season else cfg
        features, yields, feature_names, _ = build_feature_yield_arrays(
            season,
            season_cfg,
            use_gt_inputs=use_gt_inputs,
        )
        horizons = horizons_by_season.get(season, [])
        df = compute_lagged_correlations(features, yields, feature_names, horizons, season)
        all_rows.append(df)
        plot_lagged_correlation_heatmap(
            df,
            out_path=os.path.join(out_root, f"lagged_corr_{season}.png"),
            title=f"Lagged feature-yield correlations ({season})",
        )
    if not all_rows:
        return pd.DataFrame()
    result = pd.concat(all_rows, ignore_index=True)
    result.to_csv(os.path.join(out_root, "lagged_correlations.csv"), index=False)
    return result


def run_permutation_importance_analysis(
    seasons: Iterable[str],
    cfg: InterpretabilityConfig,
    *,
    model_name: str,
    seq_len_by_season: Dict[str, int],
    forecast_len_by_season: Dict[str, int],
    hid_dim: int,
    num_layers: int,
    lr: float,
    num_epochs: int,
    out_root: str,
    cfg_by_season: Dict[str, InterpretabilityConfig] | None = None,
) -> pd.DataFrame:
    all_rows = []
    for season in seasons:
        season_cfg = cfg_by_season.get(season, cfg) if cfg_by_season else cfg
        seq_len = seq_len_by_season[season]
        forecast_len = forecast_len_by_season[season]
        df, _ = compute_permutation_importance(
            season,
            season_cfg,
            model_name=model_name,
            seq_len=seq_len,
            forecast_len=forecast_len,
            hid_dim=hid_dim,
            num_layers=num_layers,
            lr=lr,
            num_epochs=num_epochs,
        )
        all_rows.append(df)
        plot_importance_heatmap(
            df,
            out_path=os.path.join(out_root, f"perm_importance_{season}_seq{seq_len}.png"),
            title=f"Permutation importance ({season}, seq={seq_len})",
        )
    if not all_rows:
        return pd.DataFrame()
    result = pd.concat(all_rows, ignore_index=True)
    result.to_csv(os.path.join(out_root, "permutation_importance.csv"), index=False)
    topk = summarize_topk_importance(result, k=5)
    topk.to_csv(os.path.join(out_root, "permutation_importance_topk.csv"), index=False)
    return result
