import numpy as np
import pandas as pd

from datetime import date, timedelta
from math import sin, cos, pi
from typing import List, Dict, Tuple
from .weather_embed_utils import load_weather_embedding_multi
from .dataloader import load_and_clean_gt
from sklearn.decomposition import PCA




def get_weather_emb(day: date,
                    cache: Dict[date, np.ndarray]) -> np.ndarray:
    """
    Retrieve the weather embedding for a given date. If the exact date is
    not in the cache, step back one day at a time until a cached embedding
    is found or the cache minimum date is exceeded.

    Args:
        day: The target date for which to fetch the embedding.
        cache: A mapping from date to its 64-dimensional embedding vector.

    Returns:
        A 1D numpy array of length 64 with the weather embedding.

    Raises:
        KeyError: If no embedding can be found on or before the earliest date in cache.
    """
    # Keep stepping back until we hit a known date or run off the cache
    while day not in cache:
        day -= timedelta(days=1)
        if day < min(cache):
            raise KeyError(f"No embedding available for {day}")
    return cache[day]


def positional_encoding(num_plots: int = 40, d_model: int = 64) -> np.ndarray:
    """
    Generate a sinusoidal positional encoding matrix for each plot index.

    Args:
        num_plots: Total number of spatial plot indices (e.g., 40).
        d_model: Dimensionality of the encoding (must be even).

    Returns:
        A (num_plots, d_model) array where each row is the encoding for a plot index.
    """
    pe = np.zeros((num_plots, d_model), dtype=np.float32)
    for pos in range(num_plots):
        for i in range(0, d_model, 2):
            div_term = 10000 ** (i / d_model)
            pe[pos, i] = sin(pos / div_term)
            pe[pos, i + 1] = cos(pos / div_term)
    return pe



def create_multistep_sequences(
    dfs: Dict[date, pd.DataFrame],
    season: str,
    gt_dir: str,
    # target_col: str,
    seq_len: int,
    input_rows: List[str],
    num_plots: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build sequences for both training and forecasting periods.

    Returns:
        X_train: (num_plots, seq_len, feat_dim)
        X_pred: (num_plots, n_forecast, feat_dim)
        Y_train_yield: (num_plots, seq_len)
        Y_train_red: (num_plots, seq_len)
        Y_yield: (num_plots, n_forecast)
        Y_red: (num_plots, n_forecast)
    """
    
    
    # Initialize PCA for weather embeddings
    pca = PCA(n_components=8, random_state=42)
    
    # 1) Split dates
    all_dates = sorted(dfs.keys())
    if len(all_dates) <= seq_len:
        raise ValueError(f"Need more dates than seq_len={seq_len}, got {len(all_dates)}")
    train_dates = all_dates[:seq_len]
    forecast_dates = all_dates[seq_len:]

    # 2) Prepare encodings
    plot_cols = dfs[train_dates[0]].rename(columns=str.lower).columns.tolist()
    # target_col = "Soil Temp (C)"
    # weather_df = load_weather_embedding(season, target_col)
    
    weather_df = load_weather_embedding_multi(season)
    # print(f"Loaded weather embeddings for {season} with shape {weather_df.shape}")
    weather_dict = {(d.date() if hasattr(d, 'date') else d): row.values.astype(np.float32)
                    for d, row in weather_df.iterrows()}
    pos_enc = positional_encoding(num_plots, d_model=64)

    # 3) Build X_train
    x_train_days: List[np.ndarray] = []
    for d in train_dates:
        df_day = (dfs[d].rename(columns=str.lower)
                   .loc[:, ~dfs[d].columns.duplicated()]
                   .loc[~dfs[d].index.duplicated()])
        # agronomic features
        agr_rows = []
        for feat in input_rows:
            vals = df_day.loc[feat].reindex(plot_cols).fillna(0).values if feat in df_day.index else np.zeros(num_plots)
            agr_rows.append(vals.astype(np.float32))
        agr_mat = np.stack(agr_rows, axis=1)
        # # weather + pos
        base_w = get_weather_emb(d, weather_dict)
        weather_mat = np.vstack([base_w + pos_enc[p] for p in range(num_plots)])
        weather_mat = pca.fit_transform(weather_mat)
        x_train_days.append(np.concatenate([agr_mat, weather_mat], axis=1))
        
        # # only agronomic features
        # x_train_days.append(agr_mat)
        
        # # only weather
        # base_w = get_weather_emb(d, weather_dict)
        # weather_mat = np.vstack([base_w + pos_enc[p] for p in range(num_plots)])
        # weather_mat = pca.fit_transform(weather_mat)
        # x_train_days.append(weather_mat) 
        
    X_pre = np.swapaxes(np.stack(x_train_days), 0, 1)

    # 4) Build X_pred for forecasting period
    x_pred_days: List[np.ndarray] = []
    for d in forecast_dates:
        df_day = (dfs[d].rename(columns=str.lower)
                   .loc[:, ~dfs[d].columns.duplicated()]
                   .loc[~dfs[d].index.duplicated()])
        agr_rows = []
        for feat in input_rows:
            vals = df_day.loc[feat].reindex(plot_cols).fillna(0).values if feat in df_day.index else np.zeros(num_plots)
            agr_rows.append(vals.astype(np.float32))
        agr_mat = np.stack(agr_rows, axis=1)
        base_w = get_weather_emb(d, weather_dict)
        weather_mat = np.vstack([base_w + pos_enc[p] for p in range(num_plots)])
        weather_mat = pca.fit_transform(weather_mat)
        x_pred_days.append(np.concatenate([agr_mat, weather_mat], axis=1))
        
        # # only agronomic features
        # x_pred_days.append(agr_mat)
        
        # # only weather
        # base_w = get_weather_emb(d, weather_dict)
        # weather_mat = np.vstack([base_w + pos_enc[p] for p in range(num_plots)])
        # weather_mat = pca.fit_transform(weather_mat)
        # x_pred_days.append(weather_mat) 
        
    X_post = np.swapaxes(np.stack(x_pred_days), 0, 1)

    # 5) Build Y_train from ground truth
    y_train_days, r_train_days = [], []
    for d in train_dates:
        df_gt = load_and_clean_gt(d, gt_dir)
        y_keys = [i for i in df_gt.index if 'yield' in i]
        if not y_keys:
            raise KeyError(f"No 'yield' row in {d}")
        y_idx = y_keys[0]
        r_idx = next(i for i in df_gt.index if 'r' in i)
        y_vals = df_gt.loc[y_idx, plot_cols].fillna(0).astype(np.float32).values
        r_vals = df_gt.loc[r_idx, plot_cols].fillna(0).astype(np.float32).values
        y_train_days.append(y_vals)
        r_train_days.append(r_vals)
    Y_pre_yield = np.stack(y_train_days, axis=0).T
    Y_pre_red   = np.stack(r_train_days, axis=0).T

    # 6) Build Y for forecast period
    y_days, r_days = [], []
    for d in forecast_dates:
        df_gt = load_and_clean_gt(d, gt_dir)
        y_keys = [i for i in df_gt.index if 'yield' in i]
        if not y_keys:
            raise KeyError(f"No 'yield' row in {d}")
        y_idx = y_keys[0]
        r_idx = next(i for i in df_gt.index if 'r' in i)
        y_vals = df_gt.loc[y_idx, plot_cols].fillna(0).astype(np.float32).values
        r_vals = df_gt.loc[r_idx, plot_cols].fillna(0).astype(np.float32).values
        y_days.append(y_vals)
        r_days.append(r_vals)
    Y_post_yield = np.stack(y_days, axis=0).T
    Y_post_red   = np.stack(r_days, axis=0).T

    return X_pre, X_post, Y_pre_yield, Y_pre_red, Y_post_yield, Y_post_red
