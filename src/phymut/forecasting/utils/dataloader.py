import os
import re
import random
import traceback

from datetime import datetime, date
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from math import sin, cos, pi

from pandas.errors import ParserError
from sklearn.preprocessing import StandardScaler


def load_dataset(
    yield_base_dir: str,
    yield_csv_name: str,
    ground_truth_dir: str,
    input_rows: List[str],
    cutoff_date: date,
    *,
    date_format: str = "%y%m%d",
    debug: bool = False,
    sample_count: int = 2,
    use_gt_inputs: bool = False
) -> Tuple[Dict[date, pd.DataFrame], Dict[date, np.ndarray]]:
    """
    Load and preprocess yield and weather data.

    1) Load yield CSVs from dated subfolders.
    2) Filter out dates <= cutoff_date.
    3) Load weather CSV → daily means dict.
    4) Load & clean ground-truth CSVs.
    5) Merge GT inputs & targets into yield_dfs if `use_gt_inputs` is True.
    6) Interpolate missing values by row medians.
    7) Scale inputs in-place.
    8) Return only the rows specified in `input_rows`.

    Parameters
    ----------
    yield_base_dir : str
        Directory containing dated subfolders with yield CSVs.
    yield_csv_name : str
        Filename of the yield CSV within each subfolder.
    weather_base_dir : str
        Directory containing the weather CSV.
    weather_csv_name : str
        Filename of the weather CSV.
    ground_truth_dir : str
        Directory containing ground-truth CSV files.
    input_rows : List[str]
        List of row labels to retain as features, e.g.
        ['strawberry_flower', 'strawberry_green', ...].
    cutoff_date : date
        Only include data after this date.
    date_format : str, optional
        Format to parse subfolder names as dates.
    debug : bool, optional
        If True, print debug info for ground-truth.
    sample_count : int, optional
        Number of GT samples to print in debug mode.
    use_gt_inputs : bool, optional
        If True, override algorithm-computed inputs with ground-truth values.

    Returns
    -------
    yield_dfs : Dict[date, pd.DataFrame]
        Mapping from date to DataFrame containing only `input_rows`,
        with missing values interpolated and scaled.
    weather_dict : Dict[date, np.ndarray]
        Mapping from date to its daily-mean weather feature vector.
    """
    # ────────────────────────────────────────────────────────────────────────────
    # 1) Load yield data
    if not os.path.isdir(yield_base_dir):
        raise FileNotFoundError(f"Yield base directory not found: {yield_base_dir}")
    yield_dfs: Dict[date, pd.DataFrame] = {}
    for sub in sorted(os.listdir(yield_base_dir)):
        subdir = os.path.join(yield_base_dir, sub)
        if not os.path.isdir(subdir) or not re.fullmatch(r"\d{6}", sub):
            continue
        try:
            d = datetime.strptime(sub, date_format).date()
        except ValueError:
            continue
        csv_path = os.path.join(subdir, yield_csv_name)
        try:
            df = pd.read_csv(csv_path, index_col=0, na_values="N/A")
        except FileNotFoundError:
            continue
        except ParserError:
            df = pd.read_csv(csv_path, index_col=0, na_values="N/A",
                             engine="python", on_bad_lines="skip")
        df = df.apply(pd.to_numeric, errors="coerce")
        df.columns = df.columns.str.lower()
        df = df.apply(pd.to_numeric, errors="coerce").astype(np.float64)
        yield_dfs[d] = df

    if not yield_dfs:
        raise FileNotFoundError(f"No yield CSVs found under {yield_base_dir!r}")

    # ────────────────────────────────────────────────────────────────────────────
    # 2) Filter by cutoff_date
    yield_dfs = {d: df for d, df in yield_dfs.items() if d > cutoff_date}
    if not yield_dfs:
        raise ValueError(f"No input data available after cutoff date {cutoff_date}")

    # grab columns for later use
    first_df = next(iter(yield_dfs.values()))
    plot_cols = list(first_df.columns)

    # ────────────────────────────────────────────────────────────────────────────
    # 3) Load & clean ground-truth
    gt_dfs: Dict[date, pd.DataFrame] = {}
    abs_gt = os.path.abspath(ground_truth_dir)
    print(f"Loading ground-truth from: {abs_gt}")
    if os.path.isdir(abs_gt):
        for fn in os.listdir(abs_gt):
            if not fn.lower().endswith(".csv"):
                continue
            m = re.search(r"(\d{6})\.csv$", fn)
            if not m:
                continue
            d = datetime.strptime(m.group(1), "%y%m%d").date()
            try:
                gtdf = pd.read_csv(os.path.join(abs_gt, fn),
                                   index_col=0, na_values="N/A")
            except Exception:
                traceback.print_exc()
                continue

            # standardize row & column names
            gtdf.columns = (
                gtdf.columns
                     .str.replace(r"""['"`´„]""", "", regex=True)
                     .str.replace(r"\u00A0", " ", regex=True)
                     .str.replace(r"\s+", " ", regex=True)
                     .str.strip()
                     .str.lower()
            )
            gtdf = gtdf.loc[:, ~gtdf.columns.str.startswith("unnamed")]
            if gtdf.columns.duplicated(keep=False).any():
                gtdf = gtdf.loc[:, ~gtdf.columns.duplicated(keep="first")]
            # drop header row if labelled "counting"
            if gtdf.index.size and str(gtdf.index[0]).strip().lower() == "counting":
                gtdf = gtdf.iloc[1:]

            # build mapping for targets and (optionally) inputs
            gt_row_map = {
                "w": "Yield (g)",
                "r": "strawberry_red"
            }
            if use_gt_inputs:
                gt_row_map.update({
                    "fl": "strawberry_flower",
                    "g": "strawberry_green",
                    "w": "strawberry_white",
                    "p": "strawberry_pink"
                })
            gtdf.rename(index=lambda x: gt_row_map.get(str(x).strip().lower(), x),
                        inplace=True)

            # keep first duplicate if any
            if not gtdf.columns.duplicated().any():
                gt_dfs[d] = gtdf

    # ────────────────────────────────────────────────────────────────────────────
    # 4) Merge ground-truth targets & (optionally) inputs
    for d, ydf in yield_dfs.items():
        gtdf = gt_dfs.get(d)
        if gtdf is None:
            continue

        # make sure all entries in gtdf are numeric
        gtdf = gtdf.apply(pd.to_numeric, errors="coerce")

        # override inputs from GT if requested
        if use_gt_inputs:
            for lbl in input_rows:
                if lbl in gtdf.index:
                    raw = gtdf.loc[lbl]
                    if isinstance(raw, pd.DataFrame):
                        raw = raw.iloc[0]
                    s = pd.Series(0.0, index=plot_cols, dtype=float)
                    s.update(raw)
                    ydf.loc[lbl] = s

        # merge the two targets in the same safe way
        for label in ["Yield (g)", "strawberry_red"]:
            if label in gtdf.index:
                raw = gtdf.loc[label]
                if isinstance(raw, pd.DataFrame):
                    # take the first duplicate row
                    raw = raw.iloc[0]
                # now raw is a 1-D Series and safe to update()
                s = pd.Series(0.0, index=plot_cols, dtype=float)
                s.update(raw)
                ydf.loc[label] = s

    # drop duplicate row labels
    for d in list(yield_dfs):
        df = yield_dfs[d]
        yield_dfs[d] = df[~df.index.duplicated(keep="first")].astype(np.float64)

    # ────────────────────────────────────────────────────────────────────────────
    # 5) Interpolate missing values by row median
    for df in yield_dfs.values():
        medians = df.median(axis=1)
        for idx, med in medians.items():
            df.loc[idx] = df.loc[idx].fillna(med)


    # ────────────────────────────────────────────────────────────────────────────
    # 6) Scale inputs
    scalers = {lbl: StandardScaler() for lbl in input_rows}
    # collect data for fitting
    all_vals = {lbl: [] for lbl in input_rows}
    for df in yield_dfs.values():
        for lbl in input_rows:
            if lbl in df.index:
                v = df.loc[lbl].dropna().values
                if v.size:
                    all_vals[lbl].append(v)
    for lbl, arrs in all_vals.items():
        if arrs:
            scalers[lbl].fit(np.concatenate(arrs).reshape(-1, 1))

    # transform in-place
    for df in yield_dfs.values():
        for lbl, sc in scalers.items():
            if lbl in df.index:
                row = df.loc[lbl]
                mask = row.notna()
                if mask.any():
                    vals = row[mask].values.reshape(-1, 1)
                    df.loc[lbl, mask] = sc.transform(vals).flatten()

    # ────────────────────────────────────────────────────────────────────────────
    # # 7) Scale targets
    # target_labels = ["Yield (g)", "strawberry_red"]
    # target_scalers = {lbl: StandardScaler() for lbl in target_labels}

    # for lbl, scaler in target_scalers.items():
    #     all_vals = np.concatenate(
    #         [df.loc[lbl].values for df in yield_dfs.values()]
    #     ).reshape(-1, 1)
    #     scaler.fit(all_vals)

    # for df in yield_dfs.values():
    #     for lbl, scaler in target_scalers.items():
    #         vals = df.loc[lbl].values.reshape(-1, 1)
    #         df.loc[lbl] = scaler.transform(vals).flatten()

    # target_mean_std = {
    #     lbl: (scaler.mean_[0], scaler.scale_[0])
    #     for lbl, scaler in target_scalers.items()
    # }


    # ────────────────────────────────────────────────────────────────────────────
    # 8) Filter to only the requested input_rows
    yield_dfs = {
        d: df.loc[input_rows].copy()
        for d, df in yield_dfs.items()
    }

    return yield_dfs
    # return yield_dfs, target_mean_std



def load_and_clean_gt(date_obj, gt_dir):
    """
    Returns a cleaned DataFrame for date_obj (YYMMDD.csv) with:
      - columns & index stripped/lowercased
      - all values coerced to float, NaNs filled with 0
    """
    fname   = date_obj.strftime("%y%m%d") + ".csv"
    path    = os.path.join(gt_dir, fname)
    # print(f"Loading ground truth from: {path}")
    df      = pd.read_csv(path, index_col=0, na_values="N/A")

    # clean labels
    df.columns = (
        df.columns.astype(str)
             .str.replace(r"""['"`´„]""", "", regex=True)
             .str.replace(r'\u00A0', ' ', regex=True)
             .str.replace(r'\s+', ' ', regex=True)
             .str.strip()
             .str.lower()
    )
    df.index = (
        df.index.astype(str)
             .str.replace(r"""['"`´„]""", "", regex=True)
             .str.replace(r'\u00A0', ' ', regex=True)
             .str.replace(r'\s+', ' ', regex=True)
             .str.strip()
             .str.lower()
    )
    # drop unnamed cols
    df = df.loc[:, ~df.columns.str.startswith("unnamed")]

    # cast *all* to numeric, fill NaNs
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)

    return df



def load_gt_means(date_obj, gt_dir):
    df = load_and_clean_gt(date_obj, gt_dir)

    # pick the yield row
    y_keys = [r for r in df.index if "yield" in r]
    if not y_keys:
        raise KeyError(f"No ‘yield’ row in {date_obj}: {list(df.index)}")
    y_idx = y_keys[0]

    # pick exactly the strawberry_red row
    if "r" not in df.index:
        raise KeyError(f"No ‘strawberry_red’ in {date_obj}: {list(df.index)}")
    r_idx = "r"

    # use *all* remaining columns as plots
    cols   = df.columns.tolist()
    y_vals = df.loc[y_idx, cols].values  # already float, NaNs → 0
    r_vals = df.loc[r_idx, cols].values

    return float(y_vals.mean()), float(r_vals.mean())