# weather_embed_utils.py
import re
from pathlib import Path
from datetime import datetime, date
from typing import List

import numpy as np
import pandas as pd

from phymut.paths import data_dir

DATE_FMT = "%y%m%d"
FEATURE_TAG_RE = re.compile(r"[^\w]+")

def _sanitize(col_name: str) -> str:
    """Convert column name to safe folder tag (same rule as trainer)."""
    return FEATURE_TAG_RE.sub("_", col_name).strip("_")


def _collect_week_dates(yield_dir: Path) -> List[date]:
    """Return YYMMDD folder names inside *yield_dir* as sorted date list."""
    days: List[date] = []
    for p in sorted(yield_dir.iterdir()):
        if p.is_dir() and re.fullmatch(r"\d{6}", p.name):
            try:
                days.append(datetime.strptime(p.name, DATE_FMT).date())
            except ValueError:
                pass
    return days


# # old function, kept for reference
# def load_weather_embedding(
#     season: str,
#     target_col: str,
#     yield_root: Path | str = "../yield_data",
#     out_root: Path | str = "wthr_out",
# ) -> pd.DataFrame:
#     """
#     Load the 64-D weekly weather embeddings and align them with their dates.

#     Parameters
#     ----------
#     season : {"2324", "2425", ...}
#         Season identifier used in the training script.
#     target_col : str
#         Exact feature name used as autoregressive target
#         (e.g. "Soil Temp (C)", "Rainfall Amount (in)").
#     yield_root : Path-like, default "../yield_data"
#         Parent directory that contains "<season>_GNV_processed/".
#     out_root : Path-like, default "wthr_out"
#         Parent directory where the embeddings were saved.

#     Returns
#     -------
#     pd.DataFrame
#         shape = (N_weeks, 64), index = weekly sample `datetime.date`,
#         columns = ["w0", "w1", ..., "w63"].
#     """
#     yield_dir = Path(yield_root) / f"{season}_GNV_processed"
#     dates = _collect_week_dates(yield_dir)

#     target_tag = _sanitize(target_col)
#     embed_path = Path(out_root) / season / target_tag / "embed.npy"
#     if not embed_path.exists():
#         raise FileNotFoundError(embed_path)

#     emb = np.load(embed_path)                     # (N_weeks, 64)
#     if emb.shape[0] != len(dates):
#         raise ValueError(
#             f"Mismatch: {embed_path} has {emb.shape[0]} rows "
#             f"but {len(dates)} week folders found in {yield_dir}"
#         )

#     df = pd.DataFrame(
#         emb,
#         index=pd.Index(dates, name="week_date"),
#         columns=[f"w{i}" for i in range(emb.shape[1])]
#     )
#     return df


def load_weather_embedding_multi(
    season: str,
    yield_root: Path | str | None = None,
    out_root: Path | str | None = None,
) -> pd.DataFrame:
    """
    Load the weekly weather embeddings for the multi-target model
    and align them with their dates.

    Parameters
    ----------
    season : str
        Season identifier used in the training script, e.g. "2324".
    yield_root : Path-like, default "../yield_data"
        Parent directory that contains "<season>_GNV_processed/".
    out_root : Path-like, default data/weather_allout
        Parent directory where the embeddings were saved.

    Returns
    -------
    pd.DataFrame
        shape = (N_weeks, D), index = weekly sample `datetime.date`,
        columns = ["w0", "w1", ..., "w{D-1}"], where D is embedding dimension.
    """
    if yield_root is None:
        yield_root = data_dir()
    if out_root is None:
        out_root = data_dir("weather_allout")

    # collect the list of week‐end dates
    yield_dir = Path(yield_root) / f"{season}_GNV_processed"
    dates = _collect_week_dates(yield_dir)  # your existing helper

    # in the new layout, all targets share one folder per season
    embed_path = Path(out_root) / season / "embed.npy"
    if not embed_path.exists():
        raise FileNotFoundError(f"Could not find embeddings at {embed_path!r}")

    # load and sanity-check
    emb = np.load(embed_path)  # shape (N_weeks, D)
    if emb.shape[0] != len(dates):
        raise ValueError(
            f"Row count mismatch: {embed_path!r} has {emb.shape[0]} rows "
            f"but found {len(dates)} week folders in {yield_dir!r}"
        )

    # build DataFrame
    df = pd.DataFrame(
        emb,
        index=pd.Index(dates, name="week_date"),
        columns=[f"w{i}" for i in range(emb.shape[1])]
    )
    return df
