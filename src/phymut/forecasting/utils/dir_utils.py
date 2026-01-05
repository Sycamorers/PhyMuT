import os
from pathlib import Path
from typing import Dict

from phymut.paths import data_dir


def build_paths(year_tag: str) -> Dict[str, Path]:
    """
    year_tag:'2324' or '2425'
    """
    base = data_dir(f"{year_tag}_GNV_processed")
    return {
        "yield_dir":  base,
        "gt_dir":     base / "counting_yield",
        "yield_csv": "consolidated_summary_with_yield.csv",  
        "weather_csv": f"{year_tag}_weather.csv"              
    }
