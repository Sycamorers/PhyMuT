#!/usr/bin/env python3
from pathlib import Path
import runpy
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

if __name__ == "__main__":
    runpy.run_module("phymut.analysis.weather_auto_regression", run_name="__main__")
