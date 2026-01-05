# PhyMuT: A Physics-Aware, Multi-Modal Time-Series Model for Strawberry Yield Forecasting


This repository contains the code for the PhyMuT strawberry yield forecasting
pipeline used in the paper. It includes:

- Multi-season yield forecasting models (sequence-based predictors)
- Weather embedding training for auxiliary features
- Data diagnostics and analysis scripts

Raw image data and processed data tables are not included. Please request data
access from the authors as described in `data/README.md`.

## Project layout

- `src/phymut/`: core library code
  - `forecasting/`: models, training, and evaluation
  - `analysis/`: dataset diagnostics and weather embedding training
- `scripts/`: runnable entry points
- `data/`: (not included) datasets and weather embeddings
- `outputs/`: generated results and plots
- `archive/`: legacy outputs and experiments (local only, ignored by git)

## Data expectations

After access is granted, arrange files like this:

```
data/
  2324_GNV_processed/
    240123/
      consolidated_summary_with_yield.csv
      ...
    counting_yield/
      240123.csv
      ...
    2324_weather.csv
  2425_GNV_processed/
    ...
  weather_allout/
    2324/
      embed.npy
      model.pth
    2425/
      embed.npy
      model.pth
```

The `weather_allout/` directory can be generated with the weather embedding
script if it is not provided.

## Setup

Recommended environment:

- Python 3.10+
- PyTorch
- numpy, pandas, scikit-learn, matplotlib, adjustText

Example install (CPU-only):

```
pip install torch numpy pandas scikit-learn matplotlib adjustText
```

## Running experiments

All entry points live in `scripts/` and manage paths automatically.

Forecasting (single-process):

```
python scripts/run_yield_forecast.py
```

Forecasting (multi-model parallel runs):

```
python scripts/run_parallel_models.py
```

Data diagnostics (plots and summary tables):

```
python scripts/data_analysis.py
```

Weather embedding training:

```
python scripts/weather_auto_regression.py
```

## Outputs

All generated artifacts are written under `outputs/`:

- `outputs/forecasting/results/`: prediction plots and metrics
- `outputs/forecasting/logs/`: parallel run logs
- `outputs/analysis_plots/`: diagnostic plots and CSV summaries

## Notes for reproducibility

- The code expects the dated subfolders to be named as `YYMMDD`.
- Ground-truth CSVs are loaded from `counting_yield/` inside each season.
- Weather embeddings are loaded from `data/weather_allout/<season>/embed.npy`.

## Citation

If you use this code, please cite the corresponding paper.
