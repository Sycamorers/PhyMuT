# Strawberry Yield Forecasting

This repository contains code and data for predicting strawberry yield using weekly
counts and weather features. Two seasons are available (`2324` and `2425`), each
split into dated subdirectories under `data/SEASON_GNV_processed/`.

### Data overview
* **Yield CSVs**: inside each date folder a `consolidated_summary_with_yield.csv`
  file contains 40 plot columns with several row labels describing the crop stage.
* **Ground truth**: CSVs in `counting_yield/` provide manual counts and yield
  measurements for the same dates.
* **Weather embeddings**: precomputed 64-D vectors reduced to 8-D features are
  stored in `data/weather_allout/SEASON/embed.npy`.

The features used per time step are:
`[strawberry_flower, strawberry_green, strawberry_white, strawberry_pink,
 Area, Volume, weather_PCA_0..7]` giving a 14-D vector per plot.

### Running experiments
Use `scripts/run_yield_forecast.py` to train and evaluate models. The script performs a grid
search over hyperparameters, trains the best model for each configuration, and
saves the resulting plots under `outputs/forecasting/results/`.

```bash
python scripts/run_yield_forecast.py
```

Multiple models can also be launched in parallel:

```bash
python scripts/run_parallel_models.py
```

The important visualisations produced for each run are:
1. **Scatter plots** of prediction vs. ground truth with per-day metrics.
2. **Time-series plots** comparing historical averages with model forecasts.

See the `outputs/forecasting/results/` directory after running for examples.
