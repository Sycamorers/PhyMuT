#!/usr/bin/env python3
from datetime import date

from phymut.analysis.interpretability import (
    InterpretabilityConfig,
    run_lagged_correlation_analysis,
    run_permutation_importance_analysis,
)
from phymut.paths import output_dir


def main() -> None:
    seasons = ["2324", "2425"]
    cfg = InterpretabilityConfig(
        input_rows=[
            "strawberry_flower",
            "strawberry_green",
            "strawberry_white",
            "strawberry_pink",
            "Area",
            "Volume",
        ],
        cutoff_date=date(2024, 1, 5),
        num_plots=40,
        n_weather_pcs=8,
    )
    cfg_by_season = {
        "2324": cfg,
        "2425": InterpretabilityConfig(
            input_rows=cfg.input_rows,
            cutoff_date=date(2025, 1, 5),
            num_plots=cfg.num_plots,
            n_weather_pcs=cfg.n_weather_pcs,
        ),
    }

    horizons_by_season = {
        "2324": [1, 2, 3],
        "2425": [1, 2, 3, 4],
    }

    seq_len_by_season = {
        "2324": 4,
        "2425": 4,
    }
    forecast_len_by_season = {
        "2324": 8 - seq_len_by_season["2324"],
        "2425": 9 - seq_len_by_season["2425"],
    }

    out_root = str(output_dir("interpretability"))

    run_lagged_correlation_analysis(
        seasons=seasons,
        cfg=cfg,
        horizons_by_season=horizons_by_season,
        out_root=out_root,
        use_gt_inputs=False,
        cfg_by_season=cfg_by_season,
    )

    run_permutation_importance_analysis(
        seasons=seasons,
        cfg=cfg,
        model_name="lstm",
        seq_len_by_season=seq_len_by_season,
        forecast_len_by_season=forecast_len_by_season,
        hid_dim=64,
        num_layers=1,
        lr=1e-3,
        num_epochs=120,
        out_root=out_root,
        cfg_by_season=cfg_by_season,
    )


if __name__ == "__main__":
    main()
