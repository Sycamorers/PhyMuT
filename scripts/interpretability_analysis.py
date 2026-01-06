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
        feature_label_map={
            "strawberry_flower": "F",
            "strawberry_green": "G",
            "strawberry_white": "W",
            "strawberry_pink": "P",
            "Area": "Area",
            "Volume": "Volume",
            "weather_pc1": "weather_0",
            "weather_pc2": "weather_1",
            "weather_pc3": "weather_2",
            "weather_pc4": "weather_3",
            "weather_pc5": "weather_4",
            "weather_pc6": "weather_5",
            "weather_pc7": "weather_6",
            "weather_pc8": "weather_7",
        },
        feature_label_order=[
            "F",
            "G",
            "W",
            "P",
            "Area",
            "Volume",
            "weather_0",
            "weather_1",
            "weather_2",
            "weather_3",
            "weather_4",
            "weather_5",
            "weather_6",
            "weather_7",
        ],
    )
    cfg_by_season = {
        "2324": cfg,
        "2425": InterpretabilityConfig(
            input_rows=cfg.input_rows,
            cutoff_date=date(2025, 1, 5),
            num_plots=cfg.num_plots,
            n_weather_pcs=cfg.n_weather_pcs,
            feature_label_map=cfg.feature_label_map,
            feature_label_order=cfg.feature_label_order,
        ),
    }

    horizons_by_season = {
        "2324": [1, 2, 3],
        "2425": [1, 2, 3, 4],
    }

    seq_len_by_season = {
        "2324": 5,
        "2425": 5,
    }
    forecast_len_by_season = {
        season: len(horizons_by_season[season]) for season in seasons
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
        horizons_by_season=horizons_by_season,
        hid_dim=64,
        num_layers=1,
        lr=1e-3,
        num_epochs=120,
        out_root=out_root,
        cfg_by_season=cfg_by_season,
    )


if __name__ == "__main__":
    main()
