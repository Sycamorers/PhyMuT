import os
import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
)
import matplotlib.pyplot as plt
from adjustText import adjust_text


def mean_absolute_percentage_error(y_true, y_pred, eps=5.0):
    """
    Compute MAPE, avoiding division by zero and extremely small values.
    Uses a minimum threshold (eps) to prevent MAPE explosion.
    Returns percentage (0–100).
    """
    # Use a larger epsilon to handle small yield values (like 1-2 grams)
    denominator = np.maximum(np.abs(y_true), eps)
    mape = np.mean(np.abs((y_true - y_pred) / denominator)) * 100
    
    # Cap MAPE at a reasonable maximum to prevent display issues
    return min(mape, 9999.99)

def plot_and_save_series_analysis(
    series_dict,
    history_dict,
    forecast_dict,
    out_dir,
    best_params=None
):
    """
    For each series in series_dict, produce and save two PNGs:
      1) Scatter of GT vs. Pred, colored by day, with per‑day boxes showing:
         RMSE, MAE, MAPE, R², Corr, and an overall‑metrics box.
      2) Time‑series comparison: GT History, Pred Forecast, GT Forecast,
         with the legend placed outside to avoid overlapping.
    Filenames include the overall R² and optional parameter suffix.
    """
    os.makedirs(out_dir, exist_ok=True)
    # offsets = [(10, 10), (10, -10), (-10, 10), (-10, -10)]

    for label, data in series_dict.items():
        gt_arr   = np.array(data["gt"])
        pred_arr = np.array(data["pred"])
        input_dates = data["dates"]

        # Flatten arrays and align dates
        if gt_arr.ndim == 2 and len(input_dates) == gt_arr.shape[1]:
            dates = np.repeat(input_dates, gt_arr.shape[0])
        else:
            dates = np.array(input_dates).ravel()
        gt   = gt_arr.ravel()
        pred = pred_arr.ravel()

        # Compute overall metrics
        overall_rmse = np.sqrt(mean_squared_error(gt, pred))
        overall_mae  = mean_absolute_error(gt, pred)
        overall_mape = mean_absolute_percentage_error(gt, pred)
        overall_r2   = r2_score(gt, pred)
        overall_corr = np.corrcoef(gt, pred)[0,1]

        # --- Scatter plot ---
        unique_days = np.unique(dates)
        cmap = plt.cm.get_cmap("tab10", len(unique_days))
        fig_sc, ax_sc = plt.subplots(figsize=(10, 8))


        texts = []
        for i, day in enumerate(unique_days):
            mask     = (dates == day)
            gt_day   = gt[mask]
            pred_day = pred[mask]

            # Per-day metrics
            rmse_day = np.sqrt(mean_squared_error(gt_day, pred_day))
            mae_day  = mean_absolute_error(gt_day, pred_day)
            mape_day = mean_absolute_percentage_error(gt_day, pred_day)
            r2_day   = r2_score(gt_day, pred_day) if len(gt_day) > 1 else np.nan
            corr_day = np.corrcoef(gt_day, pred_day)[0,1] if len(gt_day) > 1 else np.nan

            ax_sc.scatter(gt_day, pred_day, color=cmap(i), alpha=0.7, s=50)

            x_med, y_med = np.median(gt_day), np.median(pred_day)

            info_text = (
                f"{day.strftime('%Y-%m-%d')}\n"
                f"RMSE={rmse_day:.2f}\n"
                f"MAE={mae_day:.2f}\n"
                f"MAPE={mape_day:.1f}%\n"
                f"R²={r2_day:.2f}\n"
                f"Corr={corr_day:.2f}"
            )
            # text = ax_sc.annotate(
            #     info_text, xy=(x_med, y_med), xytext=(dx, dy),
            #     textcoords='offset points', ha=ha, va=va,
            #     fontsize=8,
            #     bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            #     color=cmap(i)
            # )
            txt = ax_sc.text(x_med, y_med, info_text,
                            fontsize=8,
                            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
                            color=cmap(i))
            texts.append(txt)
            
        adjust_text(
            texts,
            ax=ax_sc,
            expand_text=(3, 3),          # how much to push labels away
            expand_points=(2, 2),        # how much to push points away
        )
        

        lo, hi = min(gt.min(), pred.min()), max(gt.max(), pred.max())
        ax_sc.plot([lo, hi], [lo, hi], "--", color="red", linewidth=1)
        ax_sc.set_xlim(lo, hi)
        ax_sc.set_ylim(lo, hi)
        ax_sc.set_xlabel("Ground Truth")
        ax_sc.set_ylabel("Prediction")
        ax_sc.set_title(f"{label}: GT vs. Pred (colored by day)")

        overall_text = (
            "Overall Metrics:\n"
            f" RMSE = {overall_rmse:.3f}\n"
            f" MAE  = {overall_mae:.3f}\n"
            f" MAPE = {overall_mape:.1f}%\n"
            f" R²   = {overall_r2:.3f}\n"
            f" Corr = {overall_corr:.3f}"
        )
        ax_sc.text(
            0.98, 0.02, overall_text,
            transform=ax_sc.transAxes, fontsize=11,
            ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8)
        )

        fig_sc.tight_layout()
        safe_label = label.lower().replace(" ", "_")
        suffix = ""
        if best_params:
            suffix = f"_hid{best_params['hid_dim']}_layers{best_params['num_layers']}"
        scatter_fname = os.path.join(
            out_dir,
            f"{safe_label}_scatter_r2_{overall_r2:.2f}{suffix}_MAPE_{overall_mape:.2f}.png"
        )
        fig_sc.savefig(scatter_fname, dpi=300)
        plt.close(fig_sc)
        print(f"Saved scatter plot: {scatter_fname}")

        # --- Time-series plot with legend outside ---
        hist = history_dict[label]
        fcst = forecast_dict[label]
        fig_ts, ax_ts = plt.subplots(figsize=(12, 6))

        ax_ts.plot(
            hist["dates"], hist["values"],
            marker="o", linestyle="-", label="GT History", linewidth=2
        )
        ax_ts.plot(
            fcst["dates"], fcst["pred_values"],
            marker="s", linestyle="--", label="Pred Forecast", linewidth=2
        )
        ax_ts.plot(
            fcst["dates"], fcst["gt_values"],
            marker="^", linestyle="-.", label="GT Forecast", linewidth=2
        )

        # Ticks & labels
        all_dates = hist["dates"] + fcst["dates"]
        ax_ts.set_xticks(all_dates)
        ax_ts.set_xticklabels(
            [d.strftime("%Y-%m-%d") for d in all_dates],
            rotation=45, ha="right"
        )
        ax_ts.set_xlabel("Date")
        ax_ts.set_ylabel(f"Average {label}")
        ax_ts.set_title(f"{label}: History vs. Forecast")
        ax_ts.grid(alpha=0.3)

        # Place legend outside to the right
        ax_ts.legend(
            loc='center left',
            bbox_to_anchor=(1.02, 0.5),
            borderaxespad=0,
            frameon=True
        )

        # Make room on the right for the legend
        fig_ts.subplots_adjust(right=0.75)

        fig_ts.tight_layout()
        timeseries_fname = os.path.join(
            out_dir,
            f"{safe_label}_timeseries_r2_{overall_r2:.2f}{suffix}.png"
        )
        fig_ts.savefig(timeseries_fname, dpi=300, bbox_inches='tight')
        plt.close(fig_ts)
        print(f"Saved time-series plot: {timeseries_fname}")
