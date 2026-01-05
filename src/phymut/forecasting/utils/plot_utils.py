import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

def plot_and_save_gt_vs_pred(
    Y_yield_all,
    y_pred,
    Y_red_all,
    r_pred,
    out_dir,
):
    """
    Draws two scatter plots (Yield and Red fruit) separately with embedded metrics,
    and saves each resulting figure to out_dir with distinct filenames.
    Returns a list of saved file paths [yield_path, red_path].
    """
    # 1) Compute metrics DataFrame
    metrics = []
    for label, gt, pred in [
        ("Yield",     Y_yield_all, y_pred),
        ("Red fruit", Y_red_all,   r_pred),
    ]:
        err = pred - gt
        metrics.append({
            "RMSE":   np.sqrt(mean_squared_error(gt.ravel(), pred.ravel())),
            "R²":     r2_score(gt.ravel(), pred.ravel()),
            "Corr":   np.corrcoef(gt.ravel(), pred.ravel())[0,1],
        })
    df = pd.DataFrame(metrics).set_index("Series")

    # Ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)

    saved_paths = []

    # 2) Loop through each series and save separately
    for label, gt_all, pred_all in [("Yield", Y_yield_all, y_pred), ("Red fruit", Y_red_all, r_pred)]:
        gt = gt_all.ravel()
        pred = pred_all.ravel()

        # Create a new figure for this series
        fig, ax = plt.subplots(figsize=(6, 5))
        _plot_panel(ax, gt, pred, f"{label}: GT vs. Pred", df.loc[label])
        plt.tight_layout()

        # Build filename with R² value
        r2_val = df.loc[label]["R²"]
        safe_label = label.lower().replace(' ', '_')
        filename = f"gt_vs_pred_{safe_label}_{r2_val:.2f}.png"
        fig_path = os.path.join(out_dir, filename)

        # Save and close
        fig.savefig(fig_path, dpi=300)
        plt.close(fig)

        saved_paths.append(fig_path)

    return saved_paths


def _plot_panel(ax, gt, pred, title, metric_row):
    ax.scatter(gt, pred, alpha=0.7)
    lo, hi = min(gt.min(), pred.min()), max(gt.max(), pred.max())
    ax.plot([lo, hi], [lo, hi], "--", linewidth=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_title(title)
    ax.set_xlabel("Ground-Truth")
    ax.set_ylabel("Predicted")

    txt = (
        f"RMSE = {metric_row['RMSE']:.3f}\n"
        f"R²   = {metric_row['R²']:.3f}\n"
        f"Corr = {metric_row['Corr']:.3f}"
    )
    ax.text(
        0.05, 0.95, txt,
        transform=ax.transAxes,
        fontsize=10,
        va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8)
    )
