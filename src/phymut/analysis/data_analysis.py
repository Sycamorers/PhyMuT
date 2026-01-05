
#!/usr/bin/env python3
import os, re, glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from phymut.paths import data_dir, output_dir

plt.rcParams.update({
    "xtick.labelsize": 22, # x 轴数字
    "ytick.labelsize": 22 # y 轴数字
})

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
seasons = ["2324", "2425"]

# 仅用于标题展示的季节映射
season_display = {
    "2324": "2023–24",
    "2425": "2024–25"
}

base_path = data_dir()
yield_dirs = {s: os.path.join(base_path, f"{s}_GNV_processed") for s in seasons}
gt_dirs    = {s: os.path.join(yield_dirs[s], "counting_yield") for s in seasons}

algo_fname = "consolidated_summary_with_yield.csv"
plots_dir = str(output_dir("analysis_plots"))
os.makedirs(plots_dir, exist_ok=True)

# Map manual CSV row codes → internal feature keys
manual_map = {
    "FL": "strawberry_flower",
    "G":  "strawberry_green",
    "W":  "strawberry_white",
    "P":  "strawberry_pink",
    "R":  "strawberry_red",
}

# Friendly labels for plot titles
display_map = {
    "strawberry_flower": "Flower (FL)",
    "strawberry_green":  "Green  (G)",
    "strawberry_white":  "White  (W)",
    "strawberry_pink":   "Pink   (P)",
    "strawberry_red":    "Red    (R)",
    "Yield (g)":         "Yield (g)"
}

# Consistent plot colors
colors = {
    "manual": "#1b9e77",  # green
    "yolo":   "#7570b3",  # purple
    "2324":   "#1b9e77",  # green
    "2425":   "#7570b3"   # purple
}

# -----------------------------------------------------------------------------
# LOAD & MERGE ALL DATA
# -----------------------------------------------------------------------------
records = []
for season in seasons:
    ydir, gtdir = yield_dirs[season], gt_dirs[season]
    if not os.path.isdir(ydir) or not os.path.isdir(gtdir):
        print(f"⚠️ Missing data for season {season}: check {ydir} and {gtdir}")
        continue

    for sub in sorted(os.listdir(ydir)):
        if not re.fullmatch(r"\d{6}", sub):
            continue
        date_obj   = datetime.strptime(sub, "%y%m%d").date()
        algo_csv   = os.path.join(ydir, sub, algo_fname)
        manual_csv = os.path.join(gtdir, f"{sub}.csv")
        if not (os.path.isfile(algo_csv) and os.path.isfile(manual_csv)):
            continue

        algo_df = pd.read_csv(algo_csv, index_col=0)
        algo_df.index = algo_df.index.str.lower().str.strip()
        algo_df.columns = algo_df.columns.str.lower().str.strip()

        man_df = pd.read_csv(manual_csv, index_col=0)
        man_df = man_df.loc[man_df.index.str.lower().str.strip() != "counting"]
        man_df.index = man_df.index.str.upper().str.strip().map(manual_map)
        man_df = man_df.loc[man_df.index.dropna()]
        man_df.columns = man_df.columns.str.lower().str.strip()

        for feat in manual_map.values():
            if feat not in algo_df.index or feat not in man_df.index:
                continue
            for loc in algo_df.columns:
                if loc not in man_df.columns:
                    continue
                records.append({
                    "date":            date_obj,
                    "season":          season,
                    "location":        loc,
                    "feature":         feat,
                    "algorithm_count": algo_df.at[feat, loc],
                    "manual_count":    man_df.at[feat, loc],
                })

df = pd.DataFrame.from_records(records)
if df.empty:
    raise RuntimeError("No data loaded – please verify your paths.")

df["difference"] = df["algorithm_count"] - df["manual_count"]

# -----------------------------------------------------------------------------
# DESCRIPTIVE STATISTICS CSVs
# -----------------------------------------------------------------------------
df.groupby(["season","feature"])["algorithm_count"].describe().to_csv(os.path.join(plots_dir, "stats_yolo.csv"))
df.groupby(["season","feature"])["manual_count"].describe().to_csv(os.path.join(plots_dir, "stats_manual.csv"))
df.groupby(["season","feature"])["difference"].describe().to_csv(os.path.join(plots_dir, "stats_difference.csv"))

# -----------------------------------------------------------------------------
# PAIRED HISTOGRAMS
# -----------------------------------------------------------------------------
for feat in df["feature"].unique():
    sub = df[df["feature"] == feat]
    lo, hi = sub[["algorithm_count","manual_count"]].min().min(), \
             sub[["algorithm_count","manual_count"]].max().max()
    fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharex=True, sharey=True)
    for ax, season in zip(axes, seasons):
        sd = sub[sub["season"] == season]
        ax.hist(sd["algorithm_count"].dropna(), bins=30, alpha=0.6, range=(lo,hi),
                label="YOLO", color=colors["yolo"])
        ax.hist(sd["manual_count"].dropna(), bins=30, alpha=0.6, range=(lo,hi),
                label="Manual", color=colors["manual"])
        ax.set_title(f"{season_display[season]} Season – {display_map[feat]}", fontsize=26)
        ax.set_xlim(lo, hi)
        ax.set_xlabel("Count", fontsize=24)
        ax.set_ylabel("Frequency", fontsize=24)
        ax.legend(fontsize=26)
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, f"hist_pair_{feat}.png"))
    plt.close(fig)

# -----------------------------------------------------------------------------
# PAIRED BOXPLOTS
# -----------------------------------------------------------------------------
for feat in df["feature"].unique():
    sub = df[df["feature"] == feat]
    fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=False)
    for ax, season in zip(axes, seasons):
        sd = sub[sub["season"] == season]
        yolo_vals = sd["algorithm_count"].dropna().values
        man_vals  = sd["manual_count"].dropna().values
        data, labels, patch_colors = [], [], []
        if yolo_vals.size:
            data.append(yolo_vals); labels.append("YOLO"); patch_colors.append(colors["yolo"])
        if man_vals.size:
            data.append(man_vals); labels.append("Manual"); patch_colors.append(colors["manual"])
        bplots = ax.boxplot(data, labels=labels, patch_artist=True)
        for patch, color in zip(bplots['boxes'], patch_colors):
            patch.set_facecolor(color)
        ax.set_title(f"{season_display[season]} Season – {display_map[feat]}", fontsize=26)
        ax.set_ylabel("Count", fontsize=24)
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, f"box_pair_{feat}.png"))
    plt.close(fig)

# -----------------------------------------------------------------------------
# SCATTER PLOTS
# -----------------------------------------------------------------------------
for feat in df["feature"].unique():
    sub = df[df["feature"] == feat]
    mn, mx = sub[["manual_count","algorithm_count"]].min().min(), \
             sub[["manual_count","algorithm_count"]].max().max()
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, season in zip(axes, seasons):
        sd = sub[sub["season"] == season]
        ax.scatter(sd["manual_count"], sd["algorithm_count"], alpha=0.7, color=colors[season])
        ax.plot([mn, mx], [mn, mx], "r--", linewidth=2)
        ax.set_title(f"{season_display[season]} Season – {display_map[feat]}", fontsize=26)
        ax.set_xlabel("Manual Count", fontsize=24)
        ax.set_ylabel("YOLO Count", fontsize=24)
        ax.set_xlim(mn, mx)
        ax.set_ylim(mn, mx)
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, f"scatter_{feat}.png"))
    plt.close(fig)

# -----------------------------------------------------------------------------
# FEATURE CORRELATIONS
# -----------------------------------------------------------------------------
df.groupby("feature") \
  .apply(lambda g: g["algorithm_count"].corr(g["manual_count"])) \
  .rename("yolo_manual_corr") \
  .to_csv(os.path.join(plots_dir, "feature_correlations.csv"))

# -----------------------------------------------------------------------------
# SAVE LONG-FORM TABLE
# -----------------------------------------------------------------------------
df.to_csv(os.path.join(plots_dir, "combined_counts_longform.csv"), index=False)

# -----------------------------------------------------------------------------
# BOXPLOTS FOR YIELD ONLY
# -----------------------------------------------------------------------------
yield_records = []
for season in seasons:
    gtdir = gt_dirs[season]
    if not os.path.isdir(gtdir):
        print(f"⚠️  Missing ground-truth dir for season {season}: {gtdir}")
        continue

    pattern = os.path.join(gtdir, "*.csv")
    for manual_csv in sorted(glob.glob(pattern)):
        sub = os.path.splitext(os.path.basename(manual_csv))[0]
        man_df = pd.read_csv(manual_csv, index_col=0)
        valid_mask = man_df.notna().any(axis=1)
        if not valid_mask.any():
            continue
        last_label = valid_mask[valid_mask].index[-1]
        last_row = man_df.loc[last_label]

        for loc, raw in last_row.items():
            try:
                val = float(raw)
            except Exception:
                continue
            yield_records.append({
                "season":   season,
                "location": loc,
                "yield_g":  val
            })

yield_df = pd.DataFrame(yield_records)
if yield_df.empty:
    raise RuntimeError("No yield data found – check your CSVs for a non-empty final row")

# Yield Boxplot
fig, axes = plt.subplots(1, len(seasons), figsize=(12, 6), sharey=True)
for ax, season in zip(axes, seasons):
    vals = yield_df.loc[yield_df["season"] == season, "yield_g"].dropna().values
    if vals.size:
        bplot = ax.boxplot([vals], labels=["Yield (g)"], patch_artist=True)
        bplot['boxes'][0].set_facecolor(colors[season])
    else:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", fontsize=18)
    ax.set_title(f"{season_display[season]} Season – Yield (g)", fontsize=26)
    ax.set_ylabel("Yield (g)", fontsize=24)

plt.tight_layout()
fig.savefig(os.path.join(output_dir, "box_yield.png"))
plt.close(fig)

# -----------------------------------------------------------------------------
# HISTOGRAM FOR YIELD ONLY (by season, with fixed bin widths)
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))

# Compute consistent bins across both seasons
all_vals = yield_df["yield_g"].dropna().values
min_val, max_val = all_vals.min(), all_vals.max()
bins = np.linspace(min_val, max_val, 31)  # 30 bins → 31 edges

for season in seasons:
    vals = yield_df.loc[yield_df["season"] == season, "yield_g"].dropna().values
    if vals.size:
        ax.hist(vals, bins=bins, alpha=0.6, label=season_display[season], color=colors[season])

ax.set_title("Histogram of Yield (g/m²) by Season", fontsize=26)
ax.set_xlabel("Yield (g/m²)", fontsize=24)
ax.set_ylabel("Frequency", fontsize=24)
ax.legend(fontsize=26)
plt.tight_layout()
fig.savefig(os.path.join(output_dir, "hist_yield.png"))
plt.close(fig)

print(f"\n✅ All outputs saved under '{output_dir}/'")
