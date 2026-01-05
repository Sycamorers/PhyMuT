# Copy of run_all.py with N-BEATS and ensemble logic
from datetime import date
import numpy as np
import torch
import random
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from phymut.paths import output_dir
from .utils.dataloader import load_dataset, load_gt_means
from .utils.build_sequences import create_multistep_sequences
from .utils.dir_utils import build_paths
from .utils.train_utils import run_grid_search, train_final_models
from .utils.plot_all import plot_and_save_series_analysis

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

INPUT_ROWS = ['strawberry_flower', 'strawberry_green',
              'strawberry_white',  'strawberry_pink', 'Area', 'Volume']

SEASONS = ["2324", "2425"]
# SEASONS = ["2324"]
SEQ_LENS = [4, 5, 6, 7]
# SEQ_LENS = [4]
GT_TOGGLES = [True, False]
# GT_TOGGLES = [True]
EPOCHES = [400]
# Add nbeats to the model list
# MODELS = ["lstm", "rnn", "transformer", "gru", "s2s", "odernn", "nbeats", "tcn"]
MODELS = ["lstm"]

def main():
    results_root = output_dir("forecasting", "results")
    for season in SEASONS:
        if season == "2324":
            cutoff = date(2024, 1, 5)
        else:
            cutoff = date(2025, 1, 5)

        for seq_len in SEQ_LENS:
            for GT_TOGGLE in GT_TOGGLES:
                if season == "2324":
                    forecast_len = 8 - seq_len
                else:
                    forecast_len = 9 - seq_len
                for epoch in EPOCHES:
                    # For ensemble, store predictions for LSTM, GRU, Transformer
                    ensemble_preds_yield = []
                    ensemble_preds_red = []
                    ensemble_model_names = ["lstm", "gru", "transformer"]
                    for model_name in MODELS:
                        out_dir = results_root / f"{season}_{model_name}" / f"manual_{GT_TOGGLE}_seq{seq_len}_epoch{epoch}"
                        os.makedirs(out_dir, exist_ok=True)
                        paths = build_paths(season)
                        print(
                            "To run: season={}, cutoff={}, seq_len={}, GT={}, "
                            "forecast_len={}, out_dir={}".format(
                                season, cutoff, seq_len, GT_TOGGLE, forecast_len, out_dir
                            )
                        )
                        yield_dfs = load_dataset(
                            yield_base_dir=paths["yield_dir"],
                            yield_csv_name=paths["yield_csv"],
                            ground_truth_dir=paths["gt_dir"],
                            input_rows=INPUT_ROWS,
                            cutoff_date=cutoff,
                            debug=False,
                            use_gt_inputs=GT_TOGGLE,
                        )
                        X_pre, X_post, Y_pre_yield, Y_pre_red, Y_post_yield, Y_post_red = create_multistep_sequences(
                            yield_dfs,
                            season=season,
                            gt_dir=paths["gt_dir"],
                            seq_len=seq_len,
                            input_rows=INPUT_ROWS,
                            num_plots=40,
                        )
                        X_all = np.nan_to_num(X_pre)
                        Y_post_yield = np.nan_to_num(Y_post_yield)
                        Y_post_red = np.nan_to_num(Y_post_red)
                        X_train, X_test, Y_tr_yield, Y_te_yield, Y_tr_red, Y_te_red = train_test_split(
                            X_all,
                            Y_post_yield,
                            Y_post_red,
                            test_size=0.2,
                            random_state=42,
                            shuffle=True,
                        )
                        y_mean_train, y_std_train = Y_tr_yield.mean(), Y_tr_yield.std()
                        r_mean_train, r_std_train = Y_tr_red.mean(), Y_tr_red.std()
                        Y_yield_train = (Y_tr_yield - y_mean_train) / y_std_train
                        Y_red_train = (Y_tr_red - r_mean_train) / r_std_train
                        y_mean_testing, y_std_testing = Y_te_yield.mean(), Y_te_yield.std()
                        r_mean_testing, r_std_testing = Y_te_red.mean(), Y_te_red.std()
                        Y_yield_testing = (Y_te_yield - y_mean_testing) / y_std_testing
                        Y_red_testing = (Y_te_red - r_mean_testing) / r_std_testing
                        grid = {
                            "lr": [1e-2, 1e-3],
                            "hid_dim": [32, 64, 128],
                            "num_layers": [1, 2],
                            # "hid_dim":    [64, 128, 256, 512],
                            # "num_layers": [1, 2, 3, 4, 5]
                        }
                        # For N-BEATS, pass seq_len and in_dim to run_grid_search
                        if model_name == "nbeats":
                            best_params, best_loss = run_grid_search(
                                X_all=X_train,
                                Y_yield_all=Y_yield_train,
                                Y_red_all=Y_red_train,
                                forecast_len=forecast_len,
                                grid=grid,
                                model_name=model_name,
                                num_epochs=50,
                                n_splits=10,
                                seq_len=seq_len,
                                in_dim=X_train.shape[2],
                            )
                        else:
                            best_params, best_loss = run_grid_search(
                                X_all=X_train,
                                Y_yield_all=Y_yield_train,
                                Y_red_all=Y_red_train,
                                forecast_len=forecast_len,
                                grid=grid,
                                model_name=model_name,
                                num_epochs=50,
                                n_splits=10,
                            )
                        print(f"Best params: {best_params}, total loss: {best_loss:.4f}")
                        if X_all.shape[0] > 0 and best_params is not None:
                            # For N-BEATS, pass seq_len and in_dim to train_final_models
                            if model_name == "nbeats":
                                model_yield, model_red = train_final_models(
                                    X_train,
                                    Y_yield_train,
                                    Y_red_train,
                                    best_params,
                                    forecast_len,
                                    model_name=model_name,
                                    num_epochs_final=epoch,
                                    seq_len=seq_len,
                                    in_dim=X_train.shape[2],
                                )
                            else:
                                model_yield, model_red = train_final_models(
                                    X_train,
                                    Y_yield_train,
                                    Y_red_train,
                                    best_params,
                                    forecast_len,
                                    model_name=model_name,
                                    num_epochs_final=epoch,
                                )
                            print("✅ Trained Yield and Red models on full dataset.")
                            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                            model_yield.eval()
                            model_red.eval()
                            with torch.no_grad():
                                X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
                                y_pred = model_yield(X_test_tensor).cpu().numpy()
                                r_pred = model_red(X_test_tensor).cpu().numpy()
                        else:
                            model_yield, model_red = None, None
                            y_pred = np.zeros_like(Y_te_yield)
                            r_pred = np.zeros_like(Y_te_red)
                            print("\n⚠️ Skipping final model training due to no data or missing best parameters.")
                        y_pred = (y_pred * y_std_testing) + y_mean_testing
                        r_pred = (r_pred * r_std_testing) + r_mean_testing
                        # For ensemble, collect predictions
                        if model_name in ensemble_model_names:
                            ensemble_preds_yield.append(y_pred)
                            ensemble_preds_red.append(r_pred)
                        dates = sorted(yield_dfs.keys())
                        history_dates = dates[:seq_len]
                        forecast_dates = dates[seq_len: seq_len + forecast_len]
                        history_yield, history_red = [], []
                        forecast_yield, forecast_red = [], []
                        for d in history_dates:
                            my, mr = load_gt_means(d, paths["gt_dir"])
                            history_yield.append(my)
                            history_red.append(mr)
                        for d in forecast_dates:
                            my, mr = load_gt_means(d, paths["gt_dir"])
                            forecast_yield.append(my)
                            forecast_red.append(mr)
                        y_pred_avg = np.mean(y_pred, axis=0)
                        r_pred_avg = np.mean(r_pred, axis=0)
                        series_dict = {
                            "Yield": {
                                "gt": Y_te_yield,
                                "pred": y_pred,
                                "dates": forecast_dates,
                            },
                            "Red fruit": {
                                "gt": Y_te_red,
                                "pred": r_pred,
                                "dates": forecast_dates,
                            },
                        }
                        history_dict = {
                            "Yield": {
                                "dates": history_dates,
                                "values": history_yield,
                            },
                            "Red fruit": {
                                "dates": history_dates,
                                "values": history_red,
                            },
                        }
                        forecast_dict = {
                            "Yield": {
                                "dates": forecast_dates,
                                "gt_values": forecast_yield,
                                "pred_values": y_pred_avg.tolist(),
                            },
                            "Red fruit": {
                                "dates": forecast_dates,
                                "gt_values": forecast_red,
                                "pred_values": r_pred_avg.tolist(),
                            },
                        }
                        plot_and_save_series_analysis(
                            series_dict,
                            history_dict,
                            forecast_dict,
                            out_dir=str(out_dir),
                            best_params=best_params,
                        )
                    # After all models, if we have ensemble predictions, average and plot
                    if len(ensemble_preds_yield) == 3 and len(ensemble_preds_red) == 3:
                        ens_yield = np.mean(np.stack(ensemble_preds_yield, axis=0), axis=0)
                        ens_red = np.mean(np.stack(ensemble_preds_red, axis=0), axis=0)
                        # Use the last test split's ground truth and dates
                        series_dict = {
                            "Yield": {
                                "gt": Y_te_yield,
                                "pred": ens_yield,
                                "dates": forecast_dates,
                            },
                            "Red fruit": {
                                "gt": Y_te_red,
                                "pred": ens_red,
                                "dates": forecast_dates,
                            },
                        }
                        history_dict = {
                            "Yield": {
                                "dates": history_dates,
                                "values": history_yield,
                            },
                            "Red fruit": {
                                "dates": history_dates,
                                "values": history_red,
                            },
                        }
                        forecast_dict = {
                            "Yield": {
                                "dates": forecast_dates,
                                "gt_values": forecast_yield,
                                "pred_values": np.mean(ens_yield, axis=0).tolist(),
                            },
                            "Red fruit": {
                                "dates": forecast_dates,
                                "gt_values": forecast_red,
                                "pred_values": np.mean(ens_red, axis=0).tolist(),
                            },
                        }
                        ens_out_dir = results_root / f"{season}_ensemble" / f"manual_{GT_TOGGLE}_seq{seq_len}_epoch{epoch}"
                        os.makedirs(ens_out_dir, exist_ok=True)
                        plot_and_save_series_analysis(
                            series_dict,
                            history_dict,
                            forecast_dict,
                            out_dir=str(ens_out_dir),
                            best_params=None,
                        )


if __name__ == "__main__":
    main()
