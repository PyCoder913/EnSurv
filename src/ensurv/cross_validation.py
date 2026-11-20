"""Dataset loading and the three-fold experiment."""

import gc
import os
import pickle
import time
import warnings
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from .model import EnSurv
from .preprocessing import prepare_unos_dataset
from .evaluation import compute_all_metrics, analyze_hierarchical_importance
from .utils import plot_time_calibration


def run_cross_validation(dataset_path='/raid/home/pathak/EnSurv/Kidney_preprocessed_final.csv', output_dir="new_ensurv_cv_outputs"):
    r"""Load the kidney CSV and reproduce the three-fold EnSurv experiment.

    :param dataset_path: CSV location. The default retains the original notebook absolute path
        to the dataset.
    :type dataset_path: str or path-like, optional
    :param output_dir: Directory for per-fold CSV, NPZ, PDF, state-dictionary, and history
        files; default new_ensurv_cv_outputs. Aggregate CSVs retain their original filenames in
        the current working directory.
    :type output_dir: str or path-like, optional

    :returns: CV summary tables and the notebook variables needed by the optional post-CV
        plotting examples, including first-fold model/history and last-fold test data.
    :rtype: dict[str, object]

    Uses StratifiedKFold(n_splits=3, shuffle=True, random_state=42), model
    random_state=42 in every fold, and the original metric prediction seed 42.
    The architecture, training options, external time scaling, domain penalties,
    preprocessing order, and saved output names are preserved. No additional
    deterministic-algorithm settings or random seeds are introduced.

    Preprocessing occurs before splitting. Each fold fits feature scaling on its
    training rows and divides times by its training maximum. The generator is
    concatenated, with two 512-unit layers, as selected in the notebook. The
    function returns context only after performing the original output writes.
    """
    warnings.filterwarnings('ignore')
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print("="*80)
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print("="*80)

    df = pd.read_csv(dataset_path)
    X, durations, events, feat_names, nominal_cols, center_covariates = prepare_unos_dataset(df)

    seed = 42
    n_splits = 3
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold1_history=None
    fold1_model=None

    # Where per-fold artifacts go
    OUT_DIR = output_dir
    os.makedirs(OUT_DIR, exist_ok=True)

    # Storage for cross-validation results
    metrics_list = []
    raw_expl_list = []
    base_feat_expl_list = []
    macro_expl_list = []

    print(f"Starting {n_splits}-Fold Cross Validation...\n")
    start_total = time.time()

    for fold, (tr_idx, te_idx) in enumerate(kf.split(X, events)):
        print(f"--- Training Fold {fold + 1}/{n_splits} ---")

        # 1. Split data (X is a DataFrame, durations/events are numpy arrays)
        X_tr_raw, X_te_raw = X.iloc[tr_idx].copy(), X.iloc[te_idx].copy()
        y_tr, y_te = durations[tr_idx], durations[te_idx]
        e_tr, e_te = events[tr_idx], events[te_idx]

        # 2. Detect and Scale Features
        binary_cols = [c for c in X.columns if set(X[c].dropna().unique()).issubset({0, 1, 0.0, 1.0})]
        cols_to_scale = [c for c in X.columns if c not in binary_cols and not c.startswith(tuple(nominal_cols))]

        scaler = StandardScaler()
        X_tr_scaled = X_tr_raw.copy()
        X_te_scaled = X_te_raw.copy()

        X_tr_scaled[cols_to_scale] = scaler.fit_transform(X_tr_scaled[cols_to_scale]).astype(np.float32)
        X_te_scaled[cols_to_scale] = scaler.transform(X_te_scaled[cols_to_scale]).astype(np.float32)

        y_max = y_tr.max()
        y_tr_scaled = y_tr / y_max

        # 3. Initialize and Train Model
        model = EnSurv(
            latent_dim=None, num_layer=2, generator='concatenated', hidden_dims=[512, 512], apply_linear=True,
            dropout=0.1, noise_dist='uniform', noise_std=1.0, batch_normalization=True,
            device=DEVICE, random_state=seed
        )

        model.fit(
            X_tr_scaled.to_numpy(), y_tr_scaled, e_tr,
            epochs=50, m=5, B=20, verbose=True, patience=10, track_diagnostics=True,
            lambda_energy=0.5, lr=1e-3, km_batch_size=256, batch_size=4096, l1_reg=1e-3, feature_names=X.columns.tolist(),
            domain_constraints={'positive': ['hcvdr_D-R-'], 'negative': ['KDRI', 'CREAT_TRR', 'serum_creatinine', 'center_mean_KDRI', 'first_week_dial']}
        )    

        # 4. Compute Metrics (Unpacking the new returns)
        fold_metrics, t_grid, survival_preds = compute_all_metrics(
            model, y_tr, e_tr, X_te_scaled.to_numpy(), y_te, e_te, y_max=y_max
        )
        metrics_list.append(fold_metrics)

        # --- save this fold's metrics on their own ---
        pd.DataFrame([fold_metrics]).to_csv(f"{OUT_DIR}/EnSurv_Fold{fold+1}_Metrics.csv", index=False)

        # Save S_pred and evaluation arrays for offline metric computation
        # Using np.savez_compressed to efficiently bundle arrays into one file
        np.savez_compressed(
            f"{OUT_DIR}/EnSurv_Fold{fold+1}_Predictions.npz",
            S_pred=survival_preds,
            times_grid=t_grid,
            y_te=y_te,
            e_te=e_te,
            y_tr=y_tr,
            e_tr=e_tr
        )

        # Generate and save calibration plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
        # Generate plots using the function defined earlier
        plot_time_calibration(survival_preds, t_grid, y_te, e_te, target_time=365.25, ax=ax1)
        plot_time_calibration(survival_preds, t_grid, y_te, e_te, target_time=5*365.25, ax=ax2)
    
        plt.tight_layout()
        fig.savefig(f"{OUT_DIR}/EnSurv_Fold{fold+1}_Calibration_Plots.pdf", format='pdf', bbox_inches='tight')
        plt.close(fig) # Prevent memory leaks

        # 5. Extract Explanations
        raw_explain_df = model.explain(feature_names=X.columns.tolist(), y_max=y_max)
        raw_expl_list.append(raw_explain_df)

        _, base_feature_importance, macro_level_importance = analyze_hierarchical_importance(
            raw_explain_df, nominal_cols=nominal_cols, center_covariates=center_covariates
        )
        base_feat_expl_list.append(base_feature_importance)
        macro_expl_list.append(macro_level_importance)

        # --- save this fold's explanations on their own ---
        raw_explain_df.to_csv(f"{OUT_DIR}/EnSurv_Fold{fold+1}_Raw_Explanations.csv", index=False)
        base_feature_importance.to_csv(f"{OUT_DIR}/EnSurv_Fold{fold+1}_BaseFeature_Explanations.csv", index=False)
        macro_level_importance.to_csv(f"{OUT_DIR}/EnSurv_Fold{fold+1}_MacroLevel_Explanations.csv", index=False)

        # --- save model weights + training diagnostics for this fold ---
        torch.save(model.model.state_dict(), f"{OUT_DIR}/EnSurv_Fold{fold+1}_state_dict.pt")
        with open(f"{OUT_DIR}/EnSurv_Fold{fold+1}_history.pkl", "wb") as f:
            pickle.dump(model.history, f)
        if fold == 0:
            fold1_history = model.history
            fold1_model = model

        torch.cuda.empty_cache()
        gc.collect()

    print(f"\nCV Completed in {time.time() - start_total:.2f} seconds.")

    # ==========================================
    # Post-Processing: Metrics
    # ==========================================
    df_metrics_raw = pd.DataFrame(metrics_list)
    df_metrics_final = pd.DataFrame(index=["EnSurv"])

    for col in df_metrics_raw.columns:
        col_mean = df_metrics_raw[col].mean()
        col_std = df_metrics_raw[col].std()
        df_metrics_final[col] = rf"${col_mean:.4f}\pm{col_std:.4f}$"

    print("\nFinal Metrics (Mean ± Std):")
    df_metrics_final.to_csv("EnSurv_3Fold_Results.csv")
    display(df_metrics_final)

    # ==========================================
    # Post-Processing: Explanations (Averaged across folds)
    # ==========================================
    # 1. Raw Explanations
    df_raw_all = pd.concat(raw_expl_list)
    df_raw_mean = df_raw_all.groupby('Feature').mean().reset_index()
    df_raw_mean = df_raw_mean.sort_values(by='Total_Importance_Pct', ascending=False)
    df_raw_mean.to_csv("512_EnSurv_Raw_Explanations_Mean.csv", index=False)

    # 2. Base Feature Explanations
    df_base_all = pd.concat(base_feat_expl_list)
    df_base_mean = df_base_all.groupby(['Base_Feature', 'Covariate_Level']).mean().reset_index()
    df_base_mean = df_base_mean.sort_values(by='Total_Importance_Pct', ascending=False)
    df_base_mean.to_csv("512_EnSurv_BaseFeature_Explanations_Mean.csv", index=False)

    # 3. Macro Level Explanations
    df_macro_all = pd.concat(macro_expl_list)
    df_macro_mean = df_macro_all.groupby('Covariate_Level').mean().reset_index()
    df_macro_mean.to_csv("512_EnSurv_MacroLevel_Explanations_Mean.csv", index=False)

    return {
        'df_metrics_raw': df_metrics_raw,
        'df_metrics_final': df_metrics_final,
        'df_raw_mean': df_raw_mean,
        'df_base_mean': df_base_mean,
        'df_macro_mean': df_macro_mean,
        'fold1_history': fold1_history,
        'fold1_model': fold1_model,
        'model': model,
        'y_max': y_max,
        'X': X,
        'X_te_scaled': X_te_scaled,
        'X_te_raw': X_te_raw,
        'y_tr': y_tr,
        'y_te': y_te,
        'e_te': e_te,
        'OUT_DIR': OUT_DIR,
        'seed': seed,
    }

