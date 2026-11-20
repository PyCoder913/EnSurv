"""Post-CV plotting blocks."""

import gc
import glob
import numpy as np
import torch
import matplotlib.pyplot as plt
from .utils import set_journal_style, plot_pooled_calibration
from .diagnostics import EnSurvDiagnostics


def plot_pooled_cv_calibration(output_dir="new_ensurv_cv_outputs"):
    r"""Load saved fold predictions and reproduce notebook pooled calibration plots.

    :param output_dir: Directory containing EnSurv_Fold*_Predictions.npz; default
        new_ensurv_cv_outputs.
    :type output_dir: str or path-like, optional

    :returns: Pooled one-/five-year calibration figure, also saved as
        Final_Pooled_Calibration_Plots.pdf in the current working directory.
    :rtype: matplotlib.figure.Figure

    Preserves cell 9 aggregation, horizon handling, style, filename, and plt.show call. No folds
    are retrained.
    """
    set_journal_style()

    # 1. Define horizons and storage for pooled data
    horizons = {
        "1-Year": 365.25,
        "5-Year": 5 * 365.25
    }

    pooled_data = {
        t_name: {'preds': [], 'y': [], 'e': []} 
        for t_name in horizons.keys()
    }

    # 2. Find and load all .npz files
    OUT_DIR = output_dir
    npz_files = sorted(glob.glob(f"{OUT_DIR}/EnSurv_Fold*_Predictions.npz"))

    print(f"Found {len(npz_files)} fold prediction files. Aggregating...")

    for file_path in npz_files:
        # Load compressed arrays
        data = np.load(file_path)
        t_grid = data['times_grid']
        S_pred = data['S_pred']
        y_te = data['y_te']
        e_te = data['e_te']
    
        # Extract data for each clinical horizon
        for t_name, t_val in horizons.items():
            if t_val > t_grid.max():
                continue # Skip if this fold's domain is shorter than the horizon
            
            # Find exact index for target time
            time_idx = np.argmin(np.abs(t_grid - t_val))
        
            # Append fold predictions and ground truth to pooled lists
            pooled_data[t_name]['preds'].extend(S_pred[:, time_idx])
            pooled_data[t_name]['y'].extend(y_te)
            pooled_data[t_name]['e'].extend(e_te)

    # Convert lists to flat numpy arrays
    for t_name in horizons.keys():
        pooled_data[t_name]['preds'] = np.array(pooled_data[t_name]['preds'])
        pooled_data[t_name]['y'] = np.array(pooled_data[t_name]['y'])
        pooled_data[t_name]['e'] = np.array(pooled_data[t_name]['e'])

    # 3. Define a pooled plotting function (takes 1D arrays directly)

    # 4. Generate the final combined plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    plot_pooled_calibration(
        pooled_data["1-Year"]['preds'], 
        pooled_data["1-Year"]['y'], 
        pooled_data["1-Year"]['e'], 
        target_time=horizons["1-Year"], 
        title="Pooled 1-Year Calibration", 
        ax=ax1
    )

    plot_pooled_calibration(
        pooled_data["5-Year"]['preds'], 
        pooled_data["5-Year"]['y'], 
        pooled_data["5-Year"]['e'], 
        target_time=horizons["5-Year"], 
        title="Pooled 5-Year Calibration", 
        ax=ax2
    )

    plt.tight_layout()
    plt.savefig("Final_Pooled_Calibration_Plots.pdf", format='pdf', bbox_inches='tight')
    plt.show()
    return fig


def run_notebook_visual_diagnostics(fold1_model, y_max, X_te_scaled, X_te_raw, y_tr, y_te, e_te, output_dir="new_ensurv_cv_outputs", seed=42):
    r"""Reproduce notebook cell 12 post-testing plots with its original inputs.

    :param fold1_model: First-fold fitted model retained by the CV function. The original
        notebook uses this model with the later variables passed below.
    :type fold1_model: EnSurv
    :param y_max: External time divisor retained from the final CV fold.
    :type y_max: float
    :param X_te_scaled: Final-fold scaled test covariates.
    :type X_te_scaled: pandas.DataFrame
    :param X_te_raw: Final-fold test covariates before scaling, used as a region-label fallback.
    :type X_te_raw: pandas.DataFrame
    :param y_tr: Final-fold original training follow-up times in days.
    :type y_tr: numpy.ndarray
    :param y_te: Final-fold original test follow-up times in days.
    :type y_te: numpy.ndarray
    :param e_te: Final-fold original test failure indicators.
    :type e_te: numpy.ndarray
    :param output_dir: Existing destination directory for the three original diagnostic PDFs.
    :type output_dir: str or path-like, optional
    :param seed: NumPy seed for selecting density-plot patients; the notebook uses 42. Scatter
        subsampling separately retains seed 67.
    :type seed: int, optional

    The first-fold-model/final-fold-data combination is retained to satisfy source preservation.
    This helper also retains the temporary deterministic-algorithm toggle and all original plot
    calls and filenames. It returns None.
    """
    OUT_DIR = output_dir

    # ==========================================
    # Post-Testing Visual Diagnostics
    # ==========================================
    print("\nGenerating Diagnostic Plots...")
    diagnostics = EnSurvDiagnostics(fold1_model, y_max=y_max)

    X_test_np = X_te_scaled.to_numpy()
    global_max_time = max(y_tr.max(), y_te.max())
    eval_times_unscaled = np.linspace(0, global_max_time, 100)

    # Reconstruct Regions from one-hot encoding
    region_cols = [c for c in X_te_scaled.columns if c.startswith('REGION_')]
    if region_cols:
        test_regions = X_te_scaled[region_cols].idxmax(axis=1).values
    elif 'REGION' in X_te_raw.columns:
        test_regions = X_te_raw['REGION'].values
    else:
        test_regions = np.array(['All Regions'] * len(X_te_scaled))

    # A. Population Calibration Plot (Now a Small Multiples Grid)
    print("  - Plotting Calibration Grid...")
    fig_cal = diagnostics.plot_population_calibration(
        X_test=X_test_np, 
        Y_test=y_te, 
        Delta_test=e_te, 
        times=eval_times_unscaled, 
        center_labels=test_regions
    )
    fig_cal.savefig(f"{OUT_DIR}/plot_calibration_stratified_grid.pdf", bbox_inches='tight')
    plt.close(fig_cal)

    # B. Prediction Scatter (Subsampled to 100 points + Determinism Fix)
    print("  - Plotting Scatter...")
    was_deterministic = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(False)
    try:
        fig_scatter = diagnostics.plot_prediction_scatter(
            X_test=X_test_np, 
            Y_test=y_te, 
            Delta_test=e_te,
            figsize=(8, 8),
            max_points=200,  # Subsample heavily for clarity
            random_seed=67
        )
        fig_scatter.savefig(f"{OUT_DIR}/plot_prediction_scatter.pdf", bbox_inches='tight')
        plt.close(fig_scatter)
    finally:
        torch.use_deterministic_algorithms(was_deterministic)

    # C. Individual Densities (10 Random Patients)
    print("  - Plotting Individual Densities...")
    np.random.seed(seed)
    event_idx = np.where(e_te == 1)[0]
    censor_idx = np.where(e_te == 0)[0]

    # Try to get 5 events and 5 censors (adjust dynamically if a class is rare)
    n_events = min(len(event_idx), 5)
    n_censors = min(len(censor_idx), 10 - n_events)
    n_events = min(len(event_idx), 10 - n_censors)

    sel_events = np.random.choice(event_idx, n_events, replace=False) if n_events > 0 else []
    sel_censors = np.random.choice(censor_idx, n_censors, replace=False) if n_censors > 0 else []
    selected_patients = np.concatenate([sel_events, sel_censors])
    np.random.shuffle(selected_patients) # Shuffle so events and censors are mixed visually

    fig_dens = diagnostics.plot_individual_densities(
        X_test=X_test_np,
        patient_indices=selected_patients,
        true_times=y_te,
        true_events=e_te,
        nsample=5000,
        figsize=(10, 15)  # Increased height significantly to fit 10 distributions cleanly
    )
    fig_dens.savefig(f"{OUT_DIR}/plot_individual_densities.pdf", bbox_inches='tight')
    plt.close(fig_dens)

    # # D. Training Dynamics (In-Model Diagnostics)
    # print("  - Plotting Training Dynamics...")
    # fig_train = model.plot_training_dynamics(feature_names=X.columns.tolist(), figsize=(14, 10))
    # fig_train.savefig(f"{OUT_DIR}/plot_training_dynamics.pdf", bbox_inches='tight')
    # plt.close(fig_train)

    torch.cuda.empty_cache()
    gc.collect()

    print(f"\nFinished. All artifacts saved to '{OUT_DIR}/'.")


def plot_selected_patient_densities(model, y_max, X_te_scaled, y_tr, y_te, e_te, output_dir="new_ensurv_cv_outputs"):
    r"""Reproduce notebook cell 13 densities for its fixed patient positions.

    :param model: Final-fold fitted model retained after CV.
    :type model: EnSurv
    :param y_max: Final-fold external training-time divisor.
    :type y_max: float
    :param X_te_scaled: Final-fold scaled test covariates with at least 59,655 rows to support
        all fixed positions.
    :type X_te_scaled: pandas.DataFrame
    :param y_tr: Final-fold original training follow-up times in days.
    :type y_tr: numpy.ndarray
    :param y_te: Final-fold original test follow-up times in days.
    :type y_te: numpy.ndarray
    :param e_te: Final-fold test failure indicators.
    :type e_te: numpy.ndarray
    :param output_dir: Existing plot destination directory; the original
        plot_individual_densities.pdf is overwritten.
    :type output_dir: str or path-like, optional

    Preserves the ten hard-coded patient positions, global NumPy seed 9, subsequent shuffle, and
    original output filename. This is an optional reproduction helper because those positions
    are cohort-specific. It returns None.
    """
    OUT_DIR = output_dir

    # C. Individual Densities (10 Random Patients)
    diagnostics = EnSurvDiagnostics(model, y_max=y_max)

    X_test_np = X_te_scaled.to_numpy()
    global_max_time = max(y_tr.max(), y_te.max())
    eval_times_unscaled = np.linspace(0, global_max_time, 100)
    print("  - Plotting Individual Densities...")
    np.random.seed(9)
    event_idx = np.where(e_te == 1)[0]
    censor_idx = np.where(e_te == 0)[0]

    n_events = min(len(event_idx), 5)
    n_censors = min(len(censor_idx), 10 - n_events)
    n_events = min(len(event_idx), 10 - n_censors)

    sel_events = np.random.choice(event_idx, n_events, replace=False) if n_events > 0 else []
    sel_censors = np.random.choice(censor_idx, n_censors, replace=False) if n_censors > 0 else []
    selected_patients = np.concatenate([sel_events, sel_censors])
    np.random.shuffle(selected_patients) 

    fig_dens = diagnostics.plot_individual_densities(
        X_test=X_test_np,
        patient_indices=selected_patients,
        true_times=y_te,
        true_events=e_te,
        nsample=5000,
        figsize=(10, 15)  
    )
    fig_dens.savefig(f"{OUT_DIR}/plot_individual_densities.pdf", bbox_inches='tight')
    plt.close(fig_dens)

    torch.cuda.empty_cache()
    gc.collect()

    print(f"\nFinished. All artifacts saved to '{OUT_DIR}/'.")

