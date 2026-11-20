"""Survival metrics and hierarchical feature-importance summaries."""

import numpy as np
from SurvivalEVAL.Evaluator import SurvivalEvaluator


def compute_all_metrics(model, y_tr, e_tr, X_te_scaled, y_te, e_te, y_max):
    r"""Evaluate the survival metrics in the original day-scale domain.

    :param model: Fitted model with predict_survival, trained on durations divided externally by
        y_max.
    :type model: EnSurv or compatible predictor
    :param y_tr: Original, unscaled training follow-up times in days.
    :type y_tr: numpy.ndarray of shape (n_train,)
    :param e_tr: Original training failure indicators.
    :type e_tr: numpy.ndarray of shape (n_train,)
    :param X_te_scaled: Test covariates with the training-fold feature scaling already applied.
    :type X_te_scaled: array-like of shape (n_test, p)
    :param y_te: Original, unscaled test follow-up times in days.
    :type y_te: numpy.ndarray of shape (n_test,)
    :param e_te: Original test failure indicators.
    :type e_te: numpy.ndarray of shape (n_test,)
    :param y_max: Training-fold time divisor used externally before model.fit. The evaluation
        grid is divided by this value before querying the model.
    :type y_max: float

    :returns: Metric dictionary, original-unit evaluation grid, and patient-by-time survival
        predictions.
    :rtype: tuple[dict[str, float], numpy.ndarray, numpy.ndarray]

    The grid contains 100 evenly spaced points up to the maximum of training and
    test follow-up, plus 365.25 and 1826.25 days when supported by that maximum.
    Prediction uses nsample=2000, batch_size=256, random_state=42, and
    allow_extrapolation=True. SurvivalEVAL receives the original unscaled times.
    Metrics include concordance variants, MAE variants, Brier scores, AUPRC,
    KM calibration, and one-/five-year summaries. Horizon-specific
    metrics outside the follow-up maximum are set to NaN.
    """
    global_max_time = max(y_tr.max(), y_te.max())
    
    # Define clinical horizons in days (accounting for leap years)
    t_1yr = 365.25
    t_5yr = 5 * 365.25
    
    # Base grid
    base_grid = np.linspace(0.0, global_max_time, 100)
    
    # Inject exact 1-year and 5-year horizons into the grid if they fall within the domain
    horizons = [t for t in [t_1yr, t_5yr] if t <= global_max_time]
    times_grid = np.unique(np.sort(np.concatenate([base_grid, horizons])))
    
    # Scale the grid down to match the model's trained domain
    scaled_times_grid = times_grid / y_max
    
    # Query the model
    S_pred = model.predict_survival(X_te_scaled, scaled_times_grid, nsample=2000, batch_size=256, random_state=42, allow_extrapolation=True)
    S_pred = np.array(S_pred, dtype=np.float64)
    
    # Feed original unscaled data to the evaluator
    evl = SurvivalEvaluator(
        S_pred,
        times_grid,
        y_te,
        e_te,
        y_tr,
        e_tr,
        predict_time_method="Median"
    )
    
    
    km_cal = evl.km_calibration()
    if isinstance(km_cal, tuple):
        km_cal = km_cal[0]
        
    median_time = np.median(y_te)

    def _get_val(metric_output):
        r"""Unwrap metric outputs whose first tuple entry holds the requested value.

        :param metric_output: Return value from a SurvivalEVAL metric method.
        :type metric_output: object or tuple

        :returns: The first item for tuple outputs, otherwise the input unchanged.
        :rtype: object
        """
        return metric_output[0] if isinstance(metric_output, tuple) else metric_output

    metrics = {
        "Harrell's C": _get_val(evl.concordance(method="Harrell")),
        "Antolini's C": _get_val(evl.concordance_time_dependent(method="Antolini")),
        "Uno's C": _get_val(evl.concordance(method="Uno")),
        "Hinge Error": _get_val(evl.mae(method="Hinge")),
        "Margin Error": _get_val(evl.mae(method="Margin")),
        "Uncensored MAE": _get_val(evl.mae(method="Uncensored")),
        "Plain IBS": _get_val(evl.integrated_brier_score(IPCW_weighted=False)), 
        "IBS-IPCW": _get_val(evl.integrated_brier_score(IPCW_weighted=True)), 
        "AUPRC": _get_val(evl.auprc()),
        "KM-Calibration": km_cal
    }
    
    # 1-Year Horizon Metrics
    if t_1yr <= global_max_time:
        metrics["AUC 1-Yr"] = _get_val(evl.auc(target_time=t_1yr))
        metrics["Brier 1-Yr"] = _get_val(evl.brier_score(target_time=t_1yr))
        metrics["Uno's C 1-Yr"] = _get_val(evl.concordance(method="Uno", tau=t_1yr))
    else:
        metrics["AUC 1-Yr"], metrics["Brier 1-Yr"], metrics["Uno's C 1-Yr"] = np.nan, np.nan, np.nan
        
    # 5-Year Horizon Metrics
    if t_5yr <= global_max_time:
        metrics["AUC 5-Yr"] = _get_val(evl.auc(target_time=t_5yr))
        metrics["Brier 5-Yr"] = _get_val(evl.brier_score(target_time=t_5yr))
        metrics["Uno's C 5-Yr"] = _get_val(evl.concordance(method="Uno", tau=t_5yr))
    else:
        metrics["AUC 5-Yr"], metrics["Brier 5-Yr"], metrics["Uno's C 5-Yr"] = np.nan, np.nan, np.nan
        
    return metrics, times_grid, S_pred


def analyze_hierarchical_importance(explain_df, nominal_cols, center_covariates):
    r"""Aggregate dummy-level structural importance by base feature and covariate level.

    :param explain_df: EnSurv.explain output containing Feature and Total_Importance_Pct.
    :type explain_df: pandas.DataFrame
    :param nominal_cols: Original nominal feature names used to recognize their one-hot encoded
        prefixes.
    :type nominal_cols: sequence[str]
    :param center_covariates: Base feature names labeled Center-Level; other features are
        labeled Patient-Level. These are the original notebook labels, including REGION-based
        covariates.
    :type center_covariates: sequence[str]

    :returns: Annotated dummy-level table, summed base-feature importance, and summed
        covariate-level importance rounded to two decimals.
    :rtype: tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]
    """
    df = explain_df.copy()
    
    # 1. Map One-Hot Encoded features back to their original base name
    def get_base_feature(feat):
        r"""Map a nominal dummy-column name to the first matching base-feature prefix.

        :param feat: Feature name from the explanation table.
        :type feat: str

        :returns: Matching nominal base name, or the original feature name when no prefix matches.
        :rtype: str
        """
        for nom in nominal_cols:
            if feat.startswith(nom + '_'):
                return nom
        return feat  # If not OHE, return original name
        
    df['Base_Feature'] = df['Feature'].apply(get_base_feature)
    
    # 2. Map each Base Feature to its hierarchical level
    df['Covariate_Level'] = df['Base_Feature'].apply(
        lambda x: 'Center-Level' if x in center_covariates else 'Patient-Level'
    )
    
    # 3. Aggregation 1: Overall Base Feature Importance
    # We sum the network's capacity allocation across all dummy variables
    feature_summary = df.groupby(['Base_Feature', 'Covariate_Level'])['Total_Importance_Pct'].sum().reset_index()
    feature_summary = feature_summary.sort_values(by='Total_Importance_Pct', ascending=False).reset_index(drop=True)
    
    # 4. Aggregation 2: Center vs. Patient Macro Split
    level_summary = df.groupby('Covariate_Level')['Total_Importance_Pct'].sum().reset_index()
    level_summary['Total_Importance_Pct'] = level_summary['Total_Importance_Pct'].round(2)
    
    return df, feature_summary, level_summary

