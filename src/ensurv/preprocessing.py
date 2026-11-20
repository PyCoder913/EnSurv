import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


def preprocess_df(df, duration_col, event_col, drop_cols=None, nominal_cols=None, ordinal_cols=None):
    r"""Impute and encode a survival dataframe using the preprocessing procedure.

    :param df: Source dataframe containing covariates and the two outcome columns. The function
        copies it before processing.
    :type df: pandas.DataFrame
    :param duration_col: Observed-time column. Rows with nonpositive or missing duration are
        excluded.
    :type duration_col: str
    :param event_col: Failure-indicator column, converted to integers for the returned outcomes.
    :type event_col: str
    :param drop_cols: Columns removed before preprocessing; missing names are ignored.
    :type drop_cols: list[str] or None, optional
    :param nominal_cols: Explicit unordered categorical covariates. Nonnumeric covariates are
        also detected automatically. Nominal features are one-hot encoded with the first level
        dropped.
    :type nominal_cols: list[str] or None, optional
    :param ordinal_cols: Explicit ordered categorical covariates, retained as integers. Numeric
        binary covariates are also detected automatically. Nominal classification takes
        precedence on overlap.
    :type ordinal_cols: list[str] or None, optional

    :returns: Encoded features, durations, event indicators, encoded column names, and the
        imputed dataframe before final nominal encoding.
    :rtype: tuple[pandas.DataFrame, numpy.ndarray, numpy.ndarray, list[str], pandas.DataFrame]

    Remaining numeric features receive median imputation. Missing categories are
    predicted by a uniform-weight KD-tree KNeighborsClassifier with up to 20
    neighbors and n_jobs=15. Its distance space uses standardized temporary
    one-hot features, excluding the target category's dummy columns.

    This function fits imputation and encoding on the supplied dataframe.
    """
    df = df.copy()
    
    if drop_cols:
        df = df.drop(columns=drop_cols, errors='ignore')
        
    df = df[df[duration_col] > 0].copy()
    
    durations = df[duration_col].astype(float).values
    events = df[event_col].astype(int).values
    X = df.drop(columns=[duration_col, event_col])
    
    if nominal_cols is None: nominal_cols = []
    if ordinal_cols is None: ordinal_cols = []
        
    # 1. Auto-detect string/object columns (Treat as nominal)
    auto_nominal_cols = X.select_dtypes(exclude=['number']).columns.tolist()
    
    # 2. Auto-detect binary numeric columns (Treat as ordinal so they aren't one-hot encoded)
    binary_cols = [col for col in X.select_dtypes(include=['number']).columns 
                   if set(X[col].dropna().unique()).issubset({0, 1, 0.0, 1.0})]
    
    # Combine categorical sources
    final_nominal = list(set(nominal_cols + auto_nominal_cols))
    final_nominal = [col for col in final_nominal if col in X.columns]
    
    final_ordinal = list(set(ordinal_cols + binary_cols))
    # Ensure no overlap (nominal preference if conflict)
    final_ordinal = [col for col in final_ordinal if col in X.columns and col not in final_nominal]
    
    all_cat_cols = final_nominal + final_ordinal
    
    # Numerical columns are strictly whatever is left over
    num_cols = [col for col in X.columns if col not in all_cat_cols]
    
    # 1. Numerical Imputation: Median
    if len(num_cols) > 0:
        X[num_cols] = X[num_cols].fillna(X[num_cols].median())
        
    # 2. Categorical Imputation: KNeighborsClassifier (sklearn KD-Tree)
    # applies to both nominal and ordinal to fill missing values
    if len(all_cat_cols) > 0:
        cols_with_nans = [col for col in all_cat_cols if X[col].isna().any()]
        
        if cols_with_nans:
            X_base = X.copy()
            
            # Temporarily fill NaNs with mode for the distance space
            for c in all_cat_cols:
                if X_base[c].isna().any():
                    X_base[c] = X_base[c].fillna(X_base[c].mode()[0])
            
            # One-hot encode all for the KNN distance computation
            X_base_ohe = pd.get_dummies(X_base, columns=all_cat_cols, drop_first=True, prefix_sep='_knn_')
            X_base_ohe = X_base_ohe.loc[:, ~X_base_ohe.columns.duplicated()]
            
            scaler = StandardScaler()
            X_base_scaled = scaler.fit_transform(X_base_ohe)
            ohe_columns = X_base_ohe.columns.tolist()

            for col in cols_with_nans:
                missing_mask = X[col].isna()
                
                prefix = f"{col}_knn_"
                keep_indices = [i for i, c in enumerate(ohe_columns) if not c.startswith(prefix)]
                
                X_features_scaled = X_base_scaled[:, keep_indices]
                X_ref = X_features_scaled[~missing_mask]
                X_query = X_features_scaled[missing_mask]
                
                y_ref = X.loc[~missing_mask, col].astype(str).values
                
                k = min(20, X_ref.shape[0])
                knn = KNeighborsClassifier(n_neighbors=k, weights='uniform', algorithm='kd_tree', n_jobs=15)
                knn.fit(X_ref, y_ref)
                
                preds = knn.predict(X_query)
                X.loc[missing_mask, col] = preds
        
        # Force Ordinal/Binary to integers
        for c in final_ordinal:
            # Cast via float first to safely handle strings like "1.0"
            X[c] = X[c].astype(float).astype(int)
            
        # Create imputed original dataframe before OHE
        df_imputed = X.copy()
        df_imputed[duration_col] = df[duration_col]
        df_imputed[event_col] = df[event_col]
        df_imputed = df_imputed[[c for c in df.columns if c in df_imputed.columns]]

        # 1. One-hot encode only nominals
        if len(final_nominal) > 0:
            cat_data = pd.get_dummies(X[final_nominal].astype(str), drop_first=True, dtype='int8')
            cat_data = cat_data.loc[:, ~cat_data.columns.duplicated()]
        else:
            cat_data = pd.DataFrame(index=X.index)
        
        # 2. Combine continuous numerics AND integer ordinals/binaries
        combined_num_cols = num_cols + final_ordinal
        num_data = X[combined_num_cols].loc[:, ~X[combined_num_cols].columns.duplicated()]

        # 3. Concatenate side-by-side
        X_proc = pd.concat([num_data, cat_data], axis=1)
        
    else:
        df_imputed = X.copy()
        df_imputed[duration_col] = df[duration_col]
        df_imputed[event_col] = df[event_col]
        df_imputed = df_imputed[[c for c in df.columns if c in df_imputed.columns]]

        X_proc = X[num_cols].reset_index(drop=True)
        
    all_cols = X_proc.columns.tolist()
    return X_proc, durations, events, all_cols, df_imputed


def calculate_entropy(series):
    # Calculate the frequency (probability) of each unique protocol in the group
    r"""Compute base-two Shannon entropy of observed protocol signatures.

    :param series: Protocol signatures for one region. Frequencies follow pandas
        value_counts(normalize=True), which excludes missing values by default.
    :type series: pandas.Series

    :returns: Entropy in bits; a group with one distinct signature has zero entropy.
    :rtype: numpy scalar

    The calculation is :math:`H=-\sum_k p_k\log_2 p_k`, where :math:`p_k` is the within-group
    relative frequency.
    """
    p = series.value_counts(normalize=True).values
    # Compute Shannon entropy: -sum(p * log2(p))
    return -np.sum(p * np.log2(p))


def add_region_features(df):
    # Leave-One-Out (LOO) Center DGF Rate: The variable first_week_dial is the standard clinical proxy for Delayed Graft Function (DGF). 
    # Calculate the center's DGF rate, excluding the current patient to prevent data leakage. 
    r"""
    :param df: Cohort with REGION, first_week_dial, immuno_cat1 through immuno_cat6, KDRI,
        pra_final, LOS, and lclus_size columns.
    :type df: pandas.DataFrame

    :returns: The same dataframe with loo_center_dgf_rate, center_immuno_entropy,
        center_mean_KDRI, center_mean_PRA, and operational_strain_ratio columns.
    :rtype: pandas.DataFrame

    DGF rates subtract the patient's own contribution from the region sum and
    count. Entropy uses combined immunosuppression signatures, acuity features use
    regional means, and operational strain divides regional median LOS by
    ``lclus_size``. Missing/undefined quantities use the notebook's original
    cohort-level fallbacks. All groups are formed by REGION.
    """
    # 1. Calculate the total sum of DGF events per region and broadcast to all rows
    region_dgf_sums = df.groupby('REGION')['first_week_dial'].transform('sum')

    # 2. Calculate the total number of patients per region
    region_counts = df.groupby('REGION')['first_week_dial'].transform('count')

    # 3. Calculate LOO rate: (Total Region Sum - Patient's Own Value) / (Region Count - 1)
    df['loo_center_dgf_rate'] = (region_dgf_sums - df['first_week_dial']) / (region_counts - 1)

    # 4. Handle Edge Case: If a region only has 1 patient, the denominator is 0 (yielding NaN).
    # We fill these rare cases with the global DGF mean to prevent missing values.
    global_dgf_mean = df['first_week_dial'].mean()
    df['loo_center_dgf_rate'] = df['loo_center_dgf_rate'].fillna(global_dgf_mean)



    # Immunosuppression Protocol Entropy:
    # Calculate the Shannon entropy of the combinatorial occurrences of immuno_cat1 through immuno_cat6 within each center.
    immuno_cols = ['immuno_cat1', 'immuno_cat2', 'immuno_cat3', 
                   'immuno_cat4', 'immuno_cat5', 'immuno_cat6']

    # 2. Combine the binary flags into a single string signature (e.g., "100110")
    # We temporarily fill NaNs with 0 to prevent string concatenation errors
    protocol_signatures = df[immuno_cols].fillna(0).astype(int).astype(str).agg(''.join, axis=1)

    # Group by cluster (REGION), apply entropy, and broadcast back to patient rows
    df['center_immuno_entropy'] = protocol_signatures.groupby(df['REGION']).transform(calculate_entropy)

    # Handle edge case: if a region has only 1 patient, entropy is 0 (no variance)
    df['center_immuno_entropy'] = df['center_immuno_entropy'].fillna(0)

    # Center-Level Patient Acuity Index (Mean KDRI / Mean PRA):
    # Calculate the center's median or mean KDRI and PRA_final.
    # 1. Calculate the center's mean KDRI (proxy for organ marginality tolerance)
    df['center_mean_KDRI'] = df.groupby('REGION')['KDRI'].transform('mean')

    # 2. Calculate the center's mean PRA (proxy for highly sensitized patient volume)
    df['center_mean_PRA'] = df.groupby('REGION')['pra_final'].transform('mean')

    # 3. Handle edge cases: If an entire region has missing values, fallback to global mean
    df['center_mean_KDRI'] = df['center_mean_KDRI'].fillna(df['KDRI'].mean())
    df['center_mean_PRA'] = df['center_mean_PRA'].fillna(df['pra_final'].mean())

    # Operational Strain Ratio:
    # The ratio of the center's median LOS (Length of Stay) to its lclus_size.
    # 1. Calculate the center's median Length of Stay (LOS)
    # Median is used to prevent extreme outlier patients from skewing the center's baseline
    center_median_los = df.groupby('REGION')['LOS'].transform('median')

    # 2. Calculate the Operational Strain Ratio
    df['operational_strain_ratio'] = center_median_los / df['lclus_size']

    # 3. Handle Division by Zero Edge Cases
    # If a region has exactly 1 patient, lclus_size (which is np.log(1)) will be 0.
    # Dividing by 0 yields infinity (np.inf). We replace these with NaN and fill them.
    df['operational_strain_ratio'] = df['operational_strain_ratio'].replace([np.inf, -np.inf], np.nan)

    # Fill any NaNs (caused by division by 0 or totally missing LOS) with the global median ratio
    df['operational_strain_ratio'] = df['operational_strain_ratio'].fillna(df['operational_strain_ratio'].median())
    return df


def prepare_unos_dataset(df):
    r"""Prepare the notebook patient and region covariates for the UNOS analysis.

    :param df: Loaded kidney cohort with the columns required by add_region_features, the
        selected patient/region covariates, GTIME_KI, and GSTATUS_KI. Region features are added
        to this dataframe in place.
    :type df: pandas.DataFrame

    :returns: Encoded features, durations, events, encoded feature names, original
        nominal-column names, and original region-level covariate names.
    :rtype: tuple[pandas.DataFrame, numpy.ndarray, numpy.ndarray, list[str], list[str], list[str]]
    """
    df = add_region_features(df)

    covariates = ['hcvdr', 'age_n', 'LOS', 'bmi_grouping', 'serum_creatinine', 'perip_vasc_new', 'CREAT_TRR',
           'KDRI', 'TRTREJ1Y_KI', 'prev_tx_num', 'dialysis_at_txp',
           'first_week_dial', 'genderd_num', 'gender_num', 'aa',
           'blood_typ', 'blood_donTyp', 'pra_final', 'immuno_cat1', 'immuno_cat2',
           'immuno_cat3', 'immuno_cat4', 'immuno_cat5', 'immuno_cat6']
       
    # Define center covariates
    center_covariates = ['ctr_malePerc', 'ctr_donPerc', 'REGION', 'lclus_size',
                        'loo_center_dgf_rate', 'center_immuno_entropy', 'center_mean_KDRI', 'center_mean_PRA',
                        'operational_strain_ratio'] 

    duration = 'GTIME_KI'
    event = 'GSTATUS_KI'
    columns_to_keep = covariates + center_covariates + [duration, event]

    # Subset dataframe
    df = df[columns_to_keep]

    # 1. Define nominal variables that have no mathematical order (will be One-Hot Encoded)
    nominal_cols = ['hcvdr', 'blood_typ', 'blood_donTyp', 'REGION']

    # 2. Define ordinal variables that have mathematical order (will remain integers)
    ordinal_cols = ['bmi_grouping', 'prev_tx_num']

    # Run the updated preprocessor
    X, durations, events, feat_names, _ = preprocess_df(
        df, 
        duration, 
        event, 
        nominal_cols=nominal_cols, 
        ordinal_cols=ordinal_cols
    )
    return X, durations, events, feat_names, nominal_cols, center_covariates

