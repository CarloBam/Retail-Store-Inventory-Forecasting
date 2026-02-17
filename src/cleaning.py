import pandas as pd
import numpy as np
from scipy import stats

def apply_cleaning_pipeline(df, config):
    """
    Applies a cleaning pipeline based on the provided configuration dictionary.
    Returns:
        - cleaned dataframe
        - audit log (list of strings)
    """
    df_clean = df.copy()
    audit_log = []
    
    # 1. Type Enforcement
    # ----------------------------------------------------------------
    audit_log.append("--- Type Enforcement ---")
    numeric_cols = [
        'Inventory Level', 'Units Sold', 'Units Ordered', 'Demand Forecast', 
        'Price', 'Discount', 'Competitor Pricing'
    ]
    
    for col in numeric_cols:
        if col in df_clean.columns:
            # Force numeric, coerce errors to NaN
            # Check if conversion is needed
            if not pd.api.types.is_numeric_dtype(df_clean[col]):
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                audit_log.append(f"Coerced '{col}' to numeric.")
    
    binary_cols = ['Holiday', 'Promotion']
    for col in binary_cols:
        if col in df_clean.columns:
            # Ensure 0 or 1
            # First clean to numeric
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0).astype(int)
            # Clip to valid range just in case
            df_clean[col] = df_clean[col].apply(lambda x: 1 if x > 0 else 0)
            audit_log.append(f"Enforced binary type (0/1) on '{col}'.")

    # 2. Missing Value Handling
    # ----------------------------------------------------------------
    audit_log.append("--- Missing Value Handling ---")
    target_col = config.get('target_col', 'Units Sold')
    
    # Drop rows with missing target?
    if config.get('drop_missing_target', False) and target_col in df_clean.columns:
        initial_rows = len(df_clean)
        df_clean = df_clean.dropna(subset=[target_col])
        dropped = initial_rows - len(df_clean)
        if dropped > 0:
            audit_log.append(f"Dropped {dropped} rows with missing '{target_col}'.")
            
    # Imputation
    impute_method_num = config.get('impute_numeric', 'None') # None, Median (Global), Median (Product)
    
    numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if df_clean[col].isnull().sum() > 0:
            if impute_method_num == 'Median (Global)':
                fill_val = df_clean[col].median()
                df_clean[col] = df_clean[col].fillna(fill_val)
                audit_log.append(f"Imputed missing '{col}' with global median ({fill_val:.2f}).")
            elif impute_method_num == 'Zero':
                df_clean[col] = df_clean[col].fillna(0)
                audit_log.append(f"Imputed missing '{col}' with 0.")
            # Note: Product-level imputation is complex if Product column missing, skipping for simplicity in this artifact
            
    # Categorical Mode
    if config.get('impute_categorical_mode', False):
        cat_columns = df_clean.select_dtypes(exclude=[np.number, 'datetime']).columns
        for col in cat_columns:
            if df_clean[col].isnull().sum() > 0:
                mode_val = df_clean[col].mode()[0]
                df_clean[col] = df_clean[col].fillna(mode_val)
                audit_log.append(f"Imputed missing '{col}' with mode ('{mode_val}').")

    # 3. Outlier Handling
    # ----------------------------------------------------------------
    audit_log.append("--- Outlier Handling ---")
    outlier_cols = ['Units Sold', 'Price']
    method = config.get('outlier_method', 'None') # None, Winsorize, Clip (Z-score)
    
    for col in outlier_cols:
        if col in df_clean.columns:
            if method == 'Winsorize':
                lower = config.get('winsorize_lower', 0.05)
                upper = config.get('winsorize_upper', 0.95)
                # Winsorize returns an array, need to assignment back carefully
                # We use scipy.stats.mstats.winsorize but it changes data in place or returns array
                # Let's use quantile based clipping manually for clarity
                lower_cap = df_clean[col].quantile(lower)
                upper_cap = df_clean[col].quantile(upper)
                
                # Check if we actually modify data
                mask_lower = df_clean[col] < lower_cap
                mask_upper = df_clean[col] > upper_cap
                if mask_lower.any() or mask_upper.any():
                    df_clean[col] = df_clean[col].clip(lower=lower_cap, upper=upper_cap)
                    audit_log.append(f"Winsorized '{col}' to {lower:.0%}-{upper:.0%} percentiles.")
                    
            elif method == 'Clip (Z-score)':
                threshold = config.get('z_threshold', 3.0)
                z_scores = stats.zscore(df_clean[col], nan_policy='omit')
                abs_z = np.abs(z_scores)
                
                # Identify outliers
                outliers = abs_z > threshold
                if outliers.sum() > 0:
                    # Clip values to the threshold boundary
                    mean = df_clean[col].mean()
                    std = df_clean[col].std()
                    lower_bound = mean - threshold * std
                    upper_bound = mean + threshold * std
                    
                    df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
                    audit_log.append(f"Clipped '{col}' outliers outside Z-score {threshold} ({outliers.sum()} affected).")

    # 4. Duplicates
    # ----------------------------------------------------------------
    if config.get('remove_duplicates', False):
        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        dropped = initial_rows - len(df_clean)
        if dropped > 0:
            audit_log.append(f"Removed {dropped} duplicate rows.")

    return df_clean, audit_log

def get_missingness_table(df):
    """
    Returns a dataframe styled for missingness heatmapping.
    """
    missing = df.isnull().sum()
    percent = 100 * df.isnull().sum() / len(df)
    dtype = df.dtypes
    
    table = pd.concat([missing, percent, dtype], axis=1)
    table.columns = ['Missing Values', '% Missing', 'Dtype']
    table = table[table['Missing Values'] > 0].sort_values('% Missing', ascending=False)
    
    return table
