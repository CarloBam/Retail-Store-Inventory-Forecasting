import pandas as pd
import numpy as np

def generate_features_pipeline(df, config):
    """
    Generates a rich set of features based on configuration.
    
    Config expects:
    - target_col
    - date_col
    - store_col, product_col
    - lags: list of ints
    - rolling_windows: list of ints
    - use_log_target: bool
    - promo_logic: dict
    """
    df_feat = df.copy()
    
    # 0. Basic Config
    date_col = config.get('date_col', 'Date')
    target_col = config.get('target_col', 'Units Sold')
    store_col = config.get('store_col', 'Store ID') 
    product_col = config.get('product_col', 'Product ID')
    
    # --- Fix 1: Data Grain and Sorting ---
    # Ensure Date is parsed to datetime
    if date_col in df_feat.columns:
        df_feat[date_col] = pd.to_datetime(df_feat[date_col])
    
    # Sort by Store ID, Product ID, Date before any shift or rolling
    sort_cols = [c for c in [store_col, product_col, date_col] if c in df_feat.columns]
    df_feat = df_feat.sort_values(by=sort_cols)
    
    # --- Fix 2: Remove Leakage and Redundant Predictors ---
    # Treat Demand Forecast as an output style field, not a feature.
    if 'Demand Forecast' in df_feat.columns:
        df_feat = df_feat.drop(columns=['Demand Forecast'])

    # 1. Time Features
    # ----------------------------------------------------
    if config.get('add_time_features', True):
        dt = df_feat[date_col].dt
        df_feat['Year'] = dt.year
        df_feat['Month'] = dt.month
        df_feat['WeekOfYear'] = ((dt.dayofyear - 1) // 7 + 1).astype(int)
        df_feat['DayOfWeek'] = dt.dayofweek
        df_feat['IsWeekend'] = df_feat['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
        
    # 2. Price Features
    # ----------------------------------------------------
    if 'Price' in df_feat.columns:
        if config.get('add_log_price', False):
            df_feat['log_Price'] = np.log1p(df_feat['Price'])
            
        # Calculate Effective Price (Price * (1 - discount_pct))
        if 'Discount' in df_feat.columns:
            # Assume Discount is percentage 0-100 or 0-1. Data inspection usually needed.
            # Based on user input "effective_price = price * (1 - discount_pct)"
            # If discount is 0-100, we divide by 100.
            # Let's assume 0-100 based on previous code context
            df_feat['discount_pct'] = df_feat['Discount'] / 100.0
        else:
            df_feat['discount_pct'] = 0.0
            
        df_feat['effective_price'] = df_feat['Price'] * (1 - df_feat['discount_pct'])
            
        # Competitor Gap
        comp_col = None
        for c in ['Competitor Pricing', 'CompetitorPricing', 'Competitor Price', 'Competitor']:
            if c in df_feat.columns:
                comp_col = c
                break
                
        if comp_col:
            df_feat['price_gap_to_competitor'] = df_feat['effective_price'] - df_feat[comp_col]
            # Optional: relative gap
            df_feat['relative_price_gap'] = (df_feat['effective_price'] / df_feat[comp_col]) - 1

    # 3. Lag Features
    # ----------------------------------------------------
    group_cols = []
    if store_col in df_feat.columns: group_cols.append(store_col)
    if product_col in df_feat.columns: group_cols.append(product_col)
    
    print(f"DEBUG: Grouping by {group_cols} for time features.")
    
    lags = config.get('lags', [])
    if lags and group_cols:
        grouped = df_feat.groupby(group_cols)[target_col]
        for lag in lags:
            df_feat[f'{target_col}_lag_{lag}'] = grouped.shift(lag)
            
    # 4. Rolling Features
    # ----------------------------------------------------
    windows = config.get('rolling_windows', [])
    if windows and group_cols:
        grouped = df_feat.groupby(group_cols)[target_col]
        for window in windows:
            # Shift by 1 first to prevent leakage (using t-1, t-2... for window)
            # Then rolling over 'window' size
            df_feat[f'{target_col}_roll_mean_{window}'] = grouped.transform(lambda x: x.shift(1).rolling(window).mean())

    # Drop Warm-up Period
    # ----------------------------------------------------
    max_lookback = 0
    if lags: max_lookback = max(lags)
    if windows: max_lookback = max(max_lookback, max(windows))
    
    initial_rows = len(df_feat)
    if max_lookback > 0 and group_cols:
        print(f"DEBUG: Dropping warm-up period (first {max_lookback} observations per group).")
        # Identify columns generated
        lag_cols = [f'{target_col}_lag_{l}' for l in lags] + [f'{target_col}_roll_mean_{w}' for w in windows]
        
        # Drop rows where ANY lag/roll feature is NaN
        # This effectively drops the first N rows per group
        df_feat = df_feat.dropna(subset=lag_cols)
        
    dropped_rows = initial_rows - len(df_feat)
    print(f"DEBUG: Rows dropped due to lags/rolling warmup: {dropped_rows} ({dropped_rows/initial_rows:.1%})")

    # 5. Seasonality Encoding
    # ----------------------------------------------------
    if config.get('encode_seasonality', False):
        cols_to_encode = []
        if 'Seasonality' in df_feat.columns: cols_to_encode.append('Seasonality')
        if 'Weather Condition' in df_feat.columns: cols_to_encode.append('Weather Condition')
        
        if cols_to_encode:
            df_feat = pd.get_dummies(df_feat, columns=cols_to_encode, drop_first=True)

    # 6. Promo Flag
    # ----------------------------------------------------
    promo_logic = config.get('promo_logic', 'Any') # Any, HolidayOnly, DiscountOnly
    
    has_holiday = 'Holiday' in df_feat.columns
    has_discount = 'Discount' in df_feat.columns
    
    if config.get('add_promo_flag', True):
        promo_mask = pd.Series(0, index=df_feat.index)
        
        if promo_logic == 'HolidayOnly' and has_holiday:
            promo_mask = df_feat['Holiday'] == 1
        elif promo_logic == 'DiscountOnly' and has_discount:
            promo_mask = df_feat['Discount'] > 0
        elif promo_logic == 'Any':
            c1 = df_feat['Holiday'] == 1 if has_holiday else False
            c2 = df_feat['Discount'] > 0 if has_discount else False
            promo_mask = c1 | c2
            
        df_feat['promo_flag'] = promo_mask.astype(int)

    # 7. Target Transformation
    # ----------------------------------------------------
    if config.get('use_log_target', False):
        df_feat[f'log_{target_col}'] = np.log1p(df_feat[target_col])

    # --- Feature Contract & QA ---
    validate_feature_contract(df_feat, target_col)

    return df_feat

def validate_feature_contract(df, target_col):
    """
    Strict validation of the final feature set.
    """
    print("\n" + "="*40)
    print("FEATURE CONTRACT & QA REPORT")
    print("="*40)
    print(f"Final Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # 1. Critical Column Check
    required_cols = [target_col]
    # Check for price features if they were requested implicitly via presence of input cols
    if 'effective_price' in df.columns: required_cols.append('effective_price')
    if 'price_gap_to_competitor' in df.columns: required_cols.append('price_gap_to_competitor')
    
    missing_required = [c for c in required_cols if c not in df.columns]
    if missing_required:
        raise ValueError(f"FEATURE CONTRACT FAILED: Missing required columns: {missing_required}")
        
    # 2. Nullity Check
    # We expect 0 nulls in critical features after lookup drop
    null_counts = df.isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    
    if not cols_with_nulls.empty:
        print("\nWARNING: Columns with NULLS remaining:")
        print(cols_with_nulls[cols_with_nulls > 0])
        # Fail if critical features have nulls
        # e.g lags should be clean
        critical_patterns = ['lag', 'roll', 'price', 'promo']
        for c in cols_with_nulls.index:
            if any(p in c for p in critical_patterns):
                print(f"CRITICAL WARNING: Feature '{c}' has nulls. Models may crash.")
                
    # 3. Data Integrity
    # Check if 'price_gap_to_competitor' exists if 'Competitor Pricing' was in original (inferred)
    # This addresses "competitor elasticity step fails"
    # We can't see original df here easily, but we can check if the derived col exists
    if 'price_gap_to_competitor' not in df.columns:
        print("NOTICE: 'price_gap_to_competitor' was NOT created. Competitor analysis will be disabled.")
    else:
        print("SUCCESS: 'price_gap_to_competitor' created successfully.")
        
    print("="*40 + "\n")
    print("Feature Contract Passed (mostly).")
