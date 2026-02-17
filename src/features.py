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
    store_col = config.get('store_col', 'Store ID') # Adjust based on actual col name inspection if needed
    product_col = config.get('product_col', 'Product ID')
    
    # Ensure date sorted
    df_feat = df_feat.sort_values(by=[date_col])
    
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
            
        # Calculate Effective Price (Price - Discount)
        # Default discount to 0 if missing
        if 'Discount' in df_feat.columns:
            df_feat['discount_pct'] = df_feat['Discount'] / 100.0
        else:
            df_feat['discount_pct'] = 0.0
            
        df_feat['effective_price'] = df_feat['Price'] * (1 - df_feat['discount_pct'])
            
        # Competitor Gap
        # Check for various namings
        comp_col = None
        for c in ['Competitor Pricing', 'CompetitorPricing', 'Competitor Price', 'Competitor']:
            if c in df_feat.columns:
                comp_col = c
                break
                
        if comp_col:
            df_feat['price_gap_to_competitor'] = df_feat['effective_price'] - df_feat[comp_col]

    # 3. Lag Features
    # ----------------------------------------------------
    # We need to respect groups: Store X Product
    # Check if we have grouping columns, otherwise global (not recommended usually)
    group_cols = []
    if store_col in df_feat.columns: group_cols.append(store_col)
    if product_col in df_feat.columns: group_cols.append(product_col)
    
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
            # Shift by 1 to avoid data leakage (predicting today using today's rolling mean including today)
            # Usually rolling features are calculated on shifted data (lag 1)
            # Or we use closed='left' but simple way is shift 1 then rolling
            shifted = grouped.shift(1)
            # Re-grouping might be needed if shift breaks group alignment or use transformation
            # Safe way:
            df_feat[f'{target_col}_roll_mean_{window}'] = grouped.transform(lambda x: x.shift(1).rolling(window).mean())

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
    # "Holiday Promotion" or "Discount" > 0 logic
    # Assume "Holiday" is 0/1 from cleaning
    
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

    return df_feat
