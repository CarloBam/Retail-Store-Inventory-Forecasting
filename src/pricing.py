import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

def prepare_pricing_data(df, config):
    """
    Prepares data for pricing analysis based on config.
    Config options:
    - agg_level: 'Daily', 'Weekly'
    - exclude_promo: bool
    - correlation_threshold: int (discount depth max)
    - product_id: optional (for single product model)
    """
    df_model = df.copy()
    
    # 1. Filter Promos
    if config.get('exclude_promo', False):
        if 'promo_flag' in df_model.columns:
            df_model = df_model[df_model['promo_flag'] == 0]
        elif 'Holiday' in df_model.columns:
             df_model = df_model[df_model['Holiday'] == 0]
             
    # 2. Filter Discount Depth
    max_discount = config.get('max_discount_depth', 100)
    if 'Discount' in df_model.columns:
        df_model = df_model[df_model['Discount'] <= max_discount]
        
    # 3. Product Specific Filter
    if config.get('product_id'):
        prod_col = 'Product ID' if 'Product ID' in df_model.columns else 'Product'
        df_model = df_model[df_model[prod_col] == config['product_id']]
        
    # 4. Aggregation
    date_col = 'Date'
    agg_level = config.get('agg_level', 'Daily')
    
    # Define aggregation dict
    agg_dict = {
        'Units Sold': 'sum',
        'Price': 'mean', # Weighted average ideal but mean is proxy
        'effective_price': 'mean',
        'Discount': 'mean',
        'Competitor Pricing': 'mean'
    }
    # Add dummy/categorical columns to max or first
    for c in ['Category', 'Store ID', 'Store', 'Region', 'Holiday', 'promo_flag']:
        if c in df_model.columns:
            agg_dict[c] = 'first'
            
    # Remove keys not in columns
    agg_dict = {k: v for k, v in agg_dict.items() if k in df_model.columns}
    
    if agg_level == 'Weekly':
        # Resample requires date index
        df_model = df_model.set_index(date_col).resample('W').agg(agg_dict).dropna().reset_index()
    
    # 5. Log Transforms
    # Ensure positive for logs
    # We add small epsilon or just filter > 0
    df_model = df_model[df_model['Units Sold'] > 0]
    
    # Use effective price if available, else Price
    price_col = 'effective_price' if 'effective_price' in df_model.columns else 'Price'
    df_model = df_model[df_model[price_col] > 0]
    
    df_model['log_demand'] = np.log(df_model['Units Sold'])
    df_model['log_price'] = np.log(df_model[price_col])
    
    return df_model

def fit_elasticity_model(df, category_col='Category', fixed_effects=[], robust_cov=False):
    """
    Fits log-log regression.
    Returns summary dataframe of elasticities by Category.
    """
    results = []
    
    # If category column exists, we fit per category for clarity in "Category Level" analysis
    # Alternatively, we could run one big pooled model with interaction terms, 
    # but per-category models are cleaner to read and often preferred in simple business dashboards.
    
    categories = df[category_col].unique() if category_col in df.columns else ['All']
    
    for cat in categories:
        if cat == 'All':
            sub_df = df
        else:
            sub_df = df[df[category_col] == cat]
            
        if len(sub_df) < 10:
            continue
            
        # Formula Construction
        # Basic: log_demand ~ log_price
        formula = "log_demand ~ log_price"
        
        # Controls
        controls = ['promo_flag', 'Holiday', 'discount_pct', 'Month']
        for c in controls:
            if c in sub_df.columns:
                # Check for variance
                if sub_df[c].nunique() > 1:
                    formula += f" + {c}"
        
        # Fixed Effects (Store, Product)
        # Using C() for categorical
        for fe in fixed_effects: 
            # Check column existence and cardinality (avoid exploding model)
            col_name = 'Store_ID' if fe == 'Store' and 'Store_ID' in sub_df.columns else fe
            if col_name in sub_df.columns and sub_df[col_name].nunique() < 50: # Cap for feasibility
                formula += f" + C(Q('{col_name}'))" # Q() handles spaces
                
        try:
            # Fit
            if robust_cov:
                # HC1 or cluster? 'cluster' requires groups
                # Let's default to HC1 for robustness without group complexity in UI unless specified
                model = smf.ols(formula=formula, data=sub_df).fit(cov_type='HC1')
            else:
                model = smf.ols(formula=formula, data=sub_df).fit()
            
            # Extract Elasticity (coeff of log_price)
            elas = model.params['log_price']
            pval = model.pvalues['log_price']
            ci = model.conf_int().loc['log_price']
            
            # Interpretation
            interp = interpret_elasticity(elas, pval)
            
            results.append({
                'Category': cat,
                'Elasticity': elas,
                'P-Value': pval,
                'CI_Lower': ci[0],
                'CI_Upper': ci[1],
                'R-Squared': model.rsquared,
                'Obs': int(model.nobs),
                'Interpretation': interp
            })
            
        except Exception as e:
            # print(f"Error fitting {cat}: {e}")
            pass
            
    return pd.DataFrame(results)

def fit_competitor_model(df):
    """
    Regress units on price gap.
    """
    if 'price_gap_to_competitor' not in df.columns:
        return None, "Feature 'price_gap_to_competitor' missing."
        
    # Formula
    formula = "Q('Units Sold') ~ price_gap_to_competitor"
    
    # Clean data
    df_clean = df.dropna(subset=['Units Sold', 'price_gap_to_competitor'])
    if len(df_clean) < 10:
        return None, "Insufficient data."
        
    try:
        model = smf.ols(formula=formula, data=df_clean).fit()
        return model, None
    except Exception as e:
         return None, str(e)

def interpret_elasticity(coef, pval):
    if pval > 0.05:
        return "Insignificant (No clear price effect)"
    if coef < -1:
        return "Elastic (Revenue increases with Price Cut)"
    elif coef > -1 and coef < 0:
        return "Inelastic (Revenue increases with Price Hike)"
    elif coef > 0:
        return "Positive (Giffen/Veblen-like or Confounded)"
    else:
        return "Unit Elastic"

def calculate_what_if(elasticity, current_price, current_units, pct_change):
    """
    Calculates expected impact.
    pct_change is in percent (e.g. -10 for 10% drop)
    """
    # % Change in Q = Elasticity * % Change in P
    pct_change_decimal = pct_change / 100.0
    expected_pct_change_units = elasticity * pct_change_decimal
    
    new_price = current_price * (1 + pct_change_decimal)
    new_units = current_units * (1 + expected_pct_change_units)
    
    current_rev = current_price * current_units
    new_rev = new_price * new_units
    
    rev_change_pct = ((new_rev - current_rev) / current_rev) * 100 if current_rev != 0 else 0
    
    return {
        'New Price': new_price,
        'New Units': new_units,
        'Units Change %': expected_pct_change_units * 100,
        'Revenue Change %': rev_change_pct
    }
