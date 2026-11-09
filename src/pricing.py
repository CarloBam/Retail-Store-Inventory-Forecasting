import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

def prepare_pricing_data(df, config):
    """
    Prepares pricing data, filters invalid values, and creates log variables.
    Separates prediction logic from structural interpretation.
    """
    df_model = df.copy()
    
    # 1. Filter Promos (Optional sensitivity check)
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
        'Price': 'mean', 
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
    # Log transforms require positive prices and sales; zero-sales rows can reflect stockouts or closures.
    price_col = 'effective_price' if 'effective_price' in df_model.columns else 'Price'
    df_model = df_model[df_model[price_col] > 0]
    
        # Exclude zero-sales rows because log demand is undefined.
    df_model = df_model[df_model['Units Sold'] > 0]
    
    df_model['log_demand'] = np.log(df_model['Units Sold'])
    df_model['log_price'] = np.log(df_model[price_col])
    
    return df_model

def _fit_log_log(sub_df, formula, fixed_effects=[]):
    """
    Helper to fit the model and return stats.
    """
    # Fixed Effects
    for fe in fixed_effects: 
        col_name = 'Store_ID' if fe == 'Store' and 'Store_ID' in sub_df.columns else fe
        if col_name in sub_df.columns and sub_df[col_name].nunique() < 50:
            formula += f" + C(Q('{col_name}'))"
            
    try:
        model = smf.ols(formula=formula, data=sub_df).fit(cov_type='HC1')
        return model
    except:
        return None

def fit_elasticity_model(df, category_col='Category', fixed_effects=[], robust_cov=True):
    """
    Fits log-log regression (Structural Model).
    Returns summary dataframe of elasticities by Category + Segments.
    """
    results = []
    
    categories = df[category_col].unique() if category_col in df.columns else ['All']
    
    for cat in categories:
        if cat == 'All':
            sub_df = df
        else:
            sub_df = df[df[category_col] == cat]
            
        if len(sub_df) < 20: 
            continue
            
        # Base Formula
        base_formula = "log_demand ~ log_price"
        
        # Controls
        controls = ['Holiday', 'Seasonality', 'Month', 'weather_cluster'] # Legacy control candidates; active controls are selected below.
        # Include promotion when available in the overall model.
        overall_formula = base_formula
        control_vars = []
        possible_controls = ['promo_flag', 'Holiday', 'discount_pct', 'Month', 'Seasonality']
        for c in possible_controls:
            if c in sub_df.columns and sub_df[c].nunique() > 1:
                control_vars.append(c)
                overall_formula += f" + {c}"
        
        # 1. Overall Model
        model = _fit_log_log(sub_df, overall_formula, fixed_effects)
        
        if model:
            elas = model.params['log_price']
            pval = model.pvalues['log_price']
            ci = model.conf_int().loc['log_price']
            r2 = model.rsquared
            
            # 2. Segmented Analysis (Promo vs Non-Promo)
            elas_promo = np.nan
            elas_no_promo = np.nan
            
            if 'promo_flag' in sub_df.columns:
                # Non-Promo
                df_no = sub_df[sub_df['promo_flag'] == 0]
                if len(df_no) > 20:
                    # Remove promo_flag from controls for this segment
                    seg_formula = base_formula
                    for c in [x for x in control_vars if x != 'promo_flag']:
                         if df_no[c].nunique() > 1: seg_formula += f" + {c}"
                    
                    m_no = _fit_log_log(df_no, seg_formula, fixed_effects)
                    if m_no: elas_no_promo = m_no.params['log_price']
                
                # Promo
                df_yes = sub_df[sub_df['promo_flag'] == 1]
                if len(df_yes) > 20:
                    seg_formula = base_formula
                    for c in [x for x in control_vars if x != 'promo_flag']:
                         if df_yes[c].nunique() > 1: seg_formula += f" + {c}"
                    
                    m_yes = _fit_log_log(df_yes, seg_formula, fixed_effects)
                    if m_yes: elas_promo = m_yes.params['log_price']
            
            interp = interpret_elasticity(elas, pval)
            
            results.append({
                'Category': cat,
                'Elasticity': elas,
                'P-Value': pval,
                'CI_Lower': ci[0],
                'CI_Upper': ci[1],
                'R-Squared': r2,
                'Obs': int(model.nobs),
                'Interpretation': interp,
                'Elas_No_Promo': elas_no_promo,
                'Elas_Promo': elas_promo,
                'Controls': ", ".join(control_vars)
            })
            
    return pd.DataFrame(results)

def run_placebo_test(df, category_col='Category'):
    """
    Placebo Test: Does Future Price predict Current Demand?
    If significant, implies endogeneity (anticipation or trend confounding).
    """
    results = []
    categories = df[category_col].unique() if category_col in df.columns else ['All']
    
    # Needs date sorting
    if 'Date' in df.columns:
        df = df.sort_values('Date')
        
    for cat in categories:
        sub_df = df if cat == 'All' else df[df[category_col] == cat]
        sub_df = sub_df.copy() # Avoid SettingWithCopy
        
        # Shift prices within product-store series when both identifiers are available.
        if 'Store ID' in sub_df.columns and 'Product ID' in sub_df.columns:
            sub_df['future_log_price'] = sub_df.groupby(['Store ID', 'Product ID'])['log_price'].shift(-1)
        else:
            # Aggregate or single series
            sub_df['future_log_price'] = sub_df['log_price'].shift(-1)
            
        sub_df = sub_df.dropna(subset=['log_demand', 'future_log_price', 'log_price'])
        
        if len(sub_df) < 20: continue
        
        # Regress log_demand ~ log_price + future_log_price + controls
        formula = "log_demand ~ log_price + future_log_price"
        for c in ['Holiday', 'Month', 'promo_flag']:
            if c in sub_df.columns and sub_df[c].nunique() > 1:
                formula += f" + {c}"
                
        try:
            model = smf.ols(formula=formula, data=sub_df).fit(cov_type='HC1')
            
            # Check significance of future_log_price
            placebo_coef = model.params['future_log_price']
            placebo_pval = model.pvalues['future_log_price']
            
            flag = "FAIL" if placebo_pval < 0.05 else "PASS"
            msg = "Significant Future Price Effect (Possible Endogeneity)" if placebo_pval < 0.05 else "No Future Price Effect detected."
            
            results.append({
                'Category': cat,
                'Placebo_Coef': placebo_coef,
                'Placebo_PVal': placebo_pval,
                'Status': flag,
                'Message': msg
            })
        except:
            pass
            
    return pd.DataFrame(results)

def fit_competitor_model(df):
    """
    Regress units on price gap.
    """
    if 'price_gap_to_competitor' not in df.columns:
        return None, "Feature 'price_gap_to_competitor' missing."
        
    formula = "Q('Units Sold') ~ price_gap_to_competitor"
    
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
        return "Insignificant / No clear association"
    if coef < -1:
        return "Elastic (Price Cut -> Revenue UP)"
    elif coef > -1 and coef < 0:
        return "Inelastic (Price Hike -> Revenue UP)"
    elif coef > 0:
        return "Positive Association (Review Controls)" # Warning flag
    else:
        return "Unit Elastic"

def calculate_what_if(elasticity, current_price, current_units, pct_change):
    """
    Calculates expected impact.
    pct_change is in percent (e.g. -10 for 10% drop)
    """
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
