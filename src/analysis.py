import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
import statsmodels.api as sm

# Optional XGBoost import
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

def train_classification_models(df, target_col='Units Sold', feature_cols=None):
    """
    Trains classification models to predict High vs Low demand.
    target_col: The continuous regression target to be converted to binary.
    """
    # 1. Create Binary Target (High Demand = > Median)
    median_val = df[target_col].median()
    y = (df[target_col] > median_val).astype(int)
    
    # 2. Prepare Features
    if feature_cols is None:
        # Default: all numeric except target
        feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns 
                        if c != target_col and c != 'High_Demand']
    
    X = df[feature_cols].dropna()
    y = y.loc[X.index] # Align
    
    # 3. Split Data (Train, Validation, Test)
    # 60% Train, 20% Val, 20% Test
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # 4. Train Models
    models = {}
    
    # Logistic Regression (Standardized for easier convergence)
    # We use statsmodels for nice P-values later, but sklearn for prediction consistency in loops
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    models['Logistic Regression'] = lr
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf.fit(X_train, y_train)
    models['Random Forest'] = rf
    
    # XGBoost
    if XGB_AVAILABLE:
        xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        xgb.fit(X_train, y_train)
        models['XGBoost'] = xgb
        
    return models, X_train, X_val, X_test, y_train, y_val, y_test

def get_model_stats(df, target_col, feature_cols):
    """
    Returns a dataframe of statistical metrics (Coef, P-Value) using Statsmodels Logit.
    This mimics SAS Enterprise Miner output.
    """
    # Create Binary Target
    median_val = df[target_col].median()
    y = (df[target_col] > median_val).astype(int)
    X = df[feature_cols].dropna()
    y = y.loc[X.index]
    
    # Add constant for intercept
    X_const = sm.add_constant(X)
    
    try:
        logit_model = sm.Logit(y, X_const)
        result = logit_model.fit(disp=0)
        
        # Extract Summary
        stats_df = pd.DataFrame({
            "Variable": result.params.index,
            "Coefficient": result.params.values,
            "Std. Error": result.bse.values,
            "z-Score": result.tvalues.values,
            "P-Value": result.pvalues.values
        })
        
        # Format
        stats_df['Coefficient'] = stats_df['Coefficient'].round(4)
        stats_df['P-Value'] = stats_df['P-Value'].round(5)
        stats_df['Significance'] = stats_df['P-Value'].apply(lambda x: '***' if x < 0.001 else ('**' if x < 0.01 else ('*' if x < 0.05 else '')))
        
        return stats_df.sort_values(by="P-Value")
    
    except Exception as e:
        return pd.DataFrame({"Error": [str(e)]})

def plot_roc_curves(models, X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Generates Plotly Figures for ROC Curves.
    Returns a dictionary of figures: {'Train': fig, 'Validation': fig, 'Test': fig}
    """
    datasets = {
        'Train': (X_train, y_train),
        'Validation': (X_val, y_val),
        'Test': (X_test, y_test)
    }
    
    figs = {}
    
    for split_name, (X, y) in datasets.items():
        fig = go.Figure()
        fig.add_shape(
            type='line', line=dict(dash='dash'),
            x0=0, x1=1, y0=0, y1=1
        )
        
        for model_name, model in models.items():
            try:
                # Predict Probabilities
                if hasattr(model, "predict_proba"):
                    y_score = model.predict_proba(X)[:, 1]
                else:
                    continue # Skip if no proba
                    
                fpr, tpr, _ = roc_curve(y, y_score)
                roc_auc = auc(fpr, tpr)
                
                fig.add_trace(go.Scatter(
                    x=fpr, y=tpr,
                    mode='lines',
                    name=f'{model_name} (AUC={roc_auc:.2f})'
                ))
            except Exception as e:
                print(f"Skipping {model_name} in {split_name}: {e}")
                
        fig.update_layout(
            title=f"ROC Curve - {split_name} Set",
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            yaxis=dict(scaleanchor="x", scaleratio=1),
            xaxis=dict(constrain='domain'),
            width=700, height=500
        )
        figs[split_name] = fig
        
    return figs
