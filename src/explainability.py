
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_feature_importance(model, feature_names):
    """
    Extracts feature importance from tree-based models.
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        return pd.DataFrame({
            'Feature': [feature_names[i] for i in indices],
            'Importance': importances[indices]
        })
    return None

def get_permutation_importance(model, X_val, y_val, scoring='neg_mean_absolute_error'):
    """
    Computes permutation importance (model agnostic).
    """
    from sklearn.inspection import permutation_importance
    try:
        r = permutation_importance(model, X_val, y_val, n_repeats=5, random_state=42, scoring=scoring)
        importances = r.importances_mean
        indices = np.argsort(importances)[::-1]
        
        return pd.DataFrame({
            'Feature': [X_val.columns[i] for i in indices],
            'Importance': importances[indices]
        })
    except Exception as e:
        print(f"Permutation importance error: {e}")
        return None

def plot_partial_dependence(model, X_val, feature_names):
    """
    Generates PDP plots for specified features.
    """
    from sklearn.inspection import PartialDependenceDisplay
    
    # Filter features that exist in X_val
    valid_features = [f for f in feature_names if f in X_val.columns]
    
    if not valid_features:
        return None
        
    try:
        fig, ax = plt.subplots(figsize=(10, 4 * len(valid_features)))
        PartialDependenceDisplay.from_estimator(model, X_val, valid_features, ax=ax)
        return fig
    except Exception as e:
        print(f"PDP error: {e}")
        return None
