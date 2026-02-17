import numpy as np
import sklearn.metrics as metrics

def calculate_metrics(y_true, y_pred):
    """
    Calculates common regression metrics: RMSE, MAE, MAPE.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    mse = metrics.mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = metrics.mean_absolute_error(y_true, y_pred)
    
    # helper for MAPE to avoid division by zero
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape
    }
