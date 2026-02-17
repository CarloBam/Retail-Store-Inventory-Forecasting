import pandas as pd
import numpy as np
import warnings
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

# --- Optional Imports with graceful handling ---
try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM
    # Suppress TF warnings
    tf.get_logger().setLevel('ERROR')
    TENSORFLOW_AVAILABLE = True
except ImportError as e:
    TENSORFLOW_AVAILABLE = False
    
# --- Helpers ---

def prepare_time_series(df, date_col, target_col, agg_freq='D', store_col=None, product_col=None, 
                        store_id=None, product_id=None):
    """
    Filters and aggregates data to create a single time series.
    agg_freq options: 'D', 'W', 'M'
    """
    df_ts = df.copy()
    
    # Filter
    if store_col and store_id:
        df_ts = df_ts[df_ts[store_col] == store_id]
        
    if product_col and product_id:
        df_ts = df_ts[df_ts[product_col] == product_id]
        
    if len(df_ts) == 0:
        return None, "No data found for selected Store/Product combination."
        
    # Set index
    if date_col in df_ts.columns:
        df_ts[date_col] = pd.to_datetime(df_ts[date_col])
        df_ts = df_ts.set_index(date_col).sort_index()
    else:
        return None, f"Date column '{date_col}' not found."
    
    # Resample/Aggregate
    # We aggregate target by sum (e.g. total units) and mean for others if needed
    # For simplicity, let's focus on the target series
    # But for models with regressors, we need to agg them too.
    
    # Aggregation rules
    agg_dict = {target_col: 'sum'}
    
    # Add regressors if present
    regressors = ['Price', 'Discount', 'Holiday', 'Promotion', 'log_Price', 'discount_pct', 'promo_flag']
    for r in regressors:
        if r in df_ts.columns:
            # For 0/1 columns max is appropriate (if holiday in week, week is holiday equivalent)
            if df_ts[r].nunique() <= 2: 
                agg_dict[r] = 'max'
            else:
                agg_dict[r] = 'mean'
                
    # Auto-add other numeric columns (e.g. features) not explicitly excluded
    # This ensures lag_x, roll_y features are kept (aggregated by mean)
    for c in df_ts.select_dtypes(include=[np.number]).columns:
        if c not in agg_dict and c != target_col:
             agg_dict[c] = 'mean'
                
    try:
        ts_agg = df_ts.resample(agg_freq).agg(agg_dict).dropna()
        if len(ts_agg) < 10:
             return None, "Time series too short after aggregation (<10 points)."
        return ts_agg, None
    except Exception as e:
        return None, f"Aggregation error: {e}"

def split_train_test(ts, test_size_days=30):
    """
    Splits time series by Date.
    """
    if len(ts) <= test_size_days:
        # Fallback if series is too short
        split_idx = int(len(ts) * 0.8)
    else:
        split_idx = len(ts) - test_size_days
        
    train = ts.iloc[:split_idx]
    test = ts.iloc[split_idx:]
    return train, test

# --- Models ---

def train_arima(train, target_col, order=(1,1,1), seasonal_order=(0,0,0,0)):
    """
    Trains SARIMAX model.
    """
    if not STATSMODELS_AVAILABLE:
        return None, "Statsmodels not installed."
        
    try:
        # Suppress convergence warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            model = SARIMAX(train[target_col], order=order, seasonal_order=seasonal_order, 
                            enforce_stationarity=False, enforce_invertibility=False)
            model_fit = model.fit(disp=False)
            return model_fit, None
    except Exception as e:
        return None, str(e)

def predict_arima(model_fit, steps):
    """
    Forecasts with ARIMA.
    """
    try:
        forecast = model_fit.forecast(steps=steps)
        return forecast
    except Exception as e:
        return None

def train_prophet(train, target_col, regressors=[]):
    """
    Trains Prophet model.
    """
    if not PROPHET_AVAILABLE:
        return None, "Prophet not installed."
        
    # Prophet logic
    df_p = train.reset_index().rename(columns={train.index.name: 'ds', target_col: 'y'})
    
    m = Prophet()
    
    for reg in regressors:
        if reg in train.columns:
            m.add_regressor(reg)
            
    m.fit(df_p)
    return m, None

def predict_prophet(model, periods, freq='D', future_regressors_df=None):
    """
    Forecasts with Prophet.
    future_regressors_df must contain the columns for the future period.
    """
    try:
        future = model.make_future_dataframe(periods=periods, freq=freq)
        
        # Merge regressors if needed
        if future_regressors_df is not None:
             # Merge or use passed regressor dataframe directly if indices align
             pass
        
        forecast = model.predict(future)
        return forecast['yhat'].tail(periods)
    except Exception as e:
        print(f"Prophet predict error: {e}")
        return None

# --- ML Models (Sklearn / XGBoost) ---

def train_ml_model(train_df, target_col, model_type='RandomForest', features=None):
    """
    Trains an ML model (RF or XGB) for time series forecasting.
    Expects train_df to have all features (lags, etc.) pre-calculated.
    """
    try:
        from sklearn.ensemble import RandomForestRegressor
        from xgboost import XGBRegressor
        
        # Select features
        if features is None:
            # automatic selection: all numeric except target
            features = [c for c in train_df.columns if c != target_col and pd.api.types.is_numeric_dtype(train_df[c])]
            
        X = train_df[features]
        y = train_df[target_col]
        
        if model_type == 'RandomForest':
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        elif model_type == 'XGBoost':
            model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1)
        else:
            return None, "Unknown model type"
            
        model.fit(X, y)
        
        return {'model': model, 'features': features, 'type': model_type}, None
        
    except Exception as e:
        return None, str(e)

def predict_ml(model_meta, future_df):
    """
    Predicts using the trained ML model.
    future_df must contain the feature columns.
    For recursive forecasting (if features depend on lags of target), 
    this simple version assumes future_df already has correct features 
    (e.g. valid for direct forecasting or if lags are known/estimated elsewhere).
    """
    try:
        model = model_meta['model']
        feats = model_meta['features']
        
        # Check simple presence
        missing = [f for f in feats if f not in future_df.columns]
        if missing:
             # Try to gracefully fill or error? 
             # For this demo, let's just warn/return None if critical 
             # Or fill with 0
             for m in missing:
                 future_df[m] = 0
        
        X_future = future_df[feats]
        preds = model.predict(X_future)
        return preds
        
    except Exception as e:
        return None

def train_lstm(train, target_col, look_back=7, epochs=10, batch_size=1, neurons=50):
    """
    Trains LSTM model.
    """
    if not TENSORFLOW_AVAILABLE:
        return None, "Tensorflow not installed."
        
    try:
        data = train[target_col].values
        
        # Normalize
        # Simple MinMax 0-1 based on training
        min_val, max_val = data.min(), data.max()
        scale = max_val - min_val if max_val != min_val else 1
        data_norm = (data - min_val) / scale
        
        # Create sequences
        X, y = [], []
        for i in range(len(data_norm) - look_back):
            X.append(data_norm[i:(i + look_back)])
            y.append(data_norm[i + look_back])
            
        X, y = np.array(X), np.array(y)
        
        # Reshape [samples, time steps, features]
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        # Design network
        model = Sequential()
        model.add(LSTM(neurons, input_shape=(look_back, 1)))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        
        model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)
        
        model_meta = {
            'model': model,
            'min': min_val,
            'scale': scale,
            'look_back': look_back,
            'last_sequence': data_norm[-look_back:]
        }
        
        return model_meta, None
        
    except Exception as e:
        import traceback
        return None, str(e)

def predict_lstm(model_meta, steps):
    """
    Recursive prediction for LSTM.
    """
    try:
        model = model_meta['model']
        curr_seq = model_meta['last_sequence'].reshape((1, model_meta['look_back'], 1))
        predictions = []
        
        for _ in range(steps):
            pred_norm = model.predict(curr_seq, verbose=0)[0,0]
            predictions.append(pred_norm)
            
            # Update sequence: remove first, add predictions
            # reshaped for append
            new_val = np.array([[[pred_norm]]])
            curr_seq = np.append(curr_seq[:, 1:, :], new_val, axis=1)
            
        # Inverse transform
        predictions = np.array(predictions) * model_meta['scale'] + model_meta['min']
        return predictions
        
    except Exception as e:
        return None
