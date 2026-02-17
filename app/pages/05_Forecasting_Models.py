import streamlit as st
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.forecasting import (prepare_time_series, split_train_test, 
                             train_arima, predict_arima,
                             train_prophet, predict_prophet,
                             train_lstm, predict_lstm)
from src.metrics import calculate_metrics
from src.ui_components import render_sidebar_progress, render_page_footer, simple_header
from src.navigation import render_top_navigation

st.set_page_config(page_title="Forecasting Models", layout="wide", page_icon="🔮")

# Top Nav
render_top_navigation("Pricing Analysis", "06_Pricing_Analysis_and_Elasticity", 
                      can_proceed=(st.session_state.get('feature_data') is not None),
                      warning_msg="Generate features first")

render_sidebar_progress()
simple_header("Time Series Forecasting Lab", "🔮")

# 1. Data Loading
if 'feature_data' in st.session_state and st.session_state['feature_data'] is not None:
    df = st.session_state['feature_data'].copy()
elif 'cleaned_data' in st.session_state and st.session_state['cleaned_data'] is not None:
    df = st.session_state['cleaned_data'].copy()
else:
    st.error("Please load and clean data first.")
    st.stop()

# --- Sidebar Configuration ---
st.sidebar.header("1. Series Selection")

# Selectors
# Selectors
date_col = 'Date'
# Ensure date type
if date_col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[date_col]):
     df[date_col] = pd.to_datetime(df[date_col])

# Year Filter for Analysis
if date_col in df.columns:
    all_years_f = sorted(df[date_col].dt.year.unique().tolist())
    selected_years_f = st.sidebar.multiselect("Filter Years (History)", all_years_f, default=all_years_f)
    if selected_years_f:
        df = df[df[date_col].dt.year.isin(selected_years_f)]
store_col = 'Store ID' if 'Store ID' in df.columns else 'Store'
product_col = 'Product ID' if 'Product ID' in df.columns else 'Product'
target_col = st.sidebar.selectbox("Target Metric", [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])], index=0)

# Drill down
all_stores = df[store_col].unique().tolist() if store_col in df.columns else []
all_products = df[product_col].unique().tolist() if product_col in df.columns else []

selected_store = st.sidebar.selectbox("Select Store", all_stores, index=0)
# Filter products available in that store to reduce noise
available_products = df[df[store_col] == selected_store][product_col].unique().tolist() if store_col in df.columns else all_products
selected_product = st.sidebar.selectbox("Select Product", available_products, index=0)

# Aggregation
agg_freq = st.sidebar.radio("Aggregation Frequency", ["Daily", "Weekly", "Monthly"], index=0) # Default Daily for ML
freq_map = {"Daily": "D", "Weekly": "W", "Monthly": "M"}

# 2. Prepare Data (Cached logic usually, but here fast enough)
ts_agg, err = prepare_time_series(df, date_col, target_col, 
                                  agg_freq=freq_map[agg_freq], 
                                  store_col=store_col, product_col=product_col,
                                  store_id=selected_store, product_id=selected_product)

if err:
    st.error(f"Data Prep Error: {err}")
    st.stop()

# Show Series Preview
st.subheader(f"Analyzing: {selected_store} - {selected_product}")
# Simple plot
fig_prev, ax_prev = plt.subplots(figsize=(10, 3))
ax_prev.plot(ts_agg.index, ts_agg[target_col], label='History')
ax_prev.set_title("Historical Series")
st.pyplot(fig_prev)

# --- Model Config ---
st.sidebar.markdown("---")
st.sidebar.header("2. Model Configuration")

model_choice = st.sidebar.selectbox("Select Model", ["ARIMA", "Prophet", "RandomForest", "XGBoost", "LSTM"])

test_days = st.sidebar.number_input("Test Set Size (Steps)", min_value=1, max_value=int(len(ts_agg)*0.4), value=12)
forecast_horizon = st.sidebar.number_input("Forecast Horizon (Steps)", min_value=1, value=12)

# Model Specific Params
model_params = {}
if model_choice == "ARIMA":
    st.sidebar.subheader("ARIMA Params")
    p = st.sidebar.number_input("p (Auto-Regressive)", 0, 10, 1)
    d = st.sidebar.number_input("d (Integrated)", 0, 5, 1)
    q = st.sidebar.number_input("q (Moving Average)", 0, 10, 1)
    
    use_seasonal = st.sidebar.checkbox("Use Seasonal Order")
    if use_seasonal:
        P = st.sidebar.number_input("P", 0, 5, 0)
        D = st.sidebar.number_input("D", 0, 5, 1)
        Q = st.sidebar.number_input("Q", 0, 5, 1)
        s = st.sidebar.number_input("s (Period)", 2, 52, 7) # Default weekly
        model_params['seasonal_order'] = (P, D, Q, s)
    else:
        model_params['seasonal_order'] = (0, 0, 0, 0)
        
    model_params['order'] = (p, d, q)

elif model_choice == "Prophet":
    st.sidebar.subheader("Prophet Params")
    use_holidays = st.sidebar.checkbox("Use Holidays Regressor", value=True)
    use_discount = st.sidebar.checkbox("Use Discount Regressor", value=False)
    model_params['regressors'] = []
    if use_holidays: model_params['regressors'].append('Holiday')
    if use_discount: model_params['regressors'].append('Discount')

elif model_choice in ["RandomForest", "XGBoost"]:
    st.sidebar.subheader(f"{model_choice} Params")
    st.sidebar.info("Uses engineered features (Lags, Rolling Means). Ensure Step 3 created them.")
    
elif model_choice == "LSTM":
    st.sidebar.subheader("LSTM Params")
    look_back = st.sidebar.slider("Look Back Sequence", 1, 30, 4)
    epochs = st.sidebar.slider("Epochs", 10, 200, 50)
    neurons = st.sidebar.slider("Hidden Units", 10, 200, 50)
    model_params['look_back'] = look_back
    model_params['epochs'] = epochs
    model_params['neurons'] = neurons

# --- Train & Forecast ---

if st.button(f"Train {model_choice} & Forecast", type="primary"):
    
    # 1. Split
    train, test = split_train_test(ts_agg, test_size_days=test_days)
    
    with st.spinner(f"Training {model_choice}..."):
        model_fit = None
        error_msg = None
        predictions = None
        
        # Train
        if model_choice == "ARIMA":
            model_fit, error_msg = train_arima(train, target_col, 
                                               order=model_params['order'], 
                                               seasonal_order=model_params['seasonal_order'])
            if model_fit:
                predictions = predict_arima(model_fit, steps=len(test))
                future_forecast = predict_arima(model_fit, steps=len(test) + forecast_horizon) # Hack: Predict beyond
                # Actually predict_arima is stateless offset usually, so steps=len(test) gives forecasts for test period starting after train
                # To get future horizon, we can just ask for more steps from end of train? 
                # Yes, steps argument in forecast() usually implies steps ahead from end of fit.
                
                # Let's get "test predictions" (validation) and "future predictions"
                # Validation:
                # In statsmodels, we can use get_forecast or predict(start, end).
                # model_fit.forecast(steps) starts after end of training.
                
                # So `predictions` above covers the 'test' period exactly if steps=len(test)
                
                # Future:
                future_steps = len(test) + forecast_horizon
                full_forecast = predict_arima(model_fit, steps=future_steps) 
                
        elif model_choice == "Prophet":
            model_fit, error_msg = train_prophet(train, target_col, regressors=model_params['regressors'])
            if model_fit:
                # Prophet needs extra care for regressors in future
                # For validation (Test period):
                # Construct future frame that matches test dates
                # We need features from 'test' dataframe for regressors
                
                 # 1. Validation Forecast
                val_forecast = predict_prophet(model_fit, periods=len(test), freq=freq_map[agg_freq], future_regressors_df=test) # Passing test df as future regressors source
                if val_forecast is not None:
                     predictions = val_forecast.values
                     
                # 2. Future Forecast
                # We don't have known future features beyond test data in this dataset
                # So we can't truly predict beyond test if we use regressors without making assumptions
                # For now, let's limit "Future" visualization to just the Validation part for Prophet with Regressors
                # OR we handle it by not asking for horizon beyond test if regressors are used.
                full_forecast = val_forecast # Placeholder
                if not model_params['regressors']:
                     full_forecast_vals = predict_prophet(model_fit, periods=len(test)+forecast_horizon, freq=freq_map[agg_freq])
                     full_forecast = full_forecast_vals
            
        elif model_choice in ["RandomForest", "XGBoost"]:
            from src.forecasting import train_ml_model, predict_ml
            model_fit, error_msg = train_ml_model(train, target_col, model_type=model_choice)
            if model_fit:
                predictions = predict_ml(model_fit, test)
                # Future:
                # We need features for future.
                # Assuming simple recursive or static features not available yet for horizon
                # Warn user
                st.warning("Future forecasting for ML models currently requires future feature values. Displaying Test Set validation only.")
                full_forecast = None # Or implement recursive feature generation if robust

        elif model_choice == "LSTM":
            model_fit, error_msg = train_lstm(train, target_col, 
                                              look_back=model_params['look_back'],
                                              epochs=model_params['epochs'],
                                              neurons=model_params['neurons'])
            if model_fit:
                # Recursive prediction for test set size
                predictions = predict_lstm(model_fit, steps=len(test))
                
                # Future
                full_forecast = predict_lstm(model_fit, steps=len(test) + forecast_horizon)

        # Handling Results
        if error_msg:
            st.error(f"Training Failed: {error_msg}")
        else:
            # Metrics
            if predictions is not None and len(predictions) == len(test):
                metrics = calculate_metrics(test[target_col].values, predictions)
                
                st.subheader("Model Performance (Test Set)")
                c1, c2, c3 = st.columns(3)
                c1.metric("RMSE", f"{metrics['RMSE']:.2f}")
                c2.metric("MAE", f"{metrics['MAE']:.2f}")
                c3.metric("MAPE", f"{metrics['MAPE']:.2f}%")
            else:
                st.warning("Could not calculate metrics (prediction length mismatch or failure).")

            # Plotting
            st.subheader("Forecast Visualization")
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # 1. Train
            ax.plot(train.index, train[target_col], label='Training Data', color='gray', alpha=0.5)
            
            # 2. Test Actual
            ax.plot(test.index, test[target_col], label='Test Data (Actual)', color='black')
            
            # 3. Forecast (Validation)
            # Create index for predictions
            pred_idx = test.index
            if predictions is not None:
                ax.plot(pred_idx, predictions, label='Model Forecast (Validation)', color='blue', linestyle='--')
            
            # 4. Future Forecast (if available and different)
            # This is complex to plot neatly if formats differ, let's stick to validation plot heavily
            # If we calculated 'full_forecast', plot the tail
            
            ax.set_title(f"{model_choice} Forecast: {target_col}")
            ax.legend()
            st.pyplot(fig)
            
            # Explanation
            st.markdown("---")
            st.subheader("Model Insights")
            if model_choice == "ARIMA":
                st.info("ARIMA (AutoRegressive Integrated Moving Average) is excellent for linear trends and seasonality. It assumes data is stationary (d parameter handles this).")
            elif model_choice == "Prophet":
                st.info("Prophet (by Meta) handles seasonality, holidays, and changepoints robustly. It is interpretable and handles outliers well, but requires good regressor data for external factors.")
            elif model_choice == "LSTM":
                st.info("LSTM (Long Short-Term Memory) is a Deep Learning model capable of learning complex non-linear patterns and long-term dependencies. It requires more data and tuning.")

    render_page_footer("Pricing Analysis", "06_Pricing_Analysis_and_Elasticity", "Forecasts generated. Next: Analyze Price Elasticity.")

else:
    st.info("Configure model parameters and click 'Train' to see results.")
