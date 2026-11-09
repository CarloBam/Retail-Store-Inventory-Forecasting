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
                             train_lstm, predict_lstm,
                             walk_forward_validation)
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

test_ratio = st.sidebar.slider("Test Set Ratio", 0.1, 0.5, 0.2, 0.05)
forecast_horizon = st.sidebar.number_input("Forecast Horizon (Steps)", min_value=1, value=12)
use_walk_forward = st.sidebar.checkbox("Use Walk-Forward Validation (Robustness Check)", value=False)

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
    train, test = split_train_test(ts_agg, test_ratio=test_ratio)
    
    # 2. Baselines (Naive & Seasonal Naive)
    from src.forecasting import naive_forecast, seasonal_naive_forecast
    
    # Naive: Last value of train repeated
    naive_preds = naive_forecast(train[target_col], len(test))
    naive_metrics = calculate_metrics(test[target_col], naive_preds)
    
    # Seasonal Naive: Last season repeated (default 7 days for daily)
    s_period = 7 if agg_freq == "Daily" else 4 if agg_freq == "Weekly" else 12 
    s_naive_preds = seasonal_naive_forecast(train[target_col], len(test), period=s_period)
    s_naive_metrics = calculate_metrics(test[target_col], s_naive_preds)
    
    if use_walk_forward:
        st.info("Running Walk-Forward Validation... (This might take time)")
        
        # Wrapper for cross-validation
        def wf_wrapper(train_subset, test_subset):
             steps = len(test_subset)
             if model_choice == "ARIMA":
                 m, _ = train_arima(train_subset, target_col, order=model_params['order'], seasonal_order=model_params['seasonal_order'])
                 if m: return predict_arima(m, steps)
             elif model_choice == "Prophet":
                 cols = model_params.get('regressors', [])
                 m, _ = train_prophet(train_subset, target_col, regressors=cols)
                 if m: return predict_prophet(m, periods=steps, freq=freq_map[agg_freq], future_regressors_df=test_subset)
             elif model_choice in ["RandomForest", "XGBoost"]:
                 from src.forecasting import train_ml_model, predict_ml
                 m, _ = train_ml_model(train_subset, target_col, model_type=model_choice)
                 if m: return predict_ml(m, test_subset)
             return None

        # Execute
        wf_results = walk_forward_validation(ts_agg, target_col, wf_wrapper, n_splits=3)
        
        if not wf_results.empty:
             st.dataframe(wf_results.style.format({'RMSE': '{:.2f}', 'MAE': '{:.2f}', 'MAPE': '{:.2f}'}))
             if 'MAPE' in wf_results.columns:
                 avg_mape = wf_results['MAPE'].mean()
                 st.metric("Start-to-End Robustness (Avg CV MAPE)", f"{avg_mape:.2f}%")
        else:
             st.warning("Walk-Forward Validation produced no results.")
    
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
                future_forecast = predict_arima(model_fit, steps=len(test) + forecast_horizon) # Includes test split and requested horizon.
                # The first forecast covers the test split; the second extends beyond it.
                future_steps = len(test) + forecast_horizon
                full_forecast = predict_arima(model_fit, steps=future_steps) 
                
        elif model_choice == "Prophet":
            model_fit, error_msg = train_prophet(train, target_col, regressors=model_params['regressors'])
            if model_fit:
                 # Forecast the test split using its observed regressor values.
                val_forecast = predict_prophet(model_fit, periods=len(test), freq=freq_map[agg_freq], future_regressors_df=test) # Test rows provide observed regressor values.
                if val_forecast is not None:
                     predictions = val_forecast.values
                     
                # 2. Future Forecast
                # Future regressor values are unavailable beyond the test split.
                full_forecast = val_forecast # Validation-only when regressors are used.
                if not model_params['regressors']:
                     full_forecast_vals = predict_prophet(model_fit, periods=len(test)+forecast_horizon, freq=freq_map[agg_freq])
                     full_forecast = full_forecast_vals
            
        elif model_choice in ["RandomForest", "XGBoost"]:
            from src.forecasting import train_ml_model, predict_ml
            model_fit, error_msg = train_ml_model(train, target_col, model_type=model_choice)
            if model_fit:
                predictions = predict_ml(model_fit, test)
                # Future forecasts require feature values for the requested horizon.
                st.warning("Future forecasting for ML models currently requires future feature values. Displaying Test Set validation only.")
                full_forecast = None

        elif model_choice == "LSTM":
            model_fit, error_msg = train_lstm(train, target_col, 
                                              look_back=model_params['look_back'],
                                              epochs=model_params['epochs'],
                                              neurons=model_params['neurons'])
            if error_msg == "tensorflow_unavailable":
                st.warning("The LSTM model is disabled in the hosted demo to keep it lightweight. Run the app locally with TensorFlow installed to use it.")
                st.stop()
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
                model_metrics = calculate_metrics(test[target_col].values, predictions)
                
                st.subheader("Model Performance vs Baselines")
                
                # Comparison Table
                comp_data = {
                    "Model": [model_choice, "Naive (Last Value)", f"Seasonal Naive ({s_period}-step)"],
                    "RMSE": [model_metrics['RMSE'], naive_metrics['RMSE'], s_naive_metrics['RMSE']],
                    "MAE": [model_metrics['MAE'], naive_metrics['MAE'], s_naive_metrics['MAE']],
                    "MAPE (%)": [model_metrics['MAPE'], naive_metrics['MAPE'], s_naive_metrics['MAPE']]
                }
                
                # Save for Trust Report
                st.session_state['last_model_metrics'] = comp_data
                
                st.dataframe(pd.DataFrame(comp_data).style.format({
                    "RMSE": "{:.2f}", "MAE": "{:.2f}", "MAPE (%)": "{:.2f}"
                }).highlight_min(subset=["RMSE", "MAE", "MAPE (%)"], color="lightgreen", axis=0))
                
                # Determine winner
                best_mape = min(model_metrics['MAPE'], naive_metrics['MAPE'], s_naive_metrics['MAPE'])
                if model_metrics['MAPE'] == best_mape:
                     st.success(f"**Success!** The {model_choice} model outperforms baselines with a MAPE of {best_mape:.2f}%.")
                elif s_naive_metrics['MAPE'] == best_mape:
                     st.warning(f"**Note:** Seasonal Naive is the best performer ({best_mape:.2f}% MAPE). The complex model might be overfitting or needs tuning.")
            else:
                st.warning("Could not calculate metrics (prediction length mismatch or failure).")

            # Plotting
            st.subheader("Forecast Visualization")
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # 1. Train
            ax.plot(train.index, train[target_col], label='Training Data', color='gray', alpha=0.5)
            
            # 2. Test Actual
            ax.plot(test.index, test[target_col], label='Test Data (Actual)', color='black')
            
            # 3. Model Forecast (Validation)
            # Create index for predictions
            pred_idx = test.index
            if predictions is not None:
                ax.plot(pred_idx, predictions, label=f'{model_choice} Forecast (Validation)', color='blue', linestyle='--')
            
            # 4. Baselines
            ax.plot(test.index, naive_preds, label='Naive Baseline', color='red', linestyle=':', alpha=0.7)
            ax.plot(test.index, s_naive_preds, label='Seasonal Naive', color='green', linestyle=':', alpha=0.7)
            
            # 5. Future Forecast (if available and different)
            # Plot future predictions when available.
            # Plot the future forecast tail when available.
            
            ax.set_title(f"{model_choice} vs Baselines: {target_col}")
            ax.legend()
            st.pyplot(fig)
            
            # Insights
            st.markdown("---")
            st.subheader("💡 Model Insights")
            
            # Explainability for ML Models
            if model_choice in ["RandomForest", "XGBoost"] and model_fit:
                st.markdown("#### 🔍 Explainability (Drivers of Demand)")
                st.info("⚠️ **Interpretation Note**: These importance scores reflect 'Association', not necessarily 'Causality'. High importance means the model relies heavily on this feature for prediction.")
                
                from src.explainability import get_feature_importance, plot_partial_dependence
                
                # 1. Feature Importance
                # model_fit is the dict returned by train_ml_model containing 'model' and 'features'
                imp_df = get_feature_importance(model_fit['model'], model_fit['features'])
                
                if imp_df is not None:
                    c_imp1, c_imp2 = st.columns([1, 1])
                    with c_imp1:
                        st.dataframe(imp_df.head(10).style.background_gradient(cmap='Blues'), use_container_width=True)
                    with c_imp2:
                        # Bar chart
                        fig_imp, ax_imp = plt.subplots(figsize=(6, 4))
                        top_n = imp_df.head(10).sort_values('Importance', ascending=True)
                        ax_imp.barh(top_n['Feature'], top_n['Importance'], color='#4c72b0')
                        ax_imp.set_title("Top 10 Feature Importances")
                        st.pyplot(fig_imp)
                
                # 2. Partial Dependence (PDP)
                st.markdown("#### 📈 Partial Dependence Plots (Price Sensitivity)")
                pdp_feats = ['effective_price', 'discount_pct', 'Price', 'Discount', 'Units Sold_lag_7']
                # Filter strictly for existing cols
                valid_pdp = [f for f in pdp_feats if f in train.columns]
                
                if valid_pdp:
                     fig_pdp = plot_partial_dependence(model_fit['model'], train[model_fit['features']], valid_pdp[:3]) # Limit to top 3 relevant
                     if fig_pdp:
                         st.pyplot(fig_pdp)
                else:
                    st.caption("No price/discount features found for PDP plots.")

            elif model_choice == "ARIMA":
                st.info("ARIMA (AutoRegressive Integrated Moving Average) is excellent for linear trends and seasonality. It assumes data is stationary (d parameter handles this).")
            elif model_choice == "Prophet":
                st.info("Prophet (by Meta) handles seasonality, holidays, and changepoints robustly. It is interpretable and handles outliers well, but requires good regressor data for external factors.")
            elif model_choice == "LSTM":
                st.info("LSTM (Long Short-Term Memory) is a Deep Learning model capable of learning complex non-linear patterns and long-term dependencies. It requires more data and tuning.")

    render_page_footer("Pricing Analysis", "06_Pricing_Analysis_and_Elasticity", "Forecasts generated. Next: Analyze Price Elasticity.")

else:
    st.info("Configure model parameters and click 'Train' to see results.")
