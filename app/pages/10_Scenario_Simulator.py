
import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.ui_components import render_sidebar_progress, render_page_footer, simple_header
from src.navigation import render_top_navigation
from src.forecasting import seasonal_naive_forecast, prepare_time_series
from src.pricing import fit_elasticity_model, prepare_pricing_data

st.set_page_config(page_title="Scenario Simulator", layout="wide", page_icon="🧮")

# Top Nav
render_top_navigation("Experiments", "11_Experimentation_Roadmap",
                      can_proceed=True,
                      warning_msg="Define next steps.")

render_sidebar_progress()
simple_header("Strategic Scenario Simulator", "🧮")

st.markdown("Simulate the impact of price changes, promotions, and competitor moves on future demand and revenue.")

# 1. Data Selection
if 'feature_data' in st.session_state:
    df = st.session_state['feature_data'].copy()
elif 'cleaned_data' in st.session_state:
    df = st.session_state['cleaned_data'].copy()
    st.info("Using Cleaned Data (features might be limited).")
else:
    st.error("Please load data first.")
    st.stop()

# Select Series
col1, col2 = st.columns(2)
with col1:
    store_col = 'Store ID' if 'Store ID' in df.columns else 'Store'
    stores = df[store_col].unique()
    sel_store = st.selectbox("Select Store", stores)

with col2:
    product_col = 'Product ID' if 'Product ID' in df.columns else 'Product'
    # Filter products in store
    prods = df[df[store_col] == sel_store][product_col].unique()
    sel_prod = st.selectbox("Select Product", prods)

# 2. Baseline Forecast
# Use Seasonal Naive as the baseline when no fitted model is active.
target_col = 'Units Sold'
date_col = 'Date'

# Prep single series
ts_df = df[(df[store_col] == sel_store) & (df[product_col] == sel_prod)].copy()
ts_df[date_col] = pd.to_datetime(ts_df[date_col])
ts_df = ts_df.set_index(date_col).sort_index()
ts_series = ts_df[target_col]

# Forecast Horizon
horizon = st.sidebar.slider("Simulation Horizon (Weeks)", 1, 12, 4)
freq = 'W' # Default to weekly; daily history switches the interval below.
if pd.infer_freq(ts_series.index) == 'D':
    steps = horizon * 7
    freq = 'D'
else:
    steps = horizon
    freq = 'W'

# Generate Baseline
baseline_forecast = seasonal_naive_forecast(ts_series, steps)
# Create future index
last_date = ts_series.index.max()
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=steps, freq=freq[0])
baseline_df = pd.DataFrame({'Date': future_dates, 'Baseline_Units': baseline_forecast, 'Baseline_Revenue': baseline_forecast * ts_df['Price'].iloc[-1]}).set_index('Date')

st.subheader(f"Baseline for Next {horizon} Weeks")
st.line_chart(baseline_df['Baseline_Units'])

# 3. Simulation Parameters
st.sidebar.markdown("---")
st.sidebar.header("Scenario Constraints")

# Elasticity (Auto-detect or Manual)
# Try to fit quick elasticity for this category/product
# Elasticity (Auto-detect or Manual)
with st.spinner("Estimating Elasticity..."):
    # Determine Category
    current_cat = ts_df['Category'].iloc[0] if 'Category' in ts_df.columns else None
    
    elast_est = -1.5 # Default
    
    if current_cat:
        # Filter main df for this category to get enough data for model
        cat_df = df[df['Category'] == current_cat].copy()
        try:
             # Run simple fit
             # Config must match pricing generic requirement
             from src.pricing import fit_elasticity_model, prepare_pricing_data
             
             # Minimal config
             p_config = {'agg_level': 'Daily', 'exclude_promo': False}
             p_data = prepare_pricing_data(cat_df, p_config)
             
             # Fit
             e_res = fit_elasticity_model(p_data, category_col='Category')
             if not e_res.empty:
                 row = e_res[e_res['Category'] == current_cat]
                 if not row.empty:
                     elast_est = row['Elasticity'].values[0]
                     st.sidebar.success(f"Estimated Elasticity for {current_cat}: {elast_est:.2f}")
                 else:
                     st.sidebar.warning(f"Could not fit for {current_cat}, using default -1.5")
        except Exception as e:
            st.sidebar.error(f"Elasticity Fit Error: {e}, using default -1.5")
    else:
        st.sidebar.warning("No Category found, using default -1.5")

    # Allow Override
    # Use dynamic key to ensuring refreshing when selection changes
    sim_key = f"elast_{sel_store}_{sel_prod}"
    elast_est = st.sidebar.number_input("Elasticity (Override)", value=float(elast_est), step=0.1, key=sim_key)

price_change_pct = st.sidebar.slider("My Price Change (%)", -50, 50, 0)
competitor_change_pct = st.sidebar.slider("Competitor Price Change (%)", -50, 50, 0)
promo_active = st.sidebar.checkbox("Launch Promotion? (Lift +20%)", value=False)

# 4. Simulation Logic
# New Price
current_price = ts_df['Price'].iloc[-1]
new_price = current_price * (1 + price_change_pct/100.0)

# Effect Calculation
# 1. Own Price Effect
price_effect = elast_est * (price_change_pct / 100.0)

# 2. Competitor Effect (Cross Elasticity)
# Use 0.5 as the cross-elasticity assumption when no estimate is available.
cross_elast = 0.5 
comp_effect = cross_elast * (competitor_change_pct / 100.0)

# 3. Promo Effect
promo_lift = 0.20 if promo_active else 0.0

# Total Change Factor
total_pct_change_units = price_effect + comp_effect + promo_lift

# Apply
sim_units = baseline_df['Baseline_Units'] * (1 + total_pct_change_units)
sim_revenue = sim_units * new_price

# Combine
scenario_df = baseline_df.copy()
scenario_df['Simulated_Units'] = sim_units
scenario_df['Simulated_Revenue'] = sim_revenue

# 5. Results & Visualization
st.markdown("---")
st.header("Simulation Results")

# KPI Cards
c1, c2, c3 = st.columns(3)
base_rev_total = baseline_df['Baseline_Revenue'].sum()
sim_rev_total = scenario_df['Simulated_Revenue'].sum()
rev_diff = sim_rev_total - base_rev_total

c1.metric("Projected Revenue", f"${sim_rev_total:,.0f}", delta=f"{rev_diff:,.0f}")
c2.metric("Projected Units", f"{scenario_df['Simulated_Units'].sum():,.0f}", delta=f"{scenario_df['Simulated_Units'].sum() - baseline_df['Baseline_Units'].sum():,.0f}")
c3.metric("Margin Impact (Proxy 30%)", f"${(sim_rev_total * 0.3):,.0f}", delta=f"{(rev_diff * 0.3):,.0f}")

# Plots
tab1, tab2 = st.tabs(["Units Forecast", "Revenue Forecast"])

with tab1:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(scenario_df.index, scenario_df['Baseline_Units'], label='Baseline (Status Quo)', linestyle='--', color='gray')
    ax.plot(scenario_df.index, scenario_df['Simulated_Units'], label='Simulated Scenario', color='blue', linewidth=2)
    
    # Uncertainty Band (simple heuristic +/- 10%)
    ax.fill_between(scenario_df.index, scenario_df['Simulated_Units']*0.9, scenario_df['Simulated_Units']*1.1, color='blue', alpha=0.1, label='Risk Band (+/- 10%)')
    
    ax.set_title("Units Sold: Baseline vs Scenario")
    ax.legend()
    st.pyplot(fig)

with tab2:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(scenario_df.index, scenario_df['Baseline_Revenue'], label='Baseline Revenue', linestyle='--', color='gray')
    ax.plot(scenario_df.index, scenario_df['Simulated_Revenue'], label='Simulated Revenue', color='green', linewidth=2)
    ax.set_title("Revenue: Baseline vs Scenario")
    ax.legend()
    st.pyplot(fig)

# 6. Recommendation
if rev_diff > 0:
    st.success(f"**Outcome**: This scenario is projected to generate **${rev_diff:,.0f}** in additional revenue.")
else:
    st.error(f"**Outcome**: This scenario is projected to lose **${abs(rev_diff):,.0f}** in revenue relative to baseline.")

render_page_footer("Experiments", "11_Experimentation_Roadmap", "Design tests to validate this.")

