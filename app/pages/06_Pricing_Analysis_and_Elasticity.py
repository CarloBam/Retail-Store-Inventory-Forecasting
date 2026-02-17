import streamlit as st
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.pricing import prepare_pricing_data, fit_elasticity_model, fit_competitor_model, calculate_what_if
from src.ui_components import render_sidebar_progress, render_page_footer, simple_header
from src.navigation import render_top_navigation

st.set_page_config(page_title="Pricing Analysis", layout="wide", page_icon="🏷️")

# Top Nav
render_top_navigation("Recommendations", "07_Outputs_and_Recommendations", 
                      can_proceed=(st.session_state.get('feature_data') is not None),
                      warning_msg="Requires feature data.")

render_sidebar_progress()
simple_header("Pricing Intelligence Engine", "🏷️")

# 1. Data Loading
if 'feature_data' in st.session_state and st.session_state['feature_data'] is not None:
    df = st.session_state['feature_data'].copy()
elif 'cleaned_data' in st.session_state and st.session_state['cleaned_data'] is not None:
    df = st.session_state['cleaned_data'].copy()
    st.info("Using Cleaned Data (features like 'effective_price' might be missing if FE not run).")
else:
    st.error("Please load data first.")
    st.stop()
    
# Check prerequisites
if 'Category' not in df.columns:
    st.error("Dataset missing 'Category' column.")
    st.stop()

# --- Sidebar Controls ---
st.sidebar.header("Model Configuration")

# 1. Scope
agg_level = st.sidebar.radio("Aggregation Level", ["Daily", "Weekly"], index=1)

# 2. Filters
st.sidebar.subheader("Filters / Exclusions")
exclude_promo = st.sidebar.checkbox("Exclude Promo Periods", value=False)
max_discount = st.sidebar.slider("Exclude Discounts deeper than (%)", 0, 100, 50, help="Exclude clearance sales (e.g. >50% off) from model")

# 3. Model Spec
st.sidebar.subheader("Model Specification")
fe_options = []
if 'Store' in df.columns or 'Store ID' in df.columns: fe_options.append('Store')
if 'Region' in df.columns: fe_options.append('Region')
if 'Product' in df.columns or 'Product ID' in df.columns: fe_options.append('Product')

fixed_effects = st.sidebar.multiselect("Fixed Effects (Controls)", fe_options, default=fe_options[:1])
robust_cov = st.sidebar.checkbox("Robust Standard Errors (HC1)", value=True)

# Config Dict
config = {
    'agg_level': agg_level,
    'exclude_promo': exclude_promo,
    'max_discount_depth': max_discount,
}

# --- 1. Elasticity Analysis ---
st.header("1. Price Elasticity by Category")

# Run Analysis
with st.spinner("Fitting Regression Models..."):
    # Prepare Data
    df_model = prepare_pricing_data(df, config)
    
    # Fit
    elasticity_df = fit_elasticity_model(df_model, category_col='Category', fixed_effects=fixed_effects, robust_cov=robust_cov)
    
    if not elasticity_df.empty:
        # Display Table
        st.dataframe(elasticity_df[['Category', 'Elasticity', 'Elas_No_Promo', 'Elas_Promo', 'P-Value', 'CI_Lower', 'CI_Upper', 'Interpretation', 'Robustness']].style.format({
            "Elasticity": "{:.2f}",
            "Elas_No_Promo": "{:.2f}",
            "Elas_Promo": "{:.2f}",
            "P-Value": "{:.4f}",
            "CI_Lower": "{:.2f}", 
            "CI_Upper": "{:.2f}"
        }).background_gradient(subset=['Elasticity'], cmap="RdYlGn", vmin=-3, vmax=0), use_container_width=True)
        
        # Rank Plot
        st.subheader("Category Responsiveness Ranking")
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Sort by Elasticity (ascending, so most negative first)
        plot_df = elasticity_df.sort_values("Elasticity")
        
        y_pos = np.arange(len(plot_df))
        ax.barh(y_pos, plot_df['Elasticity'], xerr=(plot_df['Elasticity'] - plot_df['CI_Lower']), align='center', color='#4c72b0')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(plot_df['Category'])
        ax.set_xlabel("Price Elasticity (Coef)")
        ax.axvline(x=-1, color='red', linestyle='--', label='Unit Elastic (-1)')
        ax.set_title("Price Elasticity with 95% CI")
        ax.legend()
        st.pyplot(fig)
        
    else:
        st.warning("Model could not fit. Check filters or data sufficiency.")

# --- 2. What-If Calculator ---
st.markdown("---")
st.header("2. Revenue Simulator (What-If)")

st.warning("⚠️ **Endogeneity Warning**: These estimates capture 'Association', not strict 'Causality'. Prices often drop when demand is low (Clearance), which can bias elasticity towards 0. Use as a directional guide.")

c_sim1, c_sim2 = st.columns(2)

with c_sim1:
    if not elasticity_df.empty:
        sim_cat = st.selectbox("Select Category to Simulate", elasticity_df['Category'].unique())
        
        # Get baseline stats for that category
        # Recalculate based on prepared data model not full raw
        # Actually logic is fine as approximation
        base_df = df_model[df_model['Category'] == sim_cat] if 'Category' in df_model.columns else df_model
        curr_price = base_df['Price'].mean()
        curr_units = base_df['Units Sold'].mean() # Average per period (Day/Week)
        
        # Get model elasticity
        model_elast = elasticity_df[elasticity_df['Category'] == sim_cat]['Elasticity'].values[0]
        
        st.metric("Modeled Elasticity", f"{model_elast:.2f}")
        
    else:
        st.warning("No model results available.")
        st.stop()

with c_sim2:
    pct_change_input = st.slider("Proposed Price Change (%)", -50, 50, 0)
    
    if pct_change_input != 0:
        # Pass elasticity 
        sim_res = calculate_what_if(model_elast, curr_price, curr_units, pct_change_input)
        
        m1, m2 = st.columns(2)
        m1.metric("Predicted Demand Vol", f"{sim_res['Units Change %']:.1f}%", delta_color="normal")
        m2.metric("Predicted Revenue", f"{sim_res['Revenue Change %']:.1f}%", delta_color="normal")
        
        # Interpretation text
        if sim_res['Revenue Change %'] > 0:
             st.success(f"Strategy Recommended: Revenue projected to increase by {sim_res['Revenue Change %']:.1f}%")
        else:
             st.error(f"Strategy Warning: Revenue projected to decrease by {abs(sim_res['Revenue Change %']):.1f}%")

# --- 3. Competitor Analysis ---
st.markdown("---")
st.header("3. Competitor Gap Analysis")

if 'price_gap_to_competitor' in df.columns:
    model_comp, err = fit_competitor_model(df_model)
    
    if model_comp:
        coef = model_comp.params['price_gap_to_competitor']
        pval = model_comp.pvalues['price_gap_to_competitor']
        rsquared = model_comp.rsquared
        
        c_comp1, c_comp2 = st.columns([1, 2])
        
        with c_comp1:
            st.metric("Sensitivity to Competitor Price", f"{coef:.2f}")
            st.caption(f"P-Value: {pval:.4f}")
            
        with c_comp2:
            st.markdown("### Interpretation")
            if pval < 0.05:
                direction = "gains" if coef > 0 else "loses"
                st.write(f"Significant relationship found. For every $1 your effective price is **higher** than the competitor, you statistically **{direction}** {abs(coef):.2f} units of sales.")
                if coef > 0:
                    st.warning("Note: Positive coefficient might imply you are the premium option or quality leader (higher price gap = more sales).")
            else:
                st.write("No statistically significant relationship found between competitor price gap and sales volume in this dataset.")
                
    else:
        st.warning(f"Could not fit competitor model: {err}")
else:
    st.info("No competitor pricing data found (Feature 'price_gap_to_competitor').")

# --- 4. Robustness Checks (Placebo) ---
st.markdown("---")
st.header("4. Robustness Checks (Advanced)")
with st.expander("Diagnostics: Placebo & Endogeneity Tests", expanded=True):
    from src.pricing import run_placebo_test
    
    st.write("**Placebo Test**: We regress current demand on *Future Price*. In a causal world, future prices should NOT affect past sales. If they do, it suggests 'Endogeneity' (e.g. users anticipating sales, or managers setting prices based on expected trend).")
    
    placebo_res = run_placebo_test(df_model, category_col='Category')
    
    if not placebo_res.empty:
        st.dataframe(placebo_res.style.apply(lambda x: ['background-color: #ffcccc' if v == 'FAIL' else '' for v in x], subset=['Status']), use_container_width=True)
        
        fails = placebo_res[placebo_res['Status'] == 'FAIL']
        if not fails.empty:
            st.warning(f"⚠️ Potential endogeneity detected in {len(fails)} categories. Interpretation of causality should be cautious.")
        else:
            st.success("✅ Placebo test passed. No significant trend confounding detected.")
    else:
        st.info("Insufficient data for placebo test.")

render_page_footer("Recommendations", "07_Outputs_and_Recommendations", "Pricing analyzed. Next: See Executive Recommendations.")
