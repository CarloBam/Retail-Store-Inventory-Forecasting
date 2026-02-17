import streamlit as st
import sys
import os
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.features import generate_features_pipeline
from src.ui_components import render_sidebar_progress, render_page_footer, simple_header
from src.navigation import render_top_navigation

st.set_page_config(page_title="Feature Engineering", layout="wide", page_icon="🛠️")

# Top Nav
render_top_navigation("EDA", "04_EDA", 
                      can_proceed=(st.session_state.get('feature_data') is not None),
                      warning_msg="Please generate features first.")

render_sidebar_progress()
simple_header("Feature Engineering Lab", "🛠️")

# 1. Access Cleaned Data
if 'cleaned_data' not in st.session_state or st.session_state['cleaned_data'] is None:
    st.warning("⚠️ No cleaned data found. Please complete the 'Data Quality' step first.")
    render_page_footer("Data Quality", "02_Data_Quality_and_Cleaning", "Go back to clean data.")
    st.stop()

df = st.session_state['cleaned_data']

# --- Configuration Sidebar ---
st.sidebar.header("Feature Configuration")

# Target
target_candidates = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
default_target = 'Units Sold' if 'Units Sold' in target_candidates else target_candidates[0] if target_candidates else None
target_col = st.sidebar.selectbox("🎯 Target Variable", options=target_candidates, index=target_candidates.index(default_target) if default_target else 0)

use_log_target = st.sidebar.checkbox("Use Log1p Transformation for Target", value=False)

# Time
st.sidebar.subheader("1. Time Features")
add_time = st.sidebar.checkbox("Generate Date Components (Year, Month, etc.)", value=True)

# Lags & Rolling
st.sidebar.subheader("2. Lag & Rolling Features")
st.sidebar.caption("Grouping by Store ID & Product ID automatically if present.")

lags = st.sidebar.multiselect("Lag Periods (Days)", [1, 7, 14, 21, 28, 30], default=[1, 7, 14, 28])
rolling_windows = st.sidebar.multiselect("Rolling Mean Windows (Days)", [3, 7, 14, 28, 30], default=[7, 28])

# Price Features
st.sidebar.subheader("3. Price & Economics")
add_log_price = st.sidebar.checkbox("Log Transform Price", value=True)
add_econ_features = st.sidebar.checkbox("Add Discount & Gap Features", value=True)

# Seasonality & Promo
st.sidebar.subheader("4. categorical & Promo")
encode_seasonality = st.sidebar.checkbox("One-Hot Encode Seasonality/Weather", value=True)
promo_logic = st.sidebar.selectbox("Promo Flag Logic", ["Any (Holiday OR Discount)", "HolidayOnly", "DiscountOnly"], index=0)


# Config Object
config = {
    'target_col': target_col,
    'date_col': 'Date', # Assumed from cleaning
    'store_col': 'Store ID' if 'Store ID' in df.columns else 'Store',
    'product_col': 'Product ID' if 'Product ID' in df.columns else 'Product',
    'lags': lags,
    'rolling_windows': rolling_windows,
    'add_time_features': add_time,
    'add_log_price': add_log_price,
    'use_log_target': use_log_target,
    'encode_seasonality': encode_seasonality,
    'promo_logic': 'Any' if promo_logic.startswith("Any") else promo_logic,
    'add_promo_flag': True 
}

# --- Action Button ---
if st.button("Generate Features 🚀", type="primary"):
    with st.spinner("Engineering features... this may take a moment for large datasets"):
        try:
            # Force reload of features module to pick up any code changes (like Week 52 fix)
            import importlib
            import src.features
            importlib.reload(src.features)
            from src.features import generate_features_pipeline

            df_feat = generate_features_pipeline(df, config)
            st.session_state['feature_data'] = df_feat
            st.session_state['df_feat'] = df_feat # Per request
            
            st.success("Features generated successfully!")
            
            # --- Results Display ---
            st.markdown("---")
            st.subheader("Results Preview")
            
            # 1. Summary Metrics
            new_cols = len(df_feat.columns) - len(df.columns)
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Columns", len(df_feat.columns), delta=new_cols)
            c2.metric("Rows", len(df_feat))
            c3.metric("Target Variable", target_col, f"Log Transformed: {use_log_target}")
            
            # 2. DataFrame Preview
            with st.expander("🔎 View Engineered Dataframe", expanded=True):
                st.dataframe(df_feat.head(20))
                
            # 3. List of New Features
            with st.expander("📋 New Features List"):
                new_feature_names = [c for c in df_feat.columns if c not in df.columns]
                st.write(new_feature_names)

            # 4. Feature Contract & QA Dashboard
            with st.expander("🛡️ Feature Contract & QA Report", expanded=True):
                st.markdown("#### 1. Price Feature Integrity")
                if 'price_gap_to_competitor' in df_feat.columns:
                    st.success("✅ `price_gap_to_competitor` successfully created.")
                else:
                    st.warning("⚠️ `price_gap_to_competitor` MISSING. Competitor models will be disabled.")
                    st.caption("Check if 'Competitor Pricing' column exists in source data.")
                
                if 'effective_price' in df_feat.columns:
                    st.success("✅ `effective_price` successfully created.")
                
                st.markdown("#### 2. Lag/Rolling Warm-up Check")
                # Check for nulls in lag cols
                lag_cols = [c for c in df_feat.columns if '_lag_' in c or '_roll_' in c]
                if lag_cols:
                    nulls_in_lags = df_feat[lag_cols].isnull().sum().sum()
                    if nulls_in_lags == 0:
                        st.success(f"✅ Warm-up period handled. 0 nulls across {len(lag_cols)} lag/rolling features.")
                    else:
                        st.error(f"❌ Found {nulls_in_lags} nulls in lag features. Warm-up drop might have failed.")
                        st.dataframe(df_feat[lag_cols].isnull().sum().to_frame("Null Count").T)
                else:
                    st.info("No lag features requested.")

            render_page_footer("EDA", "04_EDA", "Features generated. Ready for analysis.")
                
        except Exception as e:
            st.error(f"Error during feature engineering: {e}")
            st.warning("Check if 'Store ID' and 'Product ID' columns exist for grouping logic.")

elif 'feature_data' in st.session_state and st.session_state['feature_data'] is not None:
    st.info("Features already generated. Click Button to Regenerate with new settings.")
    st.dataframe(st.session_state['feature_data'].head())
    
    render_page_footer("EDA", "04_EDA", "Proceed to Exploratory Data Analysis.")

# Remove manual Proceed button in sidebar as it's redundant with footer
st.sidebar.markdown("---")
