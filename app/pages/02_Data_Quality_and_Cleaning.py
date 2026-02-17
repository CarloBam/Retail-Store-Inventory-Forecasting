import streamlit as st
import sys
import os
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.cleaning import apply_cleaning_pipeline, get_missingness_table
from src.data_profiling import render_correlation_matrix, render_variable_drilldown
from src.ui_components import render_sidebar_progress, render_page_footer, simple_header
from src.navigation import render_top_navigation

st.set_page_config(page_title="Data Cleaning", layout="wide", page_icon="🧹")

# Top Nav
render_top_navigation("Feature Engineering", "03_Feature_Engineering",
                      can_proceed=(st.session_state.get('cleaned_data') is not None),
                      warning_msg="Please run the cleaning pipeline.")

render_sidebar_progress()
simple_header("Data Quality & Cleaning Workbench", "🧹")

# 1. Access Session Data
if 'raw_data' not in st.session_state or st.session_state['raw_data'] is None:
    st.warning("⚠️ No data found. Please upload data in the 'Data Upload' page.")
    # Show link to upload
    render_page_footer("Data Upload", "01_Data_Upload", "Go back to load data.")
    st.stop()

df_raw = st.session_state['raw_data']

# --- Configuration Sidebar ---
st.sidebar.header("🔧 Cleaning Configuration")
st.sidebar.markdown("Changes trigger automatic re-computation.")

# Target Variable
target_candidates = [col for col in df_raw.columns if pd.api.types.is_numeric_dtype(df_raw[col])]
default_target = 'Units Sold' if 'Units Sold' in target_candidates else target_candidates[0] if target_candidates else None
target_col = st.sidebar.selectbox("🎯 Target Variable", options=df_raw.columns, index=df_raw.columns.get_loc(default_target) if default_target else 0)

# 1. Missing Value Handling
st.sidebar.subheader("1. Missing Values")
drop_missing_target = st.sidebar.checkbox(f"Drop rows where '{target_col}' is missing", value=True)
impute_numeric = st.sidebar.selectbox("Impute Numeric Columns", ["None", "Median (Global)", "Zero"], index=0)
impute_categorical = st.sidebar.checkbox("Impute Categorical with Mode", value=False)

# 2. Outliers
st.sidebar.subheader("2. Outlier Handling")
outlier_method = st.sidebar.selectbox("Method for Units/Price", ["None", "Winsorize", "Clip (Z-score)"])

winsorize_lower = 0.05
winsorize_upper = 0.95
z_threshold = 3.0

if outlier_method == "Winsorize":
    winsorize_lower = st.sidebar.slider("Lower Percentile", 0.0, 0.2, 0.05, 0.01)
    winsorize_upper = st.sidebar.slider("Upper Percentile", 0.8, 1.0, 0.95, 0.01)
elif outlier_method == "Clip (Z-score)":
    z_threshold = st.sidebar.slider("Z-Score Threshold", 1.0, 5.0, 3.0, 0.1)

# 3. Operations
st.sidebar.subheader("3. General Operations")
remove_duplicates = st.sidebar.checkbox("Remove Duplicate Rows", value=True)
force_types = st.sidebar.checkbox("Enforce Standard Types", value=True, help="Ensures numeric cols are numbers, boolean cols are 0/1")

# Build Config Dictionary
config = {
    'target_col': target_col,
    'drop_missing_target': drop_missing_target,
    'impute_numeric': impute_numeric,
    'impute_categorical_mode': impute_categorical,
    'outlier_method': outlier_method,
    'winsorize_lower': winsorize_lower,
    'winsorize_upper': winsorize_upper,
    'z_threshold': z_threshold,
    'remove_duplicates': remove_duplicates,
    'force_types': force_types
}

# --- Main Logic ---

# Apply Cleaning
with st.spinner("Applying cleaning rules..."):
    df_clean, audit_log = apply_cleaning_pipeline(df_raw, config)
    st.session_state['cleaned_data'] = df_clean
    st.session_state['df_clean'] = df_clean # Per request

# --- Layout: 3 Columns ---
# 1. Before Stats | 2. After Stats | 3. Audit Log

st.subheader("📊 Impact Analysis")
c1, c2, c3 = st.columns([1, 1, 1.5])

with c1:
    st.markdown("### 🔴 Original")
    st.metric("Rows", f"{len(df_raw):,}")
    st.metric("Missing Values", f"{df_raw.isnull().sum().sum():,}")
    st.markdown("**Sample Row:**")
    st.dataframe(df_raw.head(1).T, use_container_width=True)

with c2:
    st.markdown("### 🟢 Cleaned")
    delta_rows = int(len(df_clean) - len(df_raw))
    st.metric("Rows", f"{len(df_clean):,}", delta=delta_rows)
    
    final_missing = int(df_clean.isnull().sum().sum())
    delta_missing = int(final_missing - df_raw.isnull().sum().sum())
    st.metric("Missing Values", f"{final_missing:,}", delta=delta_missing, delta_color="inverse")
    
    st.markdown("**Sample Row:**")
    st.dataframe(df_clean.head(1).T, use_container_width=True)

with c3:
    st.markdown("### 📝 Audit Log")
    if audit_log:
        for item in audit_log:
            st.text(f"• {item}")
    else:
        st.info("No cleaning operations applied.")

# --- Missingness Drill Down ---
st.markdown("---")
st.subheader("🔎 Missing Data Analysis")

if final_missing > 0 or df_raw.isnull().sum().sum() > 0:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Remaining Missing Data**")
        missing_table = get_missingness_table(df_clean)
        if not missing_table.empty:
            st.dataframe(missing_table.style.text_gradient(cmap='Reds', subset=['% Missing']), use_container_width=True)
        else:
            st.success("No missing values remained!")
else:
    st.success("Dataset is fully clean (no missing values).")

# --- Interactive Feature Drill Down ---
st.markdown("---")

# 1. Correlation Matrix
render_correlation_matrix(df_clean)

st.markdown("---")

# 2. Variable Profiling
render_variable_drilldown(df_clean)

st.sidebar.markdown("---")
render_page_footer("Feature Engineering", "03_Feature_Engineering", "Cleaning Complete. Ready for transformation.")
