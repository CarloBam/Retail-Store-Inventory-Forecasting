import streamlit as st
import sys
import os
import pandas as pd
import time

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.data_io import load_data, DEFAULT_DATA_PATH
from src.data_processing_helpers import parse_and_load_data, get_data_dictionary, get_dataset_summary
from src.ui_components import render_sidebar_progress, render_page_footer, simple_header
from src.navigation import render_top_navigation

st.set_page_config(page_title="Data Upload", layout="wide", page_icon="📂")

# Top Nav
render_top_navigation("Data Quality", "02_Data_Quality_and_Cleaning", 
                      can_proceed=(st.session_state.get('raw_data') is not None),
                      warning_msg="Please upload data first.")

render_sidebar_progress()
simple_header("Data Upload & Ingestion", "📂")

st.markdown("""
Welcome to the data ingestion module. Here you can load the retail dataset. 
The system will automatically parse dates and coerce numeric types.
""")

# --- Sidebar ---
st.sidebar.header("Configuration")
# Train / Validation Split
split_val = st.sidebar.slider("Training Data Split (%)", min_value=50, max_value=95, value=80, step=5, help="Percentage of data to use for training. The rest is validation.")
st.session_state['train_split'] = split_val / 100.0

# --- Main Area ---
st.subheader("1. Import Data")

# State Management for Data Persistance
# If data is already loaded, show current status and option to reset
if st.session_state.get('raw_data') is not None:
    st.info(f"✅ Data is currently loaded ({st.session_state['raw_data'].shape[0]} rows).")
    
    col_r1, col_r2 = st.columns([1, 4])
    with col_r1:
        if st.button("🔄 Reset / Upload New", type="secondary"):
            st.session_state['raw_data'] = None
            st.session_state['df_raw'] = None
            st.rerun()
            
    # If not resetting, just show preview below (logic falls through to sections below)
    df_parsed = st.session_state['raw_data']

else:
    # Only show uploader if no data is loaded
    uploaded_file = st.file_uploader("Upload your Retail Inventory CSV", type=['csv'])
    
    df_parsed = None
    
    if uploaded_file is not None:
        with st.spinner("Reading uploaded file..."):
            df_temp = load_data(uploaded_file)
            
            if df_temp is not None:
                # Parse Data
                with st.spinner("Parsing and verifying data types..."):
                    df_parsed = parse_and_load_data(df_temp)
                    # Verify parsing happened
                    if 'Date' in df_parsed.columns and pd.api.types.is_datetime64_any_dtype(df_parsed['Date']):
                         st.caption("✅ Date column parsed successfully.")
                    
                    # Compute initial missingness for summary
                    # We DO NOT save to session state yet until Confirmed? 
                    # Actually, usually better to save immediately so re-runs don't lose it, 
                    # but maybe mark as "unconfirmed". For this UX, let's just save it.
                    st.session_state['raw_data'] = df_parsed
                    st.session_state['df_raw'] = df_parsed
                    st.toast("Upload successful!", icon="✅")
                    # Rerun to switch to "Loaded" view
                    st.rerun()

# 2. Process & Preview (Only if data exists)
if df_parsed is not None:
    
    # Section: Quick View
    with st.expander("🔎 Raw Data Preview", expanded=True):
        st.dataframe(df_parsed.head(10), use_container_width=True)

    # Section: Data Dictionary & Type Enforcement
    with st.expander("📖 Data Schema & Type Enforcement", expanded=True):
        st.info("Verify and adjust the data types below if incorrect. Click 'Set & Confirm Data' to apply changes.")
        
        # Get current schema
        dict_df = get_data_dictionary(df_parsed)
        
        # MAP dtypes to Options ["numeric", "categorical", "datetime", "bool"]
        def map_dtype(dtype_str):
            if 'int' in dtype_str or 'float' in dtype_str:
                return 'numeric'
            elif 'datetime' in dtype_str:
                return 'datetime'
            elif 'bool' in dtype_str:
                return 'bool'
            else:
                return 'categorical'
        
        dict_df['Type'] = dict_df['Type'].apply(map_dtype)

        # Allow editing
        edited_schema = st.data_editor(
            dict_df,
            column_config={
                "Column Name": st.column_config.TextColumn("Column Name", disabled=True),
                "Type": st.column_config.SelectboxColumn(
                    "Type",
                    options=["numeric", "categorical", "datetime", "bool"],
                    required=True
                )
            },
            hide_index=True,
            use_container_width=True,
            key="schema_editor"
        )
        
        if st.button("✅ Set & Confirm Data", type="primary"):
            # Enforce the types from edited_schema
            try:
                # Create a copy to modify
                df_mod = df_parsed.copy()
                
                # Iterate through edited schema and apply types
                for index, row in edited_schema.iterrows():
                    col = row['Column Name']
                    desired_type = row['Type']
                    
                    if col not in df_mod.columns:
                        continue
                        
                    # Attempt casting
                    if desired_type == 'numeric':
                        df_mod[col] = pd.to_numeric(df_mod[col], errors='coerce')
                    elif desired_type == 'datetime':
                        df_mod[col] = pd.to_datetime(df_mod[col], errors='coerce')
                    elif desired_type == 'categorical':
                        df_mod[col] = df_mod[col].astype(str) # Pandas category or just object/string
                    elif desired_type == 'bool':
                        df_mod[col] = df_mod[col].astype(bool)
                        
                # Update Session State
                st.session_state['raw_data'] = df_mod
                st.session_state['df_raw'] = df_mod
                
                st.toast("Data types updated and confirmed!", icon="🔒")
                st.success("Dataset confirmed with updated schema.")
                time.sleep(1) # Brief pause for user to see success
                st.rerun()
                
            except Exception as e:
                st.error(f"Failed to apply types: {e}")

    # Section: Summary Metrics
    with st.expander("📊 Dataset Summary", expanded=False):
        summary = get_dataset_summary(df_parsed)
        
        # Create a grid layout for metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", f"{summary['Rows']:,}")
        c2.metric("Columns", summary['Columns'])
        c3.metric("Stores", summary['Stores'])
        c4.metric("Products", summary['Products'])
        
        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Date Range", summary.get("Date Range", "N/A"))
        c6.metric("Categories", summary["Categories"])
        c7.metric("Missing Values", summary["Missing Values"])
        c8.metric("Missing %", f"{summary['Missing %']:.2f}%")
        
        if summary['Missing %'] > 0:
            st.warning("⚠️ Data contains missing values. Please proceed to the **Data Quality & Cleaning** page.")
        else:
            st.success("✅ Data looks complete (no missing values detected).")

    st.markdown("---")
    render_page_footer("Data Quality & Cleaning", "02_Data_Quality_and_Cleaning", "Data loaded. Next: Clean and prep the data.")
