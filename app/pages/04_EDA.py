import streamlit as st
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.eda import filter_data
from src.viz import (plot_sales_trends, plot_top_categories, plot_heatmap_pivot, 
                     plot_distribution, plot_promo_effect, plot_scatter_fit)
from src.seasonality import plot_seasonal_trends
# from src.utils import save_plot # Utils save_plot works for matplotlib, might need update for plotly or removing
from src.ui_components import render_sidebar_progress, render_page_footer, simple_header
from src.navigation import render_top_navigation

st.set_page_config(page_title="EDA", layout="wide", page_icon="📊")

# Top Nav
render_top_navigation("Forecasting Models", "05_Forecasting_Models", 
                      can_proceed=(st.session_state.get('feature_data') is not None),
                      warning_msg="We recommend generating features first, though basic data works.")

render_sidebar_progress()
simple_header("Exploratory Data Analysis", "📊")

# 1. Data Loading (Priority: Feature -> Cleaned -> Raw)
if 'feature_data' in st.session_state and st.session_state['feature_data'] is not None:
    df = st.session_state['feature_data'].copy()
elif 'cleaned_data' in st.session_state and st.session_state['cleaned_data'] is not None:
    df = st.session_state['cleaned_data'].copy()
    st.info("Using Cleaned Data (Features not yet generated)")
elif 'raw_data' in st.session_state and st.session_state['raw_data'] is not None:
    df = st.session_state['raw_data'].copy()
    st.warning("Using Raw Data. Please clean data for best results.")
else:
    st.error("No data found. Upload and load data first.")
    st.stop()

# Require a Date column for time-series analysis.
date_col = 'Date'
if date_col not in df.columns:
    st.error("Dataset missing 'Date' column. Cannot proceed with Time Series EDA.")
    st.stop()
    
# Calculate Revenue if missing (Proxy)
if 'Revenue' not in df.columns:
    if 'effective_price' in df.columns and 'Units Sold' in df.columns:
         df['Revenue'] = df['Units Sold'] * df['effective_price']
    elif 'Price' in df.columns and 'Units Sold' in df.columns:
        df['Revenue'] = df['Units Sold'] * df['Price']
    else:
        df['Revenue'] = 0

# --- Sidebar Controls ---
st.sidebar.header("Filters & Controls")

# Date Range
min_date = df[date_col].min().date()
max_date = df[date_col].max().date()
date_range = st.sidebar.date_input("Date Range", [min_date, max_date])

if len(date_range) != 2:
    st.sidebar.warning("Select both start and end date.")
    st.stop()

# Year Filter
all_years = sorted(df[date_col].dt.year.unique().tolist())
selected_years = st.sidebar.multiselect("Select Years", all_years, default=all_years)
if not selected_years:
    st.sidebar.error("Please select at least one year.")
    st.stop()

# Restrict analysis to selected years before applying other filters.
df = df[df[date_col].dt.year.isin(selected_years)]

# Categories
cat_col = 'Category'
categories = []
if cat_col in df.columns:
    all_cats = df[cat_col].unique().tolist()
    categories = st.sidebar.multiselect("Category", all_cats, default=all_cats[:5])

# Regions (if exists)
region_col = 'Region'
regions = []
if region_col in df.columns:
    all_regions = df[region_col].unique().tolist()
    regions = st.sidebar.multiselect("Region", all_regions, default=all_regions)

# Stores
store_col = 'Store ID' if 'Store ID' in df.columns else 'Store'
stores = []
if store_col in df.columns:
    all_stores = df[store_col].unique().tolist()
    stores = st.sidebar.multiselect("Store ID", all_stores, default=all_stores[:5]) # limit default to avoid clutter

# Log Scale Toggle
log_scale = st.sidebar.checkbox("Log Scale (Trends)", value=False)

# --- Filter Data ---
df_filtered = filter_data(df, date_range=date_range, categories=categories, regions=regions, stores=stores)

st.markdown(f"**Analyzing {len(df_filtered):,} records** (from {min_date} to {max_date})")

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["📈 Executive Trends", "📍 Segments & Distribution", "🏷️ Promo & Pricing"])

# TAB 1: Executive Trends
with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Total Units Sold Over Time")
        
        compare_mode = st.toggle("Compare Years (Seasonality)", value=False, key="toggle_sales")
        
        if compare_mode:
            fig1 = plot_seasonal_trends(df_filtered, date_col=date_col, value_col='Units Sold', title="Year-over-Year Sales Comparison")
        else:
            fig1 = plot_sales_trends(df_filtered, date_col=date_col, value_col='Units Sold', title="Daily Units Sold", log_scale=log_scale)
            
        st.plotly_chart(fig1, use_container_width=True)
        # if st.button("💾 Save Trend", key="btn_save_trend"):
        #     path = save_plot(fig1, "slide3_sales_trend.png")
        #     st.toast(f"Saved to {path}", icon="✅")

    with col2:
        st.subheader("Revenue (Proxy) Over Time")
        # Logic for Revenue comparison
        compare_mode_rev = st.toggle("Compare Years (Revenue)", value=False, key="toggle_rev")
        
        if compare_mode_rev:
             fig2 = plot_seasonal_trends(df_filtered, date_col=date_col, value_col='Revenue', title="Year-over-Year Revenue Comparison")
        else:
             fig2 = plot_sales_trends(df_filtered, date_col=date_col, value_col='Revenue', title="Daily Revenue (Est)", log_scale=log_scale)
             
        st.plotly_chart(fig2, use_container_width=True)
        
    st.markdown("---")
    
    col3, col4 = st.columns(2)
    with col3:
         st.subheader("Top Categories by Revenue")
         if 'Category' in df_filtered.columns:
             fig3 = plot_top_categories(df_filtered, cat_col='Category', value_col='Revenue', n=10)
             st.plotly_chart(fig3, use_container_width=True)
             if st.button("💾 Save Categories", key="btn_save_cat"):
                path = save_plot(fig3, "slide3_top_categories.png")
                st.toast(f"Saved to {path}", icon="✅")
         else:
             st.warning("Category column missing.")
             
    with col4:
         st.subheader("Top Products by Units Sold")
         prod_col = 'Product Name' if 'Product Name' in df_filtered.columns else 'Product ID'
         if prod_col in df_filtered.columns:
             fig4 = plot_top_categories(df_filtered, cat_col=prod_col, value_col='Units Sold', n=10, title="Top Selling Products")
             st.plotly_chart(fig4, use_container_width=True)

# TAB 2: Segmentation
with tab2:
    st.subheader("Regional Performance Heatmap")
    
    if region_col in df_filtered.columns and cat_col in df_filtered.columns:
        fig_heat = plot_heatmap_pivot(df_filtered, index_col=region_col, columns_col=cat_col, value_col='Units Sold', title=f"Avg Units Sold by {region_col} & {cat_col}")
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.warning(f"Need '{region_col}' and '{cat_col}' columns for heatmap.")
        
    st.markdown("---")
    st.subheader("Distributions")
    
    d1, d2 = st.columns(2)
    with d1:
        st.markdown("**Price Distribution by Category**")
        if cat_col in df_filtered.columns:
            fig_dist = plot_distribution(df_filtered, col='Price', by_col='Category', title="Price Boxplot by Category")
            st.plotly_chart(fig_dist, use_container_width=True)
            
    with d2:
        st.markdown("**Discount Distribution**")
        if 'Discount' in df_filtered.columns:
            fig_hist = plot_distribution(df_filtered, col='Discount', title="Discount Histogram")
            st.plotly_chart(fig_hist, use_container_width=True)

# TAB 3: Promo & Pricing
with tab3:
    col_p1, col_p2 = st.columns(2)
    
    # Promo Logic
    promo_col = 'promo_flag' if 'promo_flag' in df_filtered.columns else 'Holiday' if 'Holiday' in df_filtered.columns else None
    
    with col_p1:
        st.subheader("Promo Impact")
        if promo_col:
            fig_promo = plot_promo_effect(df_filtered, promo_col, value_col='Units Sold', title="Avg Units Sold: Promo vs Non-Promo")
            st.plotly_chart(fig_promo, use_container_width=True)
            st.caption(f"Using '{promo_col}' as promotion indicator.")
        else:
            st.warning("No promo flag found (Feature Engineering step recommended).")
            
    with col_p2:
        st.subheader("Price Elasticity Scatter")
        # Scatter: Effective Price vs Units Sold
        # Filter for a single Category to make sense, or use overall
        price_var = 'effective_price' if 'effective_price' in df_filtered.columns else 'Price'
        
        # Let user pick a category to drill down for scatter
        target_cat_scatter = st.selectbox("Select Category for Scatter Plot", categories, index=0) if categories else None
        
        scatter_data = df_filtered[df_filtered['Category'] == target_cat_scatter] if target_cat_scatter else df_filtered
        
        if len(scatter_data) > 0:
            fig_scat = plot_scatter_fit(scatter_data, x_col=price_var, y_col='Units Sold', title=f"Price vs Demand ({target_cat_scatter or 'All'})")
            st.plotly_chart(fig_scat, use_container_width=True)
            if st.button("💾 Save Elasticity Plot", key="btn_save_scat"):
                path = save_plot(fig_scat, "slide5_elasticity_scatter.png")
                st.toast(f"Saved to {path}", icon="✅")
        else:
            st.warning("Not enough data for scatter.")
            
    st.markdown("---")
    st.subheader("Competitor Analysis")
    if 'price_gap_to_competitor' in df_filtered.columns:
        fig_comp = plot_scatter_fit(df_filtered, x_col='price_gap_to_competitor', y_col='Units Sold', title="Competitor Price Gap vs Demand")
        st.plotly_chart(fig_comp, use_container_width=True)
    else:
        st.info("Feature 'price_gap_to_competitor' not found. Perform Feature Engineering first.")

render_page_footer("Forecasting Models", "05_Forecasting_Models", "Insights gathered. Next: Train forecasting models.")
