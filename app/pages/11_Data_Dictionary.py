
import streamlit as st
import pandas as pd
import os
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.ui_components import render_sidebar_progress, render_page_footer, simple_header
from src.navigation import render_top_navigation

st.set_page_config(page_title="Data Dictionary", layout="wide", page_icon="📖")

render_top_navigation(None, None, can_proceed=False)

render_sidebar_progress()
simple_header("Data Dictionary & Definitions", "📖")

st.markdown("""
Reference guide for all raw and engineered features used in the Forecasting & Pricing models.
""")

# 1. Raw Data
st.header("1. Raw Data Features")
raw_data = [
    {"Column": "Date", "Type": "DateTime", "Description": "Transaction date (Daily). Key temporal index.", "Source": "POS System"},
    {"Column": "Store ID", "Type": "Categorical", "Description": "Unique identifier for the retail store.", "Source": "Master Data"},
    {"Column": "Product ID", "Type": "Categorical", "Description": "Unique identifier for the SKU/Item.", "Source": "Master Data"},
    {"Column": "Category", "Type": "Categorical", "Description": "Product hierarchy group (e.g. 'Electronics', 'Home'). Used for elasticity aggregation.", "Source": "Master Data"},
    {"Column": "Units Sold", "Type": "Numeric", "Description": "Quantity sold per day/store/product. Target variable for forecasting.", "Source": "POS System"},
    {"Column": "Price", "Type": "Numeric", "Description": "Listed unit price (Shelf Price).", "Source": "POS System"},
    {"Column": "Discount", "Type": "Numeric", "Description": "Discount amount or percentage applied at checkout.", "Source": "POS System"},
    {"Column": "Competitor Pricing", "Type": "Numeric", "Description": "Price of matching item at primary competitor.", "Source": "Competitive Intel"},
    {"Column": "Holiday", "Type": "Binary/Cat", "Description": "Flag indicating national or regional holiday.", "Source": "Calendar"},
    {"Column": "BOGO", "Type": "Binary", "Description": "Buy-One-Get-One promotion active.", "Source": "Marketing Calendar"}
]
st.table(pd.DataFrame(raw_data))

# 2. Engineered Features
st.header("2. Engineered Features (Model Inputs)")
eng_data = [
    {"Feature": "effective_price", "Logic": "Price - Discount (if absolute) or Price * (1-Discount_pct)", "Usage": "Primary driver for elasticity models. Represents actual customer cost."},
    {"Feature": "price_gap_to_competitor", "Logic": "effective_price - Competitor Pricing", "Usage": "Measures relative value. Positive = Expensive, Negative = Cheaper."},
    {"Feature": "Units Sold_lag_X", "Logic": "Value of Units Sold X days ago", "Usage": "Captures autocorrelation and recent trend."},
    {"Feature": "Units Sold_roll_mean_Y", "Logic": "Moving average of Units Sold over last Y days", "Usage": "Captures baseline level / smooth trend."},
    {"Feature": "promo_flag", "Logic": "1 if (Discount > 0 OR BOGO=1), else 0", "Usage": "Binary indicator for 'On Promotion' status."},
    {"Feature": "Month / DayOfWeek", "Logic": "Extracted from Date", "Usage": "Captures seasonality (Weekly/Annual cycles)."},
    {"Feature": "weather_cluster", "Logic": "K-Means cluster of Temp/Precip (if available)", "Usage": "Control for external demand factors."}
]
st.table(pd.DataFrame(eng_data))

# 3. Model Outputs
st.header("3. Model Output Definitions")
out_data = [
    {"Metric": "Elasticity", "Definition": "% Change in Quantity / % Change in Price. Measures price sensitivity.", "Range": "Typically -5.0 to 0.0"},
    {"Metric": "Forecast (yhat)", "Definition": "Predicted Units Sold for future period.", "Range": "0 to Max Capacity"},
    {"Metric": "Baseline", "Definition": "Forecast assuming NO changes (Status Quo).", "Range": "N/A"},
    {"Metric": "Experiment Lift", "Definition": "(Treatment - Control) / Control. The causal impact of an intervention.", "Range": "-100% to +inf%"}
]
st.table(pd.DataFrame(out_data))

render_page_footer("Home", "Hello", "Refrence complete.")
