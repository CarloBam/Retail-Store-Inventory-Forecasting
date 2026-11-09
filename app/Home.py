import streamlit as st
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_io import initialize_session_state
from src.ui_components import render_sidebar_progress, render_page_footer

st.set_page_config(
    page_title="Retail Forecasting App",
    page_icon="📈",
    layout="wide"
)

initialize_session_state()
render_sidebar_progress()

st.title("Retail Price Elasticity and Demand Forecasting")

st.markdown("""
### 🛍️ Retail Inventory & Demand Optimization

**Business Goal:** Understand how **price changes affect demand** across product categories and optimize inventory levels to maximize profitability while maintaining service levels.

**Data Source:** [Retail Store Inventory Forecasting Dataset (Kaggle)](https://www.kaggle.com/datasets/anirudhchauhan/retail-store-inventory-forecasting-dataset) · synthetic

#### Workflow & Models:
1. **Data Upload**: Load and validate your CSV data.
2. **Data Cleaning**: Handle missing values and outliers (Winsorization, Imputation).
3. **Feature Engineering**: Create lags, rolling means, and seasonality features.
4. **EDA**: Visual exploration of sales trends and correlations.
5. **Forecasting**: Predict future demand using:
    - **Prophet** (Trend + Seasonality decompostion)
    - **SARIMA** (Statistical Auto-Regression)
    - **XGBoost / Random Forest** (Machine Learning approach)
6. **Pricing Analysis**: Estimate Price Elasticity to simulate revenue under different pricing strategies.

#### Getting Started
No data? Use the sample dataset button on the Data Upload page.
Click the button below to begin importing your data.
""")

if st.button("Start Analysis 🚀", type="primary"):
    st.switch_page("pages/01_Data_Upload.py")

st.sidebar.markdown("---")
st.sidebar.subheader("Global Settings")
seed = st.sidebar.number_input("Random Seed", value=st.session_state['config']['random_seed'])
st.session_state['config']['random_seed'] = seed

# Setup styling
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .sidebar .sidebar-content {
        background: #ffffff;
    }
</style>
""", unsafe_allow_html=True)
