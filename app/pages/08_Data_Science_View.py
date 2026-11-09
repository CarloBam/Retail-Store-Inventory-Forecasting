import streamlit as st
import sys
import os
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.pricing import prepare_pricing_data, fit_elasticity_model
from src.ui_components import render_sidebar_progress, render_page_footer, simple_header
from src.navigation import render_top_navigation

st.set_page_config(page_title="Data Science Deep Dive", layout="wide", page_icon="🧬")

# Top Nav - Back to Home or End
render_top_navigation("Home", "Home", can_proceed=True)

render_sidebar_progress()
simple_header("Data Science Deep Dive: The Core Model", "🧬")

# 1. Model Definition (Math)
st.header("1. The Core Elasticity Model")

st.markdown("""
We use **log-log regression** (also known as a constant elasticity model or power model) to estimate price sensitivity. 
This functional form is industry standard for retail demand modeling because coefficients can be directly interpreted as **elasticities** (percentage changes).
""")

st.latex(r'''
\log(\text{Units}) = \alpha + \beta \cdot \log(\text{Effective Price}) + \gamma_1 \cdot \text{Promo} + \gamma_2 \cdot \text{Discount\%} + \gamma_3 \cdot \text{Holiday} + \text{Controls} + \epsilon
''')

# 2. Term Explanation Table
st.header("2. Term Dictionary")

terms_data = [
    {
        "Term": "log(Units)",
        "Statistical Meaning": "Natural log of dependent variable. Stabilizes variance (homoscedasticity).",
        "Business Meaning": "The magnitude of demand. Mapping changes to %, not absolute units.",
        "Estimation": "Observed Data",
        "Common Pitfalls": "Zero sales days (log(0) is undefined) require handling (filter or log1p)."
    },
    {
        "Term": "α (Alpha)",
        "Statistical Meaning": "Intercept.",
        "Business Meaning": "Baseline demand scale when predictors are zero (theoretical).",
        "Estimation": "OLS Parameter",
        "Common Pitfalls": "Often not interpretable on its own in log space."
    },
    {
        "Term": "β (Beta)",
        "Statistical Meaning": "Coefficient of log(Price). The Elasticity.",
        "Business Meaning": "**The Price Elasticity**. % Change in Demand for 1% Change in Price.",
        "Estimation": "OLS Parameter",
        "Common Pitfalls": "Endogeneity: High prices might correlate with high demand periods (reverse causality) biasing β towards 0."
    },
    {
        "Term": "γ (Gamma terms)",
        "Statistical Meaning": "Coefficients for dummy/continuous shifters.",
        "Business Meaning": "Uplift factors. E.g., Promo flag captures marketing pressure beyond just price cut.",
        "Estimation": "OLS Parameter",
        "Common Pitfalls": "Multi-collinearity: If Price Drop always happens with Promo Flag, hard to separate β and γ."
    },
    {
        "Term": "Controls",
        "Statistical Meaning": "Fixed Effects (Categories, Month, Store).",
        "Business Meaning": "Accounting for structural differences (e.g. Dec is always busy, Store A is huge).",
        "Estimation": "Dummy Variables",
        "Common Pitfalls": "Overfitting if too many levels (e.g. Product ID fixed effects with sparse data)."
    },
    {
        "Term": "ε (Epsilon)",
        "Statistical Meaning": "Residual / Error Term.",
        "Business Meaning": "Unexplained variance (Random noise + Missing factors like Competitor moves).",
        "Estimation": "Residuals",
        "Common Pitfalls": "Autocorrelation: If errors are correlated over time, confidence intervals are too narrow."
    }
]

st.table(pd.DataFrame(terms_data))

# 3. Assumptions & Diagnostics
st.header("3. Critical Assumptions & Diagnostics")

c1, c2 = st.columns(2)

with c1:
    st.markdown("### 🏗️ Assumptions")
    st.markdown("""
    1. **Functional Form**: We assume the relationship is linear in logs. A 10% price drop has the same relative impact whether price is \\$10 or \\$100.
    2. **Independence**: We assume daily sales errors are independent (unless we use robust errors/Clustering). In reality, demand has temporal memory.
    3. **Exogeneity**: We assume Price is set independently of the error term. If we crash prices *because* we know demand will be low (clearance), β is biased.
    """)

with c2:
    st.markdown("### 🩺 Diagnostics Used")
    st.markdown("""
    - **R-Squared**: % of variance explained. Low R² isn't bad for causal inference, but bad for prediction.
    - **P-Values**: Probability that the relationship is random noise. We look for p < 0.05.
    - **Confidence Intervals**: The range where the true Elasticity likely lives. Wide intervals = High uncertainty.
    - **Robust Errors (HC1)**: We apply heteroscedasticity-consistent standard errors to handle volatile variance in sales data.
    """)

# 4. Interactive Walkthrough
st.header("4. Live Model Inspection")
st.markdown("Select a category to see the **Real Coefficient** estimated from your data.")

# Load Data for Walkthrough
if 'feature_data' in st.session_state and st.session_state['feature_data'] is not None:
    df = st.session_state['feature_data'].copy()
elif 'cleaned_data' in st.session_state and st.session_state['cleaned_data'] is not None:
    df = st.session_state['cleaned_data'].copy()
else:
    st.error("Please load data to enable the walkthrough.")
    st.stop()

if 'Category' in df.columns:
    cats = df['Category'].unique()
    selected_cat = st.selectbox("Inspect Category:", cats)
    
    # Fit tiny model for just this cat
    # Prepare pricing data and fit a model for the selected category.
    
    with st.spinner("Fitting specific model..."):
        from src.pricing import prepare_pricing_data, interpret_elasticity
        
        # Prepare
        df_mod = prepare_pricing_data(df, {'agg_level': 'Weekly', 'exclude_promo': False}) # Use Weekly for stability in demo
        sub_df = df_mod[df_mod['Category'] == selected_cat]
        
        if len(sub_df) > 10:
            import statsmodels.formula.api as smf
            
            # Formula
            # Ensure month column exists for seasonality control
            if 'Month' not in sub_df.columns:
                 if 'month' in sub_df.columns:
                      sub_df['Month'] = sub_df['month']
                 elif 'Date' in sub_df.columns:
                      sub_df['Month'] = sub_df['Date'].dt.month.astype(str) # categorical
            
            formula = "log_demand ~ log_price + C(Month)" # Use C() for categorical treatment
            if 'promo_flag' in sub_df.columns: formula += " + promo_flag"
            
            model = smf.ols(formula, data=sub_df).fit(cov_type='HC1')
            
            beta = model.params['log_price']
            ci = model.conf_int().loc['log_price']
            pval = model.pvalues['log_price']
            
            # Display
            col_res1, col_res2 = st.columns([1, 2])
            
            with col_res1:
                st.metric("β (Price Elasticity)", f"{beta:.2f}", delta_color="off")
                st.caption(f"95% CI: [{ci[0]:.2f}, {ci[1]:.2f}]")
                
            with col_res2:
                st.subheader("Interpretation")
                interp_text = interpret_elasticity(beta, pval)
                st.info(f"**{interp_text}**")
                
                direction = "decrease" if beta < 0 else "increase"
                st.write(f"""
                For every **10% increase** in Effective Price for **{selected_cat}**, 
                we expect Units Sold to **{direction}** by approx **{abs(beta)*10:.1f}%**, 
                holding seasonality and promo activity constant.
                """)
                
                st.markdown(f"*Model Confidence (P-Value): {pval:.4f}*")
        else:
            st.warning("Not enough data points for this category.")
else:
    st.warning("No Category column found.")


# 5. Classification & Model Assessment (ROC Curves)
st.header("5. Model Assessment: Classification & ROC")
st.markdown("""
To provide a robust comparison of model performance (e.g. for a "High Demand" vs "Low Demand" classification task), 
we analyze **ROC Curves** and **Statistical Coefficients**.
""")

if st.button("Run Classification Analysis (ROC & Stats)"):
    from src.analysis import train_classification_models, plot_roc_curves, get_model_stats
    
    # --- Slicers ---
    col_filt1, col_filt2 = st.columns(2)
    with col_filt1:
        # Year Slicer
        if 'Year' not in df.columns and 'Date' in df.columns:
             df['Year'] = pd.to_datetime(df['Date']).dt.year
             
        all_years = sorted(df['Year'].unique().tolist()) if 'Year' in df.columns else []
        selected_years = st.multiselect("Select Years (Time Period)", all_years, default=all_years)
        
    with st.spinner("Training Classification Models (Logistic, Random Forest, XGB)..."):
        # Use feature_data if available, else cleaned_data
        target_df = df.copy()
        
        # Apply Year Filter
        if selected_years:
            target_df = target_df[target_df['Year'].isin(selected_years)]
            st.caption(f"Analysis Period: {min(selected_years)} to {max(selected_years)} ({len(target_df):,} records)")
        else:
            st.warning("Please select at least one year.")
            st.stop()
        
        # Train
        models, X_train, X_val, X_test, y_train, y_val, y_test = train_classification_models(target_df)
        
        # 1. ROC Curves
        st.subheader("ROC Curves")
        figs = plot_roc_curves(models, X_train, y_train, X_val, y_val, X_test, y_test)
        
        col_roc1, col_roc2, col_roc3 = st.columns(3)
        with col_roc1: st.plotly_chart(figs['Train'], use_container_width=True)
        with col_roc2: st.plotly_chart(figs['Validation'], use_container_width=True)
        with col_roc3: st.plotly_chart(figs['Test'], use_container_width=True)
        
        # 2. Statistical Table (SAS Style)
        st.subheader("Statistical Report (Logistic Regression)")
        st.markdown("**Model Coefficients & P-Values** (Similar to SAS Enterprise Miner Output)")
        
        # Get stats for the Logistic Regression model
        # Rebuild the feature set used by the selected model.
        feature_cols = X_train.columns.tolist()
        stats_df = get_model_stats(target_df, 'Units Sold', feature_cols)
        
        # Display as a styled dataframe
        st.dataframe(stats_df.style.format({
            "Coefficient": "{:.4f}",
            "Std. Error": "{:.4f}", 
            "z-Score": "{:.2f}",
            "P-Value": "{:.5f}"
        }).background_gradient(subset=['P-Value'], cmap="Reds_r", vmin=0, vmax=0.1))
        
        st.success("Analysis Complete.")

# Debug Section (Optional but good practice for "Data Science View")
st.markdown("---")
with st.expander("🛠️ Debug: Session State & System Info"):
    st.json({k: str(v)[:200] for k, v in st.session_state.items() if k not in ['raw_data', 'cleaned_data', 'feature_data']})
    if 'cleaned_data' in st.session_state:
        st.write(f"Cleaned Data Shape: {st.session_state['cleaned_data'].shape}")

render_page_footer("Trust & QA Report", "09_Trust_Report", "Verify pipeline integrity.")
