
import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import os
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.ui_components import render_sidebar_progress, render_page_footer, simple_header
from src.navigation import render_top_navigation

st.set_page_config(page_title="Experimentation Roadmap", layout="wide", page_icon="🧪")

# Top Nav
render_top_navigation("Data Dictionary", "11_Data_Dictionary", can_proceed=True)

render_sidebar_progress()
simple_header("Experimentation & Causal Inference Roadmap", "🧪")

st.markdown("""
To move from **Correlation** (Model Associations) to **Causality** (True Driver Identification), 
we recommend a phased experimentation approach. This roadmap guides the organization from observational studies to rigorous randomized trials.
""")

# --- Phase 1: Timeline ---
st.header("1. Strategic Roadmap")

roadmap_data = [
    {"Phase": "Phase 1: Observational inference", "Timeline": "Months 1-2", "Key Activities": "Prophet/Regression Models, Placebo Tests, Granger Causality", "Goal": "Hypothesis Generation & Correlation baselines"},
    {"Phase": "Phase 2: Quasi-Experiments", "Timeline": "Months 3-4", "Key Activities": "Difference-in-Differences (DiD), Synthetic Controls on historical localized events (e.g., regional promos)", "Goal": "Validate impact of past decisions with stronger causal claims"},
    {"Phase": "Phase 3: Controlled Pilots (A/B)", "Timeline": "Months 5-6", "Key Activities": "Randomized Pricing Tests (Store Clusters), Holdout Groups", "Goal": "Gold-standard measurement of Price Elasticity"},
    {"Phase": "Phase 4: Automated Bandits", "Timeline": "Month 6+", "Key Activities": "Multi-armed Contextual Bandits for personalized pricing/promos", "Goal": "Continuous optimization"}
]

st.table(pd.DataFrame(roadmap_data))

# --- Phase 2: Sample Size Calculator ---
st.markdown("---")
st.header("2. A/B Test Design (Sample Size Calculator)")
st.caption("Plan your next Pricing Pilot. How many stores/days do you need to detect a lift?")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Parameters")
    # Base Conversion / Volume
    baseline_mean = st.number_input("Baseline Daily Units per Store", value=100)
    baseline_std = st.number_input("Standard Deviation (Day-to-Day Volatility)", value=20)
    
    # Effect Size
    mde_pct = st.slider("Minimum Detectable Effect (Lift %)", 1, 25, 5)
    
    # Statistical Power
    alpha = st.selectbox("Significance Level (Alpha)", [0.01, 0.05, 0.10], index=1)
    power = st.selectbox("Statistical Power (1-Beta)", [0.8, 0.9, 0.95], index=0)

with col2:
    st.subheader("Required Sample Size")
    
    # Simple calculation for T-test (Two-sided)
    # n = (2 * sigma^2 * (z_alpha + z_beta)^2) / delta^2
    
    delta = baseline_mean * (mde_pct / 100.0)
    variance = baseline_std ** 2
    
    z_alpha = stats.norm.ppf(1 - alpha/2)
    z_beta = stats.norm.ppf(power)
    
    n_required = (2 * variance * (z_alpha + z_beta)**2) / (delta**2)
    n_required = int(np.ceil(n_required))
    
    st.metric("Required Sample Size (Store-Days)", f"{n_required}")
    
    st.info(f"""
    To detect a **{mde_pct}%** lift (from {baseline_mean} to {baseline_mean + delta:.1f}):
    - You need **{n_required} store-days** per group (Control & Test).
    - If you have 10 test stores, run the test for **{int(np.ceil(n_required/10))} days**.
    """)

# --- Phase 3: Causal Methods explainer ---
st.markdown("---")
st.header("3. Methodological Guide (Refresher)")

tab1, tab2, tab3 = st.tabs(["Difference-in-Differences (DiD)", "Synthetic Control", "Regression Discontinuity"])

with tab1:
    st.markdown("### Difference-in-Differences (DiD)")
    st.write("**Use Case:** Measuring the impact of a policy change (e.g., new tax, regional price hike) that affected one group but not another.")
    st.write("**Key Assumption:** 'Parallel Trends' - The treatment and control groups would have moved effectively in parallel if the intervention hadn't happened.")
    st.latex(r'''
        \text{Impact} = (\text{Treat}_{post} - \text{Treat}_{pre}) - (\text{Control}_{post} - \text{Control}_{pre})
    ''')
    
with tab2:
    st.markdown("### Synthetic Control")
    st.write("**Use Case:** When you have only ONE treated unit (e.g., you changed price in California only).")
    st.write("**Method:** Construct a weighted combination of 'untreated' units (e.g., 0.3*Nevada + 0.5*Arizona + 0.2*Oregon) that creates a 'Synthetic California' which mimics real California perfectly in the pre-period.")
    st.info("This is generally more robust than DiD for single-unit interventions.")

with tab3:
    st.markdown("### Regression Discontinuity (RDD)")
    st.write("**Use Case:** When an intervention is assigned based on a strict cutoff (e.g., 'Stores with >1000 sqft get the promo').")
    st.write("**Method:** Compare stores just above (1001 sqft) and just below (999 sqft) the cutoff. They are likely very similar, so any jump in sales at the cutoff is causal.")

render_page_footer("Data Dictionary", "11_Data_Dictionary", "View full definitions.")
