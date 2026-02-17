import streamlit as st
import sys
import os
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.pricing import prepare_pricing_data, fit_elasticity_model, fit_competitor_model
from src.forecasting import prepare_time_series # Optional if needed, but we focus on aggregate insights
from src.ui_components import render_sidebar_progress, render_page_footer, simple_header
from src.navigation import render_top_navigation

st.set_page_config(page_title="Strategic Recommendations", layout="wide", page_icon="📋")

# Top Nav
render_top_navigation("Data Science View", "08_Data_Science_View", 
                      can_proceed=True,
                      warning_msg="")

render_sidebar_progress()
simple_header("Executive Summary & Recommendations", "📋")

# 1. Data Access
if 'feature_data' in st.session_state and st.session_state['feature_data'] is not None:
    df = st.session_state['feature_data'].copy()
elif 'cleaned_data' in st.session_state and st.session_state['cleaned_data'] is not None:
    df = st.session_state['cleaned_data'].copy()
else:
    st.error("Data not loaded. Please go to Data Upload.")
    st.stop()

if 'Category' not in df.columns:
    st.error("Analysis requires 'Category' column.")
    st.stop()

# --- Helpers to Generate Insights on the Fly ---
# We re-run lightweight models to ensure freshness without requiring user to visit previous pages first

@st.cache_data
def get_strategic_insights(df):
    insights = {}
    
    # 1. Elasticity Logic
    # Default config for insights
    cfg = {'agg_level': 'Weekly', 'exclude_promo': False, 'max_discount_depth': 50}
    df_price = prepare_pricing_data(df, cfg)
    elast_df = fit_elasticity_model(df_price, category_col='Category', fixed_effects=['Store', 'Region'])
    
    if not elast_df.empty:
        # Elastic (-inf to -1) -> Sensitive -> Opportunity to lower price to gain share? Or risky to raise.
        # Inelastic (-1 to 0) -> Insensitive -> Opportunity to raise price for revenue.
        
        insights['elastic_df'] = elast_df
        insights['top_elastic'] = elast_df[elast_df['Elasticity'] < -1].sort_values('Elasticity')
        insights['top_inelastic'] = elast_df[(elast_df['Elasticity'] > -1) & (elast_df['Elasticity'] < 0)].sort_values('Elasticity', ascending=False)
    
    # 2. Promo Uplift
    # Simple agg: Avg Sales (Promo=1) / Avg Sales (Promo=0) - 1
    if 'promo_flag' in df.columns:
        # Aggregate by category
        promo_stats = df.groupby(['Category', 'promo_flag'])['Units Sold'].mean().unstack()
        if 1 in promo_stats.columns and 0 in promo_stats.columns:
            promo_stats['uplift_pct'] = ((promo_stats[1] - promo_stats[0]) / promo_stats[0]) * 100
            insights['promo_uplift'] = promo_stats['uplift_pct'].sort_values(ascending=False)
            
    # 3. Competitor Sensitivity
    if 'price_gap_to_competitor' in df.columns:
        model_comp, err = fit_competitor_model(df_price)
        if model_comp:
            insights['competitor_coef'] = model_comp.params['price_gap_to_competitor']
            insights['competitor_pval'] = model_comp.pvalues['price_gap_to_competitor']

    return insights

# Execute
with st.spinner("Synthesizing Strategic Insights..."):
    insights = get_strategic_insights(df)

# --- Layout ---

# Section 1: Key Findings
st.header("1. Key Market Findings")

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("### 🏷️ Price Sensitivity")
    if 'top_inelastic' in insights and not insights['top_inelastic'].empty:
        top_inel = insights['top_inelastic'].head(3)
        st.success("**Top Inelastic Categories** (Customers accept price hikes):")
        for _, row in top_inel.iterrows():
            st.write(f"- **{row['Category']}** (E={row['Elasticity']:.2f})")
    else:
        st.info("No clearly inelastic categories identified.")

    if 'top_elastic' in insights and not insights['top_elastic'].empty:
        st.markdown("---")
        st.warning("**Top Elastic Categories** (Customers flee price hikes):")
        top_el = insights['top_elastic'].head(3)
        for _, row in top_el.iterrows():
            st.write(f"- **{row['Category']}** (E={row['Elasticity']:.2f})")

with c2:
    st.markdown("### 💥 Promotion Impact")
    if 'promo_uplift' in insights:
        top_uplift = insights['promo_uplift'].head(3)
        st.info("**Highest Promo Responders:**")
        for cat, val in top_uplift.items():
            st.write(f"- **{cat}**: +{val:.0f}% Uplift")
            
        st.markdown("---")
        low_uplift = insights['promo_uplift'].tail(3)
        st.write("**Lowest Promo Responders (Re-evaluate):**")
        for cat, val in low_uplift.items():
            st.write(f"- **{cat}**: {val:.0f}% Uplift")
    else:
        st.warning("Promo data unavailable.")

with c3:
    st.markdown("### ⚔️ Competitive Landscape")
    if 'competitor_coef' in insights:
        coef = insights['competitor_coef']
        pval = insights['competitor_pval']
        if pval < 0.05:
            impact = "Strong" if abs(coef) > 2 else "Moderate"
            direction = "gain" if coef > 0 else "lose"
            st.metric("Competitor Sensitivity", impact)
            st.write(f"We statistically **{direction}** {abs(coef):.1f} units for every $1 price gap advantage.")
        else:
            st.metric("Competitor Sensitivity", "Low/None")
            st.write("Competitor pricing does not significantly drive our volume.")
    else:
        st.info("Competitor data unavailable.")

# Section 2: Recommended Actions
st.markdown("---")
st.header("2. Strategic Experiments & Actions")

recommendations = []

# Logic for Recommendations
# 1. Price Hike Experiment
if 'top_inelastic' in insights and not insights['top_inelastic'].empty:
    target_cat = insights['top_inelastic'].iloc[0]['Category']
    recommendations.append({
        "Action": f"Increase Price on '{target_cat}'",
        "Type": "Profit Optimization",
        "Rationale": "Demand is inelastic. A small price increase should yield pure profit margin expansion with minimal volume loss.",
        "Test Design": "A/B Test: Select 20 comparable stores. Increase price by 3% in 10 test stores for 4 weeks. Measure Gross Margin $."
    })

# 2. Promo Rationalization
if 'promo_uplift' in insights:
    # Find low uplift category
    low_performers = insights['promo_uplift'][insights['promo_uplift'] < 10] # Less than 10% uplift
    if not low_performers.empty:
        target_cat = low_performers.index[0]
        recommendations.append({
            "Action": f"Reduce/Stop Promos on '{target_cat}'",
            "Type": "Cost Saving",
            "Rationale": f"Promo uplift is negligible (<10%). Discounts are likely cannibalizing full-price sales.",
            "Test Design": "Holdout Test: Stop discounting in 1 region. Measure total Category Revenue vs Control region."
        })

# 3. Volume Driver
if 'top_elastic' in insights and not insights['top_elastic'].empty:
    target_cat = insights['top_elastic'].iloc[0]['Category']
    recommendations.append({
        "Action": f"Strategic Price Cut/Bundle on '{target_cat}'",
        "Type": "Volume/Share Growth",
        "Rationale": "Highly elastic demand. Price cuts will significantly boost volume and traffic, potentially driving basket size.",
        "Test Design": "Discount Test: 10% off for 2 weeks. Measure Unit Velocity and Basket Penetration."
    })

# 4. Inventory/Stock (Generic)
recommendations.append({
    "Action": "Review Safety Stock Levels for Top Sellers",
    "Type": "Availability",
    "Rationale": "Ensure high-velocity SKUs (Top 10%) have 98%+ service levels to maximize realized revenue.",
    "Test Design": "Audit: Compare 'Units Ordered' vs 'Units Sold' gap for top decile products."
})

# 5. Competitor Response
if 'competitor_coef' in insights and insights['competitor_coef'] > 0:
    recommendations.append({
        "Action": "Dynamic Matching Rules",
        "Type": "Competitive Defense",
        "Rationale": "Volume is sensitive to price gaps. Automate pricing to maintain a fixed gap (e.g. +$0.50) vs primary competitor.",
        "Test Design": "Pilot dynamic pricing rules on 50 SKUs."
    })

# Render Recommendations
for i, rec in enumerate(recommendations, 1):
    with st.expander(f"Recommendation #{i}: {rec['Action']}", expanded=(i==1)):
        c_a, c_b = st.columns([1, 2])
        with c_a:
            st.write(f"**Type:** {rec['Type']}")
            st.write(f"**Rationale:** {rec['Rationale']}")
        with c_b:
            st.info(f"🧪 **Suggested Test:** {rec['Test Design']}")

# Section 3: Risks & Caveats
st.markdown("---")
st.header("3. Risks & Methodological Caveats")
st.warning("""
- **Observational Bias**: Elasticity is estimated from historical data, not randomized controlled trials. Factors like unobserved marketing or seasonality might confound results.
- **Supply Constraints**: If 'Units Sold' was limited by out-of-stocks, true demand is under-estimated, making demand appear less elastic than it is.
- **Price Range Limit**: Models are valid only within the historical price range. Extrapolating elasticity to new price points (e.g. +50%) is unreliable.
- **Cannibalization**: Estimates largely treat products/categories in isolation and do not fully account for cross-elasticity (switching between products).
""")

# Section 4: Export
st.markdown("---")
st.header("4. Export Report")

# Create Export Content
report_md = f"""# Executive Pricing & Forecasting Report
*Generated on {pd.Timestamp.now().date()}*

## Key Findings
- **Inelastic Categories**: {', '.join(insights['top_inelastic']['Category'].tolist()[:3]) if 'top_inelastic' in insights else 'None'}
- **Elastic Categories**: {', '.join(insights['top_elastic']['Category'].tolist()[:3]) if 'top_elastic' in insights else 'None'}

## Top Recommendations
"""
for i, rec in enumerate(recommendations, 1):
    report_md += f"{i}. **{rec['Action']}**: {rec['Rationale']}\n"

col_ex1, col_ex2, col_ex3 = st.columns(3)

with col_ex1:
    st.download_button(
        label="📥 Download Executive Brief (MD)",
        data=report_md,
        file_name="executive_summary.md",
        mime="text/markdown"
    )

with col_ex2:
    if 'elastic_df' in insights:
        csv = insights['elastic_df'].to_csv(index=False)
        st.download_button(
            label="📥 Download Elasticity Data (CSV)",
            data=csv,
            file_name="elasticity_estimates.csv",
            mime="text/csv"
        )

with col_ex3:
    # Dynamic PPTX Generation
    try:
        from src.reporting import create_ppt_presentation
        
        # We need to pass the dataframe, insights, and recommendations
        # insights and recommendations are already computed above
        
        ppt_buffer = create_ppt_presentation(df, insights, recommendations)
        
        st.download_button(
            label="📊 Download Slides (PPTX)",
            data=ppt_buffer,
            file_name="Retail_Strategy_Deck.pptx",
            mime="application/vnd.openxmlformats-officedocument.presentationml.presentation"
        )
    except ImportError:
        st.error("python-pptx library not found. Please install it.")
    except Exception as e:
        st.error(f"Failed to generate PPTX: {e}")

    # Word and PDF
    col_w, col_p = st.columns(2)
    with col_w:
        try:
            from src.reporting_exports import create_word_report
            docx = create_word_report(df, insights, recommendations)
            st.download_button("📝 Download Word Report", docx, "Retail_Report.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
        except Exception as e:
            st.warning(f"Word export unavailable: {e}")
            
    with col_p:
        try:
            from src.reporting_exports import create_pdf_report
            pdf_data = create_pdf_report(df, insights, recommendations)
            st.download_button("📄 Download PDF Report", pdf_data, "Retail_Report.pdf", "application/pdf")
        except Exception as e:
            st.warning(f"PDF export unavailable: {e}")

render_page_footer("Data Science Deep Dive", "08_Data_Science_View", "View the math behind the models.")
