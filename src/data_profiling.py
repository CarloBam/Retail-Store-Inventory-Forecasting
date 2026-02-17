import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def render_correlation_matrix(df):
    """
    Renders correlation matrix for numeric columns.
    """
    st.subheader("🔗 Correlation Analysis")
    
    # Select numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    
    if len(numeric_df.columns) < 2:
        st.info("Not enough numeric variables for correlation analysis.")
        return

    # Compute correlation
    corr = numeric_df.corr()
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax)
    st.pyplot(fig)
    
    # Check for high correlation
    high_corr_pairs = []
    
    # Iterate upper triangle
    for i in range(len(corr.columns)):
        for j in range(i+1, len(corr.columns)):
            val = corr.iloc[i, j]
            if abs(val) > 0.8:
                high_corr_pairs.append((corr.columns[i], corr.columns[j], val))
                
    if high_corr_pairs:
        st.warning("⚠️ High Multi-collinearity Detected:")
        for v1, v2, val in high_corr_pairs:
            st.write(f"- **{v1}** and **{v2}**: {val:.2f}")
            
def render_variable_drilldown(df):
    """
    Renders a drill-down section for each variable with stats, plots, and actions.
    """
    st.subheader("🕵️‍♀️ Variable Drill-Down")
    
    for col in df.columns:
        # Create an expander for each variable
        with st.expander(f"Variable: **{col}** ({df[col].dtype})"):
            
            # Layout: Stats | Plot
            c1, c2 = st.columns([1, 2])
            
            # --- 1. Summary Stats ---
            with c1:
                st.markdown("#### Summary Stats")
                
                # Basic info
                null_count = df[col].isnull().sum()
                unique_count = df[col].nunique()
                
                st.write(f"**Missing:** {null_count} ({null_count/len(df):.1%})")
                st.write(f"**Unique:** {unique_count}")
                
                if pd.api.types.is_numeric_dtype(df[col]):
                    desc = df[col].describe()
                    st.write(f"**Mean:** {desc['mean']:.2f}")
                    st.write(f"**Std:** {desc['std']:.2f}")
                    st.write(f"**Min:** {desc['min']}")
                    st.write(f"**Max:** {desc['max']}")
                elif pd.api.types.is_datetime64_any_dtype(df[col]):
                    st.write(f"**Min Date:** {df[col].min()}")
                    st.write(f"**Max Date:** {df[col].max()}")
                else:
                    # Categorical / Object
                    top_val = df[col].mode()[0] if not df[col].mode().empty else "N/A"
                    st.write(f"**Mode:** {top_val}")
            
            # --- 2. Plots ---
            with c2:
                # Chart Type Selector
                chart_types = ["Distribution"]
                if pd.api.types.is_numeric_dtype(df[col]):
                    chart_types = ["Histogram", "Box Plot"]
                
                chart_type = st.selectbox(f"Chart Type ({col})", chart_types, key=f"chart_type_{col}", label_visibility="collapsed")
                
                fig, ax = plt.subplots(figsize=(6, 2.5))
                
                if pd.api.types.is_numeric_dtype(df[col]):
                    if chart_type == "Histogram":
                        sns.histplot(df[col], kde=True, ax=ax, color='#4c72b0')
                        ax.set_title(f"Distribution of {col}", fontsize=10)
                    elif chart_type == "Box Plot":
                        sns.boxplot(x=df[col], ax=ax, color='#55a868')
                        ax.set_title(f"Box Plot of {col}", fontsize=10)
                        
                elif pd.api.types.is_datetime64_any_dtype(df[col]):
                     # Time series count
                     df[col].dt.date.value_counts().sort_index().plot(kind='line', ax=ax)
                     ax.set_title("Timeline Frequency")
                     
                else:
                    # Categorical - Bar Chart of Top 10
                    top_n = df[col].value_counts().head(10)
                    sns.barplot(x=top_n.values, y=top_n.index, ax=ax, palette="viridis")
                    ax.set_title(f"Top 10 Categories in {col}", fontsize=10)
                
                plt.tight_layout()
                st.pyplot(fig)
            
            # --- 3. Actions (Missing Values) ---
            if null_count > 0:
                st.markdown("---")
                st.markdown("#### 🛠️ Fix Missing Values")
                
                c_act1, c_act2 = st.columns([2, 1])
                with c_act1:
                     method = st.selectbox(f"Method for {col}", ["Drop Rows", "Fill with Mean/Mode", "Fill with Zero/Unknown"], key=f"miss_mtd_{col}")
                
                with c_act2:
                    if st.button("Apply Fix", key=f"btn_fix_{col}"):
                        if method == "Drop Rows":
                            df.dropna(subset=[col], inplace=True)
                            st.toast(f"Dropped rows with missing {col}")
                        elif method == "Fill with Mean/Mode":
                            if pd.api.types.is_numeric_dtype(df[col]):
                                val = df[col].mean()
                                df[col].fillna(val, inplace=True)
                            else:
                                val = df[col].mode()[0]
                                df[col].fillna(val, inplace=True)
                            st.toast(f"Filled {col} with {val}")
                        elif method == "Fill with Zero/Unknown":
                            if pd.api.types.is_numeric_dtype(df[col]):
                                df[col].fillna(0, inplace=True)
                            else:
                                df[col].fillna("Unknown", inplace=True)
                            st.toast(f"Filled {col} with 0/Unknown")
                        
                        # Update session state and rerun
                        st.session_state['cleaned_data'] = df # Update cleaned data in place
                        st.rerun()

