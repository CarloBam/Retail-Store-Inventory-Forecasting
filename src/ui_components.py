import streamlit as st

def render_sidebar_progress():
    """
    Renders a progress tracker in the sidebar based on session state.
    """
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🚦 Workflow Progress")
    
    # Define Steps and their completion keys
    steps = [
        ("1. Data Upload", 'raw_data'),
        ("2. Cleaning", 'cleaned_data'),
        ("3. Feature Eng", 'feature_data'),
        ("4. EDA", 'feature_data'), # EDA usually depends on features or cleaned
        ("5. Forecasting", None), # Forecasting is an activity, hard to track 'completion' unless we store a model
        ("6. Pricing", None),
        ("7. Recommendations", None)
    ]
    
    for label, key in steps:
        if key and key in st.session_state and st.session_state[key] is not None:
             st.sidebar.markdown(f"✅ **{label}**")
        elif key is None:
             # Activity status is not tracked in session state.
             pass 
    
    def status(key):
        return "✅" if (key in st.session_state and st.session_state[key] is not None) else "⬜"
        
    md = f"""
    {status('raw_data')} Data Upload  
    {status('cleaned_data')} Quality & Cleaning  
    {status('feature_data')} Feature Engineering  
    """
    st.sidebar.markdown(md)
    
    if st.session_state.get('feature_data') is not None:
        st.sidebar.success("Ready for Analysis!")
    else:
        st.sidebar.info("Complete steps to unlock analysis.")
        
    st.sidebar.markdown("---")

def render_page_footer(next_page_name, next_page_link, current_status_msg=None):
    """
    Renders a footer with a link to the next page.
    """
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if current_status_msg:
            st.info(f"ℹ️ {current_status_msg}")
            
    with col2:
        st.markdown(f"### Next Step ➡️")
        
        # Build the relative path to the selected page.
        page_path = f"pages/{next_page_link}.py"
        if next_page_link == "Home":
             page_path = "Home.py"
             
        try:
            st.page_link(page_path, label=next_page_name, icon="👉")
        except AttributeError: 
             # Fallback for older streamlit versions
            st.markdown(f"**Go to [{next_page_name}](/{next_page_link})**")
        except Exception:
            # Fallback if path not found
            st.info(f"Please navigate to **{next_page_name}** in the sidebar.")


def simple_header(title, icon):
    st.title(f"{icon} {title}")
    st.markdown("---")
