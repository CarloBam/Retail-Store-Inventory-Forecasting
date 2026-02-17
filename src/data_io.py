import pandas as pd
import streamlit as st
import os

DEFAULT_DATA_PATH = "retail_store_inventory.csv" # Assumes file is in CWD or handle search
LOCAL_DATA_PATH = "data/retail_store_inventory.csv"  # Fallback or local copy

@st.cache_data
def load_data(file_source):
    """
    Loads data from a CSV file source (path or buffer).
    """
    try:
        df = pd.read_csv(file_source)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def initialize_session_state():
    """
    Initializes session state variables if they don't exist.
    """
    if 'raw_data' not in st.session_state:
        st.session_state['raw_data'] = None
    if 'cleaned_data' not in st.session_state:
        st.session_state['cleaned_data'] = None
    if 'feature_data' not in st.session_state:
        st.session_state['feature_data'] = None
    if 'config' not in st.session_state:
        st.session_state['config'] = {
            'random_seed': 42,
            'forecast_horizon': 30
        }

def save_uploaded_file(uploaded_file):
    """
    Saves uploaded file to local data directory (optional).
    """
    if uploaded_file is not None:
        with open(os.path.join("data", uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        return os.path.join("data", uploaded_file.name)
    return None
