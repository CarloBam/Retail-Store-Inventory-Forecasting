import pandas as pd
import numpy as np

def filter_data(df, date_range=None, categories=None, regions=None, stores=None):
    """
    Filters the dataframe based on provided criteria.
    """
    df_filtered = df.copy()
    
    # Date Range
    if date_range:
        start_date, end_date = date_range
        # Filter by date only when the Date column is available.
        if 'Date' in df_filtered.columns:
            mask = (df_filtered['Date'].dt.date >= start_date) & (df_filtered['Date'].dt.date <= end_date)
            df_filtered = df_filtered[mask]
            
    # Categories
    if categories:
        if 'Category' in df_filtered.columns:
            df_filtered = df_filtered[df_filtered['Category'].isin(categories)]
            
    # Regions
    if regions:
        if 'Region' in df_filtered.columns:
            df_filtered = df_filtered[df_filtered['Region'].isin(regions)]
            
    # Stores
    if stores:
        # Check for Store or Store ID
        store_col = 'Store ID' if 'Store ID' in df_filtered.columns else 'Store'
        if store_col in df_filtered.columns:
            df_filtered = df_filtered[df_filtered[store_col].isin(stores)]
            
    return df_filtered

def get_summary_stats(df):
    return df.describe()

def get_correlation_matrix(df):
    numeric_df = df.select_dtypes(include=[np.number])
    return numeric_df.corr()
