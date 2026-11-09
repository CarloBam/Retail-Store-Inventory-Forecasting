import pandas as pd
import numpy as np
import streamlit as st

def parse_and_load_data(df):
    """
    Parse the date and known numeric columns, coercing invalid values.
    """
    # 1. Date Parsing
    # Parse the expected Date column when present.
    if 'Date' in df.columns:
        try:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        except Exception as e:
            st.warning(f"Could not parse 'Date' column: {e}")
    
    # 2. Numeric Coercion
    # Coerce known numeric fields when present.
    numeric_candidates = ['Sales', 'Price', 'Inventory', 'Demand']
    for col in numeric_candidates:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    return df

def get_data_dictionary(df):
    """
    Returns a DataFrame describing the columns.
    """
    # Infer meanings based on content
    data_dict = []
    
    for col in df.columns:
        dtype = str(df[col].dtype)
        example = df[col].dropna().iloc[0] if not df[col].dropna().empty else "N/A"
        
        meaning = "Unknown"
        col_lower = col.lower()
        if "date" in col_lower:
            meaning = "Transaction Date"
        elif "id" in col_lower or "sku" in col_lower:
            meaning = "Unique Identifier"
        elif "price" in col_lower:
            meaning = "Unit Price of Product"
        elif "sales" in col_lower or "demand" in col_lower:
            meaning = "Quantity Sold / Demand"
        elif "inventory" in col_lower or "stock" in col_lower:
            meaning = "Available Stock Level"
        elif "category" in col_lower:
            meaning = "Product Category"
        elif "product" in col_lower:
            meaning = "Product Name/ID"
            
        data_dict.append({
            "Column Name": col,
            "Type": dtype,
            "Example": str(example),
            "Meaning": meaning
        })
        
    return pd.DataFrame(data_dict)

def get_dataset_summary(df):
    """
    Returns summary metrics.
    """
    summary = {
        "Rows": df.shape[0],
        "Columns": df.shape[1],
        "Missing Values": df.isnull().sum().sum(),
        "Missing %": (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
    }
    
    if 'Date' in df.columns and pd.api.types.is_datetime64_any_dtype(df['Date']):
        summary["Date Range"] = f"{df['Date'].min().date()} to {df['Date'].max().date()}"
    else:
        summary["Date Range"] = "N/A"
        
    if 'Store_ID' in df.columns:
        summary["Stores"] = df['Store_ID'].nunique()
    elif 'Store' in df.columns:
        summary["Stores"] = df['Store'].nunique()
    else:
        # Fallback if no specific store column
        summary["Stores"] = "N/A"

    if 'Product_ID' in df.columns:
        summary["Products"] = df['Product_ID'].nunique()
    elif 'Product' in df.columns:
        summary["Products"] = df['Product'].nunique()
    else:
         summary["Products"] = "N/A"

    if 'Category' in df.columns:
        summary["Categories"] = df['Category'].nunique()
    else:
        summary["Categories"] = "N/A"
        
    return summary
