import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# --- Interactive Plotly Functions ---

def plot_sales_trends(df, date_col='Date', value_col='Units Sold', title='Sales Trends', log_scale=False):
    """
    Interactive Line plot of value over time.
    """
    daily = df.groupby(date_col)[value_col].sum().reset_index()
    
    fig = px.line(daily, x=date_col, y=value_col, title=title)
    fig.update_layout(template="plotly_dark", xaxis_title="Date", yaxis_title=value_col)
    
    if log_scale:
        fig.update_yaxes(type="log")
        
    return fig

def plot_top_categories(df, cat_col='Category', value_col='Revenue', n=10, title='Top Categories'):
    """
    Interactive Bar chart of top categories.
    """
    top_n = df.groupby(cat_col)[value_col].sum().sort_values(ascending=True).tail(n).reset_index()
    
    fig = px.bar(top_n, x=value_col, y=cat_col, orientation='h', title=title, color=value_col)
    fig.update_layout(template="plotly_dark", yaxis={'categoryorder':'total ascending'})
    return fig

def plot_heatmap_pivot(df, index_col, columns_col, value_col, title='Heatmap'):
    """
    Interactive Heatmap.
    """
    pivot = df.pivot_table(index=index_col, columns=columns_col, values=value_col, aggfunc='mean')
    
    fig = px.imshow(pivot, labels=dict(x=columns_col, y=index_col, color=value_col), title=title)
    fig.update_layout(template="plotly_dark")
    return fig

def plot_distribution(df, col, by_col=None, title='Distribution'):
    """
    Interactive Boxplot or Hist.
    """
    if by_col:
        fig = px.box(df, x=by_col, y=col, title=title, color=by_col)
    else:
        fig = px.histogram(df, x=col, nbins=30, title=title, marginal="box")
        
    fig.update_layout(template="plotly_dark")
    return fig

def plot_promo_effect(df, promo_col, value_col, title='Promo Effect'):
    """
    Interactive Bar chart comparison.
    """
    avg_sales = df.groupby(promo_col, as_index=False)[value_col].mean()
    # Map boolean/int to string for better labels
    avg_sales['Label'] = avg_sales[promo_col].astype(str).replace({'0': 'No Promo', '1': 'Promo', 'False': 'No Promo', 'True': 'Promo'})
    
    fig = px.bar(avg_sales, x='Label', y=value_col, color='Label', title=title)
    fig.update_layout(template="plotly_dark")
    return fig

def plot_scatter_fit(df, x_col, y_col, title='Scatter Plot'):
    """
    Interactive Scatter with Trendline (OLS).
    """
    # Downsample if too large for speed
    plot_df = df.sample(min(len(df), 2000)) if len(df) > 2000 else df
    
    fig = px.scatter(plot_df, x=x_col, y=y_col, title=title, trendline="ols", opacity=0.6)
    fig.update_layout(template="plotly_dark")
    return fig
    
# --- Static Matplotlib Functions (Renamed/Preserved for Reporting) ---
import matplotlib.pyplot as plt
import seaborn as sns

def plot_sales_trends_static(df, date_col='Date', value_col='Units Sold', title='Sales Trends', log_scale=False):
    daily = df.groupby(date_col)[value_col].sum().reset_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(daily[date_col], daily[value_col], linewidth=2, color='#4c72b0')
    if log_scale: ax.set_yscale('log')
    ax.set_title(title); ax.set_xlabel("Date"); ax.set_ylabel(value_col)
    plt.xticks(rotation=45); plt.tight_layout()
    return fig

def plot_top_categories_static(df, cat_col='Category', value_col='Revenue', n=10, title='Top Categories'):
    top_n = df.groupby(cat_col)[value_col].sum().sort_values(ascending=True).tail(n)
    fig, ax = plt.subplots(figsize=(10, 6))
    top_n.plot(kind='barh', ax=ax, color='#55a868')
    ax.set_title(title); ax.set_xlabel(value_col); plt.tight_layout()
    return fig

# (Other static functions can be kept if needed, but these are the main ones used in PPTX)
