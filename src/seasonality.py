import plotly.express as px
import pandas as pd

def plot_seasonal_trends(df, date_col='Date', value_col='Units Sold', title='Year-over-Year Comparison'):
    """
    Plots interactive lines for each year on a common Month-Day axis.
    """
    df_plot = df.copy()
    if date_col not in df_plot.columns:
        return None
        
    # Ensure date
    if not pd.api.types.is_datetime64_any_dtype(df_plot[date_col]):
        df_plot[date_col] = pd.to_datetime(df_plot[date_col])

    df_plot['Year'] = df_plot[date_col].dt.year.astype(str)
    # Create a CommonDate for plotting (Year 2020)
    # Map dates to leap year 2020 so month-day positions align.
    df_plot['CommonDate'] = df_plot[date_col].apply(lambda x: x.replace(year=2020))
    
    # Aggregate to daily level to avoid huge scatter
    daily = df_plot.groupby(['Year', 'CommonDate'])[value_col].sum().reset_index()
    
    fig = px.line(daily, x='CommonDate', y=value_col, color='Year', title=title)
    
    fig.update_layout(
        template="plotly_dark",
        xaxis=dict(
            tickformat="%b",
            title="Month"
        )
    )
    return fig
