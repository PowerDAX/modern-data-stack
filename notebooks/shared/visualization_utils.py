"""
Shared Visualization Utilities Module

This module provides common visualization functions and utilities for notebooks.
Includes statistical plots, interactive visualizations, and custom styling.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import warnings
from typing import Dict, List, Optional, Union, Any, Tuple
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set default style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Color palettes
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'accent': '#F18F01',
    'success': '#C73E1D',
    'warning': '#F4A261',
    'info': '#264653',
    'light': '#E9C46A',
    'dark': '#2A2A2A'
}

CATEGORICAL_COLORS = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#F4A261', '#264653', '#E9C46A']


class PlotStyler:
    """Utility class for consistent plot styling"""
    
    @staticmethod
    def apply_style(fig, title: str = None, theme: str = 'default'):
        """Apply consistent styling to matplotlib figure"""
        if title:
            fig.suptitle(title, fontsize=16, fontweight='bold')
        
        for ax in fig.get_axes():
            ax.grid(True, alpha=0.3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#CCCCCC')
            ax.spines['bottom'].set_color('#CCCCCC')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def apply_plotly_style(fig, title: str = None, theme: str = 'default'):
        """Apply consistent styling to plotly figure"""
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Arial, sans-serif", size=12, color="#2A2A2A"),
            title=dict(text=title, x=0.5, font=dict(size=16, color="#2A2A2A")),
            xaxis=dict(showgrid=True, gridcolor='#EEEEEE', linecolor='#CCCCCC'),
            yaxis=dict(showgrid=True, gridcolor='#EEEEEE', linecolor='#CCCCCC'),
            legend=dict(bgcolor='rgba(255,255,255,0.8)', bordercolor='#CCCCCC', borderwidth=1)
        )
        return fig


def plot_distribution(data: pd.Series, 
                     title: str = None, 
                     bins: int = 30,
                     kde: bool = True,
                     interactive: bool = False) -> Union[plt.Figure, go.Figure]:
    """
    Plot distribution of a numerical variable
    
    Args:
        data: Pandas Series with numerical data
        title: Plot title
        bins: Number of histogram bins
        kde: Whether to include KDE curve
        interactive: Whether to create interactive plot
    
    Returns:
        Matplotlib or Plotly figure
    """
    if title is None:
        title = f'Distribution of {data.name}'
    
    if interactive:
        fig = px.histogram(
            x=data, 
            nbins=bins, 
            title=title,
            marginal='box',
            hover_data=[data.name] if data.name else None
        )
        return PlotStyler.apply_plotly_style(fig, title)
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Histogram
        ax.hist(data.dropna(), bins=bins, alpha=0.7, color=COLORS['primary'], edgecolor='black')
        
        # KDE curve
        if kde:
            from scipy import stats
            x_range = np.linspace(data.min(), data.max(), 100)
            kde_values = stats.gaussian_kde(data.dropna())(x_range)
            ax2 = ax.twinx()
            ax2.plot(x_range, kde_values, color=COLORS['secondary'], linewidth=2)
            ax2.set_ylabel('Density', color=COLORS['secondary'])
            ax2.tick_params(axis='y', labelcolor=COLORS['secondary'])
        
        # Add statistics
        mean_val = data.mean()
        median_val = data.median()
        ax.axvline(mean_val, color=COLORS['accent'], linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
        ax.axvline(median_val, color=COLORS['success'], linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}')
        
        ax.set_xlabel(data.name or 'Value')
        ax.set_ylabel('Frequency')
        ax.legend()
        
        return PlotStyler.apply_style(fig, title)


def plot_correlation_matrix(df: pd.DataFrame, 
                          method: str = 'pearson',
                          interactive: bool = False,
                          title: str = None) -> Union[plt.Figure, go.Figure]:
    """
    Plot correlation matrix heatmap
    
    Args:
        df: DataFrame with numerical columns
        method: Correlation method ('pearson', 'spearman', 'kendall')
        interactive: Whether to create interactive plot
        title: Plot title
    
    Returns:
        Matplotlib or Plotly figure
    """
    if title is None:
        title = f'{method.capitalize()} Correlation Matrix'
    
    # Calculate correlation matrix
    corr_matrix = df.corr(method=method)
    
    if interactive:
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect='auto',
            color_continuous_scale='RdBu',
            title=title,
            zmin=-1,
            zmax=1
        )
        return PlotStyler.apply_plotly_style(fig, title)
    else:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create heatmap
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=True,
            cmap='RdBu',
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": .8},
            ax=ax
        )
        
        return PlotStyler.apply_style(fig, title)


def plot_categorical_analysis(data: pd.Series,
                            max_categories: int = 20,
                            interactive: bool = False,
                            title: str = None) -> Union[plt.Figure, go.Figure]:
    """
    Plot categorical variable analysis
    
    Args:
        data: Pandas Series with categorical data
        max_categories: Maximum number of categories to show
        interactive: Whether to create interactive plot
        title: Plot title
    
    Returns:
        Matplotlib or Plotly figure
    """
    if title is None:
        title = f'Distribution of {data.name}'
    
    # Get value counts
    value_counts = data.value_counts().head(max_categories)
    
    if interactive:
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Count Plot', 'Pie Chart'),
            specs=[[{"secondary_y": False}, {"type": "domain"}]]
        )
        
        # Bar chart
        fig.add_trace(
            go.Bar(x=value_counts.index, y=value_counts.values, name='Count'),
            row=1, col=1
        )
        
        # Pie chart
        fig.add_trace(
            go.Pie(labels=value_counts.index, values=value_counts.values, name='Distribution'),
            row=1, col=2
        )
        
        return PlotStyler.apply_plotly_style(fig, title)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Bar plot
        value_counts.plot(kind='bar', ax=axes[0], color=CATEGORICAL_COLORS[:len(value_counts)])
        axes[0].set_title('Count Plot')
        axes[0].set_xlabel(data.name or 'Category')
        axes[0].set_ylabel('Count')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Pie chart
        axes[1].pie(value_counts.values, labels=value_counts.index, autopct='%1.1f%%', colors=CATEGORICAL_COLORS[:len(value_counts)])
        axes[1].set_title('Distribution')
        
        return PlotStyler.apply_style(fig, title)


def plot_time_series(df: pd.DataFrame,
                    date_column: str,
                    value_column: str,
                    interactive: bool = False,
                    title: str = None) -> Union[plt.Figure, go.Figure]:
    """
    Plot time series data
    
    Args:
        df: DataFrame with time series data
        date_column: Name of the date column
        value_column: Name of the value column
        interactive: Whether to create interactive plot
        title: Plot title
    
    Returns:
        Matplotlib or Plotly figure
    """
    if title is None:
        title = f'{value_column} Over Time'
    
    # Ensure date column is datetime
    df_copy = df.copy()
    df_copy[date_column] = pd.to_datetime(df_copy[date_column])
    df_copy = df_copy.sort_values(date_column)
    
    if interactive:
        fig = px.line(
            df_copy, 
            x=date_column, 
            y=value_column,
            title=title,
            markers=True
        )
        return PlotStyler.apply_plotly_style(fig, title)
    else:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(df_copy[date_column], df_copy[value_column], color=COLORS['primary'], linewidth=2, marker='o', markersize=4)
        ax.set_xlabel('Date')
        ax.set_ylabel(value_column)
        
        # Add trend line
        from scipy import stats
        x_numeric = np.arange(len(df_copy))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_numeric, df_copy[value_column])
        trend_line = slope * x_numeric + intercept
        ax.plot(df_copy[date_column], trend_line, color=COLORS['accent'], linestyle='--', linewidth=2, label=f'Trend (R²={r_value**2:.3f})')
        
        ax.legend()
        return PlotStyler.apply_style(fig, title)


def plot_scatter_with_regression(df: pd.DataFrame,
                                x_column: str,
                                y_column: str,
                                color_column: str = None,
                                size_column: str = None,
                                interactive: bool = False,
                                title: str = None) -> Union[plt.Figure, go.Figure]:
    """
    Plot scatter plot with regression line
    
    Args:
        df: DataFrame with data
        x_column: Name of x-axis column
        y_column: Name of y-axis column
        color_column: Name of color grouping column
        size_column: Name of size column
        interactive: Whether to create interactive plot
        title: Plot title
    
    Returns:
        Matplotlib or Plotly figure
    """
    if title is None:
        title = f'{y_column} vs {x_column}'
    
    if interactive:
        fig = px.scatter(
            df, 
            x=x_column, 
            y=y_column,
            color=color_column,
            size=size_column,
            title=title,
            trendline='ols'
        )
        return PlotStyler.apply_plotly_style(fig, title)
    else:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if color_column:
            categories = df[color_column].unique()
            for i, cat in enumerate(categories):
                subset = df[df[color_column] == cat]
                ax.scatter(subset[x_column], subset[y_column], 
                          color=CATEGORICAL_COLORS[i % len(CATEGORICAL_COLORS)], 
                          label=cat, alpha=0.7)
        else:
            ax.scatter(df[x_column], df[y_column], color=COLORS['primary'], alpha=0.7)
        
        # Add regression line
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(df[x_column], df[y_column])
        line = slope * df[x_column] + intercept
        ax.plot(df[x_column], line, color=COLORS['accent'], linewidth=2, label=f'Regression (R²={r_value**2:.3f})')
        
        ax.set_xlabel(x_column)
        ax.set_ylabel(y_column)
        ax.legend()
        
        return PlotStyler.apply_style(fig, title)


def plot_missing_values(df: pd.DataFrame,
                       interactive: bool = False,
                       title: str = None) -> Union[plt.Figure, go.Figure]:
    """
    Plot missing values analysis
    
    Args:
        df: DataFrame to analyze
        interactive: Whether to create interactive plot
        title: Plot title
    
    Returns:
        Matplotlib or Plotly figure
    """
    if title is None:
        title = 'Missing Values Analysis'
    
    # Calculate missing values
    missing_data = df.isnull().sum()
    missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
    
    if len(missing_data) == 0:
        print("No missing values found in the dataset")
        return None
    
    missing_percentage = (missing_data / len(df)) * 100
    
    if interactive:
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Missing Values Count', 'Missing Values Percentage'),
            vertical_spacing=0.1
        )
        
        # Count plot
        fig.add_trace(
            go.Bar(x=missing_data.index, y=missing_data.values, name='Count'),
            row=1, col=1
        )
        
        # Percentage plot
        fig.add_trace(
            go.Bar(x=missing_percentage.index, y=missing_percentage.values, name='Percentage'),
            row=2, col=1
        )
        
        return PlotStyler.apply_plotly_style(fig, title)
    else:
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # Count plot
        missing_data.plot(kind='bar', ax=axes[0], color=COLORS['primary'])
        axes[0].set_title('Missing Values Count')
        axes[0].set_ylabel('Count')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Percentage plot
        missing_percentage.plot(kind='bar', ax=axes[1], color=COLORS['secondary'])
        axes[1].set_title('Missing Values Percentage')
        axes[1].set_ylabel('Percentage (%)')
        axes[1].tick_params(axis='x', rotation=45)
        
        return PlotStyler.apply_style(fig, title)


def plot_outliers_analysis(df: pd.DataFrame,
                          numerical_columns: List[str] = None,
                          interactive: bool = False,
                          title: str = None) -> Union[plt.Figure, go.Figure]:
    """
    Plot outliers analysis using box plots
    
    Args:
        df: DataFrame to analyze
        numerical_columns: List of numerical columns to analyze
        interactive: Whether to create interactive plot
        title: Plot title
    
    Returns:
        Matplotlib or Plotly figure
    """
    if title is None:
        title = 'Outliers Analysis'
    
    if numerical_columns is None:
        numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if interactive:
        fig = go.Figure()
        
        for col in numerical_columns:
            fig.add_trace(go.Box(y=df[col], name=col))
        
        fig.update_layout(title=title, showlegend=False)
        return PlotStyler.apply_plotly_style(fig, title)
    else:
        n_cols = min(3, len(numerical_columns))
        n_rows = (len(numerical_columns) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes
        
        for i, col in enumerate(numerical_columns):
            if i < len(axes):
                df[col].plot(kind='box', ax=axes[i])
                axes[i].set_title(f'Box Plot - {col}')
                axes[i].set_ylabel(col)
        
        # Hide unused subplots
        for i in range(len(numerical_columns), len(axes)):
            axes[i].set_visible(False)
        
        return PlotStyler.apply_style(fig, title)


def create_dashboard(df: pd.DataFrame,
                    target_column: str = None,
                    max_categorical_categories: int = 10) -> go.Figure:
    """
    Create a comprehensive dashboard for data exploration
    
    Args:
        df: DataFrame to analyze
        target_column: Name of target column for supervised learning
        max_categorical_categories: Maximum categories to show for categorical variables
    
    Returns:
        Plotly figure with dashboard
    """
    # Identify column types
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[
            'Dataset Overview', 'Missing Values',
            'Numerical Distributions', 'Categorical Distributions',
            'Correlation Matrix', 'Target Analysis' if target_column else 'Summary Statistics'
        ],
        specs=[
            [{"type": "table"}, {"type": "bar"}],
            [{"colspan": 2}, None],
            [{"type": "heatmap"}, {"type": "bar"}]
        ]
    )
    
    # Dataset overview
    overview_data = [
        ['Shape', f'{df.shape[0]} rows × {df.shape[1]} columns'],
        ['Memory Usage', f'{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB'],
        ['Missing Values', f'{df.isnull().sum().sum()} ({df.isnull().sum().sum()/df.size*100:.1f}%)'],
        ['Duplicates', f'{df.duplicated().sum()} ({df.duplicated().sum()/len(df)*100:.1f}%)'],
        ['Numerical Columns', str(len(numerical_cols))],
        ['Categorical Columns', str(len(categorical_cols))]
    ]
    
    fig.add_trace(
        go.Table(
            header=dict(values=['Metric', 'Value'], fill_color='lightblue'),
            cells=dict(values=list(zip(*overview_data)), fill_color='white')
        ),
        row=1, col=1
    )
    
    # Missing values
    missing_data = df.isnull().sum()
    missing_data = missing_data[missing_data > 0]
    if len(missing_data) > 0:
        fig.add_trace(
            go.Bar(x=missing_data.index, y=missing_data.values, name='Missing Values'),
            row=1, col=2
        )
    
    # Numerical distributions (sample)
    if numerical_cols:
        sample_col = numerical_cols[0]
        fig.add_trace(
            go.Histogram(x=df[sample_col], name=f'{sample_col} Distribution'),
            row=2, col=1
        )
    
    # Categorical distributions (sample)
    if categorical_cols:
        sample_col = categorical_cols[0]
        value_counts = df[sample_col].value_counts().head(max_categorical_categories)
        fig.add_trace(
            go.Bar(x=value_counts.index, y=value_counts.values, name=f'{sample_col} Distribution'),
            row=2, col=2
        )
    
    # Correlation matrix
    if len(numerical_cols) > 1:
        corr_matrix = df[numerical_cols].corr()
        fig.add_trace(
            go.Heatmap(z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.index, colorscale='RdBu'),
            row=3, col=1
        )
    
    # Target analysis
    if target_column and target_column in df.columns:
        if df[target_column].dtype in ['object', 'category']:
            target_counts = df[target_column].value_counts()
            fig.add_trace(
                go.Bar(x=target_counts.index, y=target_counts.values, name='Target Distribution'),
                row=3, col=2
            )
        else:
            fig.add_trace(
                go.Histogram(x=df[target_column], name='Target Distribution'),
                row=3, col=2
            )
    
    fig.update_layout(
        height=1200,
        showlegend=False,
        title_text="Data Exploration Dashboard",
        title_x=0.5
    )
    
    return PlotStyler.apply_plotly_style(fig, "Data Exploration Dashboard")


# Utility functions
def save_plot(fig: Union[plt.Figure, go.Figure], 
              filename: str, 
              format: str = 'png',
              width: int = 800,
              height: int = 600,
              dpi: int = 300):
    """
    Save plot to file
    
    Args:
        fig: Figure to save
        filename: Output filename
        format: Output format ('png', 'jpg', 'pdf', 'svg', 'html')
        width: Width in pixels (for plotly)
        height: Height in pixels (for plotly)
        dpi: DPI for matplotlib
    """
    if isinstance(fig, go.Figure):
        if format == 'html':
            fig.write_html(filename)
        else:
            fig.write_image(filename, width=width, height=height, format=format)
    else:
        fig.savefig(filename, dpi=dpi, bbox_inches='tight', format=format)
    
    logger.info(f"Plot saved as {filename}")


def get_color_palette(n_colors: int, palette_type: str = 'default') -> List[str]:
    """
    Get color palette for visualizations
    
    Args:
        n_colors: Number of colors needed
        palette_type: Type of palette ('default', 'pastel', 'dark', 'bright')
    
    Returns:
        List of color hex codes
    """
    if palette_type == 'default':
        colors = CATEGORICAL_COLORS
    elif palette_type == 'pastel':
        colors = ['#FFB3BA', '#FFDFBA', '#FFFFBA', '#BAFFC9', '#BAE1FF', '#E1BAFF', '#FFBAE1']
    elif palette_type == 'dark':
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    elif palette_type == 'bright':
        colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF', '#FFA500']
    else:
        colors = CATEGORICAL_COLORS
    
    # Extend palette if needed
    while len(colors) < n_colors:
        colors.extend(colors)
    
    return colors[:n_colors]


# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    df = pd.DataFrame({
        'numerical1': np.random.normal(0, 1, 1000),
        'numerical2': np.random.normal(5, 2, 1000),
        'categorical1': np.random.choice(['A', 'B', 'C'], 1000),
        'categorical2': np.random.choice(['X', 'Y'], 1000),
        'target': np.random.choice([0, 1], 1000)
    })
    
    # Add some missing values
    df.loc[np.random.choice(df.index, 50, replace=False), 'numerical1'] = np.nan
    df.loc[np.random.choice(df.index, 30, replace=False), 'categorical1'] = np.nan
    
    print("Testing visualization utilities...")
    
    # Test distribution plot
    fig = plot_distribution(df['numerical1'], interactive=False)
    save_plot(fig, 'distribution_test.png')
    
    # Test correlation matrix
    fig = plot_correlation_matrix(df[['numerical1', 'numerical2']], interactive=False)
    save_plot(fig, 'correlation_test.png')
    
    # Test dashboard
    fig = create_dashboard(df, target_column='target')
    save_plot(fig, 'dashboard_test.html', format='html')
    
    print("Visualization tests completed!") 