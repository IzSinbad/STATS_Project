"""
Statistical analysis tools for music streaming data.

This module provides functions for calculating various statistical measures
and performing analyses on music streaming and listener preference data.
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_basic_stats(data, columns=None):
    """
    Calculate basic descriptive statistics for specified columns.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The dataframe to analyze.
    columns : list, optional
        List of column names to analyze. If None, uses all numeric columns.
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing statistics (mean, median, std, min, max, etc.)
        for each specified column.
    """
    if columns is None:
        # Use all numeric columns
        columns = data.select_dtypes(include=['number']).columns.tolist()
    
    # Calculate statistics
    stats_df = pd.DataFrame({
        'mean': data[columns].mean(),
        'median': data[columns].median(),
        'std': data[columns].std(),
        'min': data[columns].min(),
        'max': data[columns].max(),
        '25%': data[columns].quantile(0.25),
        '75%': data[columns].quantile(0.75)
    })
    
    return stats_df

def calculate_correlation_matrix(data, columns=None, method='pearson'):
    """
    Calculate correlation matrix for specified columns.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The dataframe to analyze.
    columns : list, optional
        List of column names to include. If None, uses all numeric columns.
    method : str, optional
        Correlation method ('pearson', 'spearman', or 'kendall').
        
    Returns:
    --------
    pandas.DataFrame
        Correlation matrix.
    """
    if columns is None:
        # Use all numeric columns
        columns = data.select_dtypes(include=['number']).columns.tolist()
    
    # Calculate correlation matrix
    corr_matrix = data[columns].corr(method=method)
    
    return corr_matrix

def plot_correlation_heatmap(corr_matrix, figsize=(12, 10), cmap='coolwarm', 
                             save_path=None):
    """
    Plot a correlation matrix as a heatmap.
    
    Parameters:
    -----------
    corr_matrix : pandas.DataFrame
        Correlation matrix to plot.
    figsize : tuple, optional
        Figure size (width, height) in inches.
    cmap : str, optional
        Colormap for the heatmap.
    save_path : str, optional
        Path to save the figure. If None, the figure is not saved.
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object.
    """
    plt.figure(figsize=figsize)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # Plot heatmap
    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                annot=True, fmt=".2f", square=True, linewidths=.5)
    
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()

def calculate_mode(data, column):
    """
    Calculate the mode (most common value) for a column.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The dataframe to analyze.
    column : str
        Column name to find mode for.
        
    Returns:
    --------
    mode_value
        The most common value in the column.
    count : int
        Number of occurrences of the mode.
    """
    mode_result = data[column].mode()
    if len(mode_result) > 0:
        mode_value = mode_result[0]
        count = data[column].value_counts()[mode_value]
        return mode_value, count
    else:
        return None, 0

def calculate_group_statistics(data, group_by, value_col):
    """
    Calculate statistics for a value column grouped by another column.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The dataframe to analyze.
    group_by : str
        Column name to group by.
    value_col : str
        Column name to calculate statistics for.
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with statistics for each group.
    """
    grouped_stats = data.groupby(group_by)[value_col].agg([
        'count', 'mean', 'median', 'std', 'min', 'max'
    ]).sort_values('count', ascending=False)
    
    return grouped_stats

def perform_ttest(data, column, group_col, group1, group2):
    """
    Perform a t-test comparing two groups.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The dataframe containing the data.
    column : str
        Column name with the values to compare.
    group_col : str
        Column name containing group identifiers.
    group1 : str
        Name of first group to compare.
    group2 : str
        Name of second group to compare.
        
    Returns:
    --------
    t_stat : float
        T-statistic.
    p_value : float
        P-value.
    mean1 : float
        Mean of group1.
    mean2 : float
        Mean of group2.
    """
    # Extract data for each group
    group1_data = data[data[group_col] == group1][column].dropna()
    group2_data = data[data[group_col] == group2][column].dropna()
    
    # Perform t-test
    t_stat, p_value = stats.ttest_ind(group1_data, group2_data, equal_var=False)
    
    # Calculate means
    mean1 = group1_data.mean()
    mean2 = group2_data.mean()
    
    return t_stat, p_value, mean1, mean2

def calculate_percentiles(data, column, percentiles=None):
    """
    Calculate percentiles for a column.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The dataframe to analyze.
    column : str
        Column name to calculate percentiles for.
    percentiles : list, optional
        List of percentiles to calculate. If None, uses [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99].
        
    Returns:
    --------
    pandas.Series
        Series with percentile values.
    """
    if percentiles is None:
        percentiles = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    
    return data[column].quantile(percentiles)

def detect_outliers(data, column, method='iqr', threshold=1.5):
    """
    Detect outliers in a column.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The dataframe to analyze.
    column : str
        Column name to detect outliers in.
    method : str, optional
        Method to use for outlier detection ('iqr' or 'zscore').
    threshold : float, optional
        Threshold for outlier detection (1.5 for IQR, 3 for z-score).
        
    Returns:
    --------
    pandas.Series
        Boolean series indicating outliers.
    """
    if method == 'iqr':
        # IQR method
        q1 = data[column].quantile(0.25)
        q3 = data[column].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        return (data[column] < lower_bound) | (data[column] > upper_bound)
    
    elif method == 'zscore':
        # Z-score method
        z_scores = np.abs(stats.zscore(data[column].dropna()))
        return pd.Series(z_scores > threshold, index=data[column].dropna().index)
    
    else:
        raise ValueError("Method must be 'iqr' or 'zscore'")

def calculate_skewness_kurtosis(data, columns=None):
    """
    Calculate skewness and kurtosis for specified columns.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The dataframe to analyze.
    columns : list, optional
        List of column names to analyze. If None, uses all numeric columns.
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with skewness and kurtosis for each column.
    """
    if columns is None:
        # Use all numeric columns
        columns = data.select_dtypes(include=['number']).columns.tolist()
    
    # Calculate skewness and kurtosis
    skew_kurt = pd.DataFrame({
        'skewness': data[columns].skew(),
        'kurtosis': data[columns].kurtosis()
    })
    
    return skew_kurt
