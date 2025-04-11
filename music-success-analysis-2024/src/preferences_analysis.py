"""
Analysis utilities for music streaming listener preferences.

This module provides functions for analyzing listener preferences data,
including platform usage, genre preferences, and listening habits.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

def analyze_platform_distribution(data, platform_col='Streaming Platform'):
    """
    Analyze the distribution of streaming platforms.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The dataframe containing the listener preferences data.
    platform_col : str, optional
        Column name for streaming platform.
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with platform distribution statistics.
    """
    # Count platforms
    platform_counts = data[platform_col].value_counts()
    
    # Calculate percentages
    platform_pcts = platform_counts / len(data) * 100
    
    # Combine into a DataFrame
    platform_stats = pd.DataFrame({
        'count': platform_counts,
        'percentage': platform_pcts
    })
    
    return platform_stats

def analyze_genre_preferences(data, genre_col='Top Genre', 
                             group_by=None, normalize=True):
    """
    Analyze genre preferences, optionally grouped by another variable.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The dataframe containing the listener preferences data.
    genre_col : str, optional
        Column name for genre.
    group_by : str, optional
        Column name to group by (e.g., 'Age', 'Country').
    normalize : bool, optional
        Whether to normalize counts within groups.
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with genre preference statistics.
    """
    if group_by is None:
        # Simple genre counts
        genre_stats = pd.DataFrame(data[genre_col].value_counts())
        genre_stats.columns = ['count']
        genre_stats['percentage'] = genre_stats['count'] / len(data) * 100
        return genre_stats
    else:
        # Cross-tabulation with grouping variable
        genre_cross = pd.crosstab(
            data[group_by], 
            data[genre_col], 
            normalize='index' if normalize else False
        )
        
        if normalize:
            genre_cross = genre_cross * 100  # Convert to percentages
        
        return genre_cross

def analyze_listening_time(data, time_col='Listening Time (Morning/Afternoon/Night)', 
                          group_by=None):
    """
    Analyze listening time preferences.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The dataframe containing the listener preferences data.
    time_col : str, optional
        Column name for listening time.
    group_by : str, optional
        Column name to group by (e.g., 'Age', 'Country').
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with listening time statistics.
    """
    if group_by is None:
        # Simple time period counts
        time_stats = pd.DataFrame(data[time_col].value_counts())
        time_stats.columns = ['count']
        time_stats['percentage'] = time_stats['count'] / len(data) * 100
        return time_stats
    else:
        # Cross-tabulation with grouping variable
        time_cross = pd.crosstab(
            data[group_by], 
            data[time_col], 
            normalize='index'
        ) * 100  # Convert to percentages
        
        return time_cross

def analyze_streaming_minutes(data, minutes_col='Minutes Streamed Per Day', 
                             group_by=None, bins=None):
    """
    Analyze streaming minutes distribution.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The dataframe containing the listener preferences data.
    minutes_col : str, optional
        Column name for minutes streamed.
    group_by : str, optional
        Column name to group by (e.g., 'Age', 'Country').
    bins : list or int, optional
        Bins for categorizing minutes. If None, uses quartiles.
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with streaming minutes statistics.
    """
    # Create a copy to avoid modifying the original
    df = data.copy()
    
    # Create bins if specified
    if bins is not None:
        if isinstance(bins, int):
            # Create equal-width bins
            min_val = df[minutes_col].min()
            max_val = df[minutes_col].max()
            bins = np.linspace(min_val, max_val, bins + 1)
        
        # Create a categorical variable
        labels = [f'{bins[i]:.0f}-{bins[i+1]:.0f}' for i in range(len(bins)-1)]
        df['minutes_binned'] = pd.cut(df[minutes_col], bins=bins, labels=labels)
    else:
        # Use quartiles
        df['minutes_binned'] = pd.qcut(df[minutes_col], q=4, labels=['Low', 'Medium-Low', 'Medium-High', 'High'])
    
    if group_by is None:
        # Simple statistics
        minutes_stats = df.groupby('minutes_binned')[minutes_col].agg(['count', 'mean', 'std', 'min', 'max'])
        minutes_stats['percentage'] = minutes_stats['count'] / len(df) * 100
        return minutes_stats
    else:
        # Cross-tabulation with grouping variable
        minutes_cross = pd.crosstab(
            df[group_by], 
            df['minutes_binned'], 
            normalize='index'
        ) * 100  # Convert to percentages
        
        return minutes_cross

def analyze_subscription_types(data, subscription_col='Subscription Type', 
                              group_by=None):
    """
    Analyze subscription type distribution.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The dataframe containing the listener preferences data.
    subscription_col : str, optional
        Column name for subscription type.
    group_by : str, optional
        Column name to group by (e.g., 'Age', 'Country').
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with subscription type statistics.
    """
    if group_by is None:
        # Simple subscription type counts
        sub_stats = pd.DataFrame(data[subscription_col].value_counts())
        sub_stats.columns = ['count']
        sub_stats['percentage'] = sub_stats['count'] / len(data) * 100
        return sub_stats
    else:
        # Cross-tabulation with grouping variable
        sub_cross = pd.crosstab(
            data[group_by], 
            data[subscription_col], 
            normalize='index'
        ) * 100  # Convert to percentages
        
        return sub_cross

def analyze_engagement_metrics(data, metrics=None, group_by=None):
    """
    Analyze engagement metrics.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The dataframe containing the listener preferences data.
    metrics : list, optional
        List of engagement metric columns. If None, uses default metrics.
    group_by : str, optional
        Column name to group by (e.g., 'Age', 'Country').
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with engagement metrics statistics.
    """
    if metrics is None:
        # Default engagement metrics
        metrics = [
            'Minutes Streamed Per Day',
            'Number of Songs Liked',
            'Discover Weekly Engagement (%)',
            'Repeat Song Rate (%)'
        ]
    
    # Filter to include only metrics that exist in the data
    metrics = [m for m in metrics if m in data.columns]
    
    if group_by is None:
        # Calculate statistics for each metric
        engagement_stats = data[metrics].agg(['count', 'mean', 'median', 'std', 'min', 'max']).T
        return engagement_stats
    else:
        # Group by the specified column
        engagement_by_group = data.groupby(group_by)[metrics].agg(['mean', 'median', 'std'])
        return engagement_by_group

def analyze_artist_popularity(data, artist_col='Most Played Artist', top_n=10):
    """
    Analyze most popular artists.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The dataframe containing the listener preferences data.
    artist_col : str, optional
        Column name for artist.
    top_n : int, optional
        Number of top artists to include.
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with artist popularity statistics.
    """
    # Count artists
    artist_counts = data[artist_col].value_counts().head(top_n)
    
    # Calculate percentages
    artist_pcts = artist_counts / len(data) * 100
    
    # Combine into a DataFrame
    artist_stats = pd.DataFrame({
        'count': artist_counts,
        'percentage': artist_pcts
    })
    
    return artist_stats

def analyze_country_distribution(data, country_col='Country'):
    """
    Analyze listener country distribution.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The dataframe containing the listener preferences data.
    country_col : str, optional
        Column name for country.
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with country distribution statistics.
    """
    # Count countries
    country_counts = data[country_col].value_counts()
    
    # Calculate percentages
    country_pcts = country_counts / len(data) * 100
    
    # Combine into a DataFrame
    country_stats = pd.DataFrame({
        'count': country_counts,
        'percentage': country_pcts
    })
    
    return country_stats

def analyze_age_distribution(data, age_col='Age', bins=None):
    """
    Analyze listener age distribution.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The dataframe containing the listener preferences data.
    age_col : str, optional
        Column name for age.
    bins : list or int, optional
        Bins for categorizing age. If None, uses standard age groups.
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with age distribution statistics.
    """
    # Create a copy to avoid modifying the original
    df = data.copy()
    
    # Create bins if not specified
    if bins is None:
        # Standard age groups
        bins = [0, 18, 25, 35, 45, 55, 65, 100]
        labels = ['<18', '18-24', '25-34', '35-44', '45-54', '55-64', '65+']
    elif isinstance(bins, int):
        # Create equal-width bins
        min_age = df[age_col].min()
        max_age = df[age_col].max()
        bins = np.linspace(min_age, max_age, bins + 1)
        labels = [f'{bins[i]:.0f}-{bins[i+1]:.0f}' for i in range(len(bins)-1)]
    else:
        # Custom bins with custom labels
        labels = [f'{bins[i]}-{bins[i+1]}' for i in range(len(bins)-1)]
    
    # Create age groups
    df['age_group'] = pd.cut(df[age_col], bins=bins, labels=labels)
    
    # Count age groups
    age_counts = df['age_group'].value_counts().sort_index()
    
    # Calculate percentages
    age_pcts = age_counts / len(df) * 100
    
    # Combine into a DataFrame
    age_stats = pd.DataFrame({
        'count': age_counts,
        'percentage': age_pcts
    })
    
    return age_stats

def plot_preference_heatmap(data, row_var, col_var, figsize=(12, 8), 
                           cmap='viridis', save_path=None):
    """
    Create a heatmap of preferences.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The dataframe containing the listener preferences data.
    row_var : str
        Column name for row variable.
    col_var : str
        Column name for column variable.
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
    # Create cross-tabulation
    cross_tab = pd.crosstab(data[row_var], data[col_var], normalize='index') * 100
    
    # Create heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(cross_tab, annot=True, fmt='.1f', cmap=cmap)
    plt.title(f'{row_var} vs {col_var} Preferences')
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()

def segment_listeners(data, features=None, n_clusters=3, random_state=42):
    """
    Segment listeners using K-means clustering.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The dataframe containing the listener preferences data.
    features : list, optional
        List of features to use for clustering. If None, uses numeric columns.
    n_clusters : int, optional
        Number of clusters.
    random_state : int, optional
        Random state for reproducibility.
        
    Returns:
    --------
    pandas.DataFrame
        Original dataframe with cluster assignments.
    cluster_centers : numpy.ndarray
        Cluster centers.
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    
    # Create a copy to avoid modifying the original
    df = data.copy()
    
    # Select features
    if features is None:
        # Use numeric columns
        features = df.select_dtypes(include=['number']).columns.tolist()
    
    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(df[features])
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    df['cluster'] = kmeans.fit_predict(X)
    
    # Get cluster centers
    cluster_centers = pd.DataFrame(
        scaler.inverse_transform(kmeans.cluster_centers_),
        columns=features
    )
    
    return df, cluster_centers
