"""
Data loading and preprocessing utilities for music streaming analysis.

This module provides functions to load, clean, and preprocess the Spotify streaming
and listener preferences datasets.
"""

import pandas as pd
import numpy as np
import os

def load_spotify_data(filepath=None):
    """
    Load the Spotify most streamed songs dataset.
    
    Parameters:
    -----------
    filepath : str, optional
        Path to the CSV file. If None, uses the default path.
        
    Returns:
    --------
    pandas.DataFrame
        The loaded and minimally processed dataset.
    """
    if filepath is None:
        # Default path relative to project root
        filepath = os.path.join('data', 'most_streamed_spotify_2024.csv')
    
    # Load the data
    df = pd.read_csv(filepath)
    
    # Basic cleaning
    # Remove any duplicate rows
    df = df.drop_duplicates()
    
    # Convert date columns to datetime if present
    if 'Release Date' in df.columns:
        df['Release Date'] = pd.to_datetime(df['Release Date'], errors='coerce')
    
    return df

def load_listener_preferences(filepath=None):
    """
    Load the global listener preferences dataset.
    
    Parameters:
    -----------
    filepath : str, optional
        Path to the CSV file. If None, uses the default path.
        
    Returns:
    --------
    pandas.DataFrame
        The loaded and minimally processed dataset.
    """
    if filepath is None:
        # Default path relative to project root
        filepath = os.path.join('data', 'global_listener_preferences.csv')
    
    # Load the data
    df = pd.read_csv(filepath)
    
    # Basic cleaning
    # Remove any duplicate rows
    df = df.drop_duplicates()
    
    # Convert numeric columns that might be loaded as strings
    numeric_cols = ['Age', 'Minutes Streamed Per Day', 'Number of Songs Liked', 
                   'Discover Weekly Engagement (%)', 'Repeat Song Rate (%)']
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

def clean_spotify_data(df):
    """
    Perform more thorough cleaning on the Spotify dataset.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe to clean.
        
    Returns:
    --------
    pandas.DataFrame
        The cleaned dataframe.
    """
    # Make a copy to avoid modifying the original
    df_clean = df.copy()
    
    # Handle missing values
    # For numeric columns, fill with median or 0 depending on context
    numeric_cols = df_clean.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        # Fill missing values with median
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    
    # For categorical columns, fill with 'Unknown'
    categorical_cols = df_clean.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df_clean[col] = df_clean[col].fillna('Unknown')
    
    # Remove any rows that still have NaN values
    df_clean = df_clean.dropna()
    
    return df_clean

def clean_listener_preferences(df):
    """
    Perform more thorough cleaning on the listener preferences dataset.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe to clean.
        
    Returns:
    --------
    pandas.DataFrame
        The cleaned dataframe.
    """
    # Make a copy to avoid modifying the original
    df_clean = df.copy()
    
    # Handle missing values
    # For numeric columns, fill with median
    numeric_cols = df_clean.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    
    # For categorical columns, fill with 'Unknown'
    categorical_cols = df_clean.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df_clean[col] = df_clean[col].fillna('Unknown')
    
    # Remove any rows that still have NaN values
    df_clean = df_clean.dropna()
    
    return df_clean

def merge_datasets(spotify_df, listener_df, on=None):
    """
    Merge the Spotify and listener preferences datasets if possible.
    
    Parameters:
    -----------
    spotify_df : pandas.DataFrame
        The Spotify streaming data.
    listener_df : pandas.DataFrame
        The listener preferences data.
    on : str or list, optional
        Column(s) to join on. If None, attempts to find common columns.
        
    Returns:
    --------
    pandas.DataFrame
        The merged dataframe, or None if no common columns found.
    """
    # This is a placeholder - actual implementation would depend on
    # how the datasets can be meaningfully joined
    
    # If no join column specified, try to find common columns
    if on is None:
        common_cols = set(spotify_df.columns) & set(listener_df.columns)
        if common_cols:
            on = list(common_cols)[0]  # Use the first common column
        else:
            print("No common columns found for merging datasets")
            return None
    
    # Perform the merge
    merged_df = pd.merge(spotify_df, listener_df, on=on, how='inner')
    
    return merged_df

def get_feature_matrix(df, features=None, target=None):
    """
    Extract a feature matrix and target variable for modeling.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe to extract features from.
    features : list, optional
        List of column names to use as features. If None, uses all numeric columns.
    target : str, optional
        Column name to use as target variable. If None, returns only X.
        
    Returns:
    --------
    X : pandas.DataFrame
        Feature matrix.
    y : pandas.Series, optional
        Target variable, only if target is specified.
    """
    # If no features specified, use all numeric columns
    if features is None:
        features = df.select_dtypes(include=['number']).columns.tolist()
        # Remove the target from features if it's there
        if target in features:
            features.remove(target)
    
    # Extract feature matrix
    X = df[features]
    
    # Extract target if specified
    if target is not None:
        y = df[target]
        return X, y
    else:
        return X
