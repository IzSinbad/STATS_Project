"""
Visualization utilities for music streaming data analysis.

This module provides functions for creating various types of visualizations
to analyze and present insights from music streaming and listener preference data.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

# Set default style
sns.set(style="whitegrid")

def save_figure(fig, filename, directory='visuals', dpi=300):
    """
    Save a matplotlib figure to the specified directory.
    
    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        The figure to save.
    filename : str
        Name of the file (without directory).
    directory : str, optional
        Directory to save the figure in.
    dpi : int, optional
        Resolution of the saved figure.
        
    Returns:
    --------
    str
        Path to the saved figure.
    """
    # Create directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Full path
    filepath = os.path.join(directory, filename)
    
    # Save figure
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
    
    return filepath

def plot_bar_chart(data, x, y, title=None, xlabel=None, ylabel=None, 
                   figsize=(10, 6), color='skyblue', save_as=None):
    """
    Create a bar chart.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The dataframe containing the data.
    x : str
        Column name for x-axis.
    y : str
        Column name for y-axis.
    title : str, optional
        Chart title.
    xlabel : str, optional
        X-axis label.
    ylabel : str, optional
        Y-axis label.
    figsize : tuple, optional
        Figure size (width, height) in inches.
    color : str, optional
        Bar color.
    save_as : str, optional
        Filename to save the figure as. If None, the figure is not saved.
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create bar chart
    sns.barplot(x=x, y=y, data=data, color=color, ax=ax)
    
    # Set labels and title
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    
    # Rotate x-axis labels if there are many categories
    if data[x].nunique() > 5:
        plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Save figure if filename provided
    if save_as:
        save_figure(fig, save_as)
    
    return fig

def plot_histogram(data, column, bins=None, title=None, xlabel=None, ylabel='Frequency',
                   figsize=(10, 6), color='skyblue', kde=True, save_as=None):
    """
    Create a histogram with optional KDE.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The dataframe containing the data.
    column : str
        Column name to plot.
    bins : int, optional
        Number of bins. If None, uses Freedman-Diaconis rule.
    title : str, optional
        Chart title.
    xlabel : str, optional
        X-axis label.
    ylabel : str, optional
        Y-axis label.
    figsize : tuple, optional
        Figure size (width, height) in inches.
    color : str, optional
        Histogram color.
    kde : bool, optional
        Whether to overlay a KDE plot.
    save_as : str, optional
        Filename to save the figure as. If None, the figure is not saved.
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create histogram
    sns.histplot(data=data, x=column, bins=bins, kde=kde, color=color, ax=ax)
    
    # Set labels and title
    if xlabel:
        ax.set_xlabel(xlabel)
    else:
        ax.set_xlabel(column)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    
    plt.tight_layout()
    
    # Save figure if filename provided
    if save_as:
        save_figure(fig, save_as)
    
    return fig

def plot_scatter(data, x, y, hue=None, title=None, xlabel=None, ylabel=None,
                figsize=(10, 6), palette='viridis', alpha=0.7, save_as=None):
    """
    Create a scatter plot.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The dataframe containing the data.
    x : str
        Column name for x-axis.
    y : str
        Column name for y-axis.
    hue : str, optional
        Column name for color encoding.
    title : str, optional
        Chart title.
    xlabel : str, optional
        X-axis label.
    ylabel : str, optional
        Y-axis label.
    figsize : tuple, optional
        Figure size (width, height) in inches.
    palette : str or list, optional
        Color palette for hue.
    alpha : float, optional
        Transparency of points.
    save_as : str, optional
        Filename to save the figure as. If None, the figure is not saved.
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create scatter plot
    sns.scatterplot(x=x, y=y, hue=hue, data=data, palette=palette, alpha=alpha, ax=ax)
    
    # Set labels and title
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    
    # Add regression line
    sns.regplot(x=x, y=y, data=data, scatter=False, ax=ax, color='red')
    
    plt.tight_layout()
    
    # Save figure if filename provided
    if save_as:
        save_figure(fig, save_as)
    
    return fig

def plot_pie_chart(data, column, title=None, figsize=(10, 8), colors=None,
                  autopct='%1.1f%%', save_as=None):
    """
    Create a pie chart for categorical data.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The dataframe containing the data.
    column : str
        Column name to plot.
    title : str, optional
        Chart title.
    figsize : tuple, optional
        Figure size (width, height) in inches.
    colors : list, optional
        List of colors for pie slices.
    autopct : str, optional
        Format string for percentage labels.
    save_as : str, optional
        Filename to save the figure as. If None, the figure is not saved.
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Count values
    value_counts = data[column].value_counts()
    
    # Limit to top 10 categories if there are many
    if len(value_counts) > 10:
        other_count = value_counts[10:].sum()
        value_counts = value_counts[:10]
        value_counts['Other'] = other_count
    
    # Create pie chart
    wedges, texts, autotexts = ax.pie(
        value_counts, 
        labels=value_counts.index,
        autopct=autopct,
        colors=colors,
        startangle=90,
        wedgeprops={'edgecolor': 'w', 'linewidth': 1}
    )
    
    # Ensure text is readable
    for text in texts + autotexts:
        text.set_fontsize(10)
    
    # Equal aspect ratio ensures that pie is drawn as a circle
    ax.axis('equal')
    
    if title:
        ax.set_title(title)
    
    plt.tight_layout()
    
    # Save figure if filename provided
    if save_as:
        save_figure(fig, save_as)
    
    return fig

def plot_line_chart(data, x, y, hue=None, title=None, xlabel=None, ylabel=None,
                   figsize=(12, 6), palette='viridis', markers=True, save_as=None):
    """
    Create a line chart.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The dataframe containing the data.
    x : str
        Column name for x-axis.
    y : str
        Column name for y-axis.
    hue : str, optional
        Column name for color encoding.
    title : str, optional
        Chart title.
    xlabel : str, optional
        X-axis label.
    ylabel : str, optional
        Y-axis label.
    figsize : tuple, optional
        Figure size (width, height) in inches.
    palette : str or list, optional
        Color palette for hue.
    markers : bool, optional
        Whether to show markers at data points.
    save_as : str, optional
        Filename to save the figure as. If None, the figure is not saved.
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create line chart
    sns.lineplot(x=x, y=y, hue=hue, data=data, palette=palette, 
                markers=markers, ax=ax)
    
    # Set labels and title
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    
    plt.tight_layout()
    
    # Save figure if filename provided
    if save_as:
        save_figure(fig, save_as)
    
    return fig

def plot_box_plot(data, x, y, hue=None, title=None, xlabel=None, ylabel=None,
                 figsize=(12, 6), palette='viridis', save_as=None):
    """
    Create a box plot.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The dataframe containing the data.
    x : str
        Column name for x-axis.
    y : str
        Column name for y-axis.
    hue : str, optional
        Column name for color encoding.
    title : str, optional
        Chart title.
    xlabel : str, optional
        X-axis label.
    ylabel : str, optional
        Y-axis label.
    figsize : tuple, optional
        Figure size (width, height) in inches.
    palette : str or list, optional
        Color palette for hue.
    save_as : str, optional
        Filename to save the figure as. If None, the figure is not saved.
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create box plot
    sns.boxplot(x=x, y=y, hue=hue, data=data, palette=palette, ax=ax)
    
    # Set labels and title
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    
    # Rotate x-axis labels if there are many categories
    if data[x].nunique() > 5:
        plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Save figure if filename provided
    if save_as:
        save_figure(fig, save_as)
    
    return fig

def plot_heatmap(data, title=None, figsize=(12, 10), cmap='viridis', 
                annot=True, fmt='.2f', save_as=None):
    """
    Create a heatmap.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The dataframe to plot as heatmap (e.g., correlation matrix).
    title : str, optional
        Chart title.
    figsize : tuple, optional
        Figure size (width, height) in inches.
    cmap : str, optional
        Colormap for the heatmap.
    annot : bool, optional
        Whether to annotate cells with values.
    fmt : str, optional
        Format string for annotations.
    save_as : str, optional
        Filename to save the figure as. If None, the figure is not saved.
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(data, annot=annot, fmt=fmt, cmap=cmap, ax=ax)
    
    if title:
        ax.set_title(title)
    
    plt.tight_layout()
    
    # Save figure if filename provided
    if save_as:
        save_figure(fig, save_as)
    
    return fig

def plot_count_plot(data, x, hue=None, title=None, xlabel=None, ylabel='Count',
                   figsize=(10, 6), palette='viridis', save_as=None):
    """
    Create a count plot for categorical data.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The dataframe containing the data.
    x : str
        Column name for categories.
    hue : str, optional
        Column name for color encoding.
    title : str, optional
        Chart title.
    xlabel : str, optional
        X-axis label.
    ylabel : str, optional
        Y-axis label.
    figsize : tuple, optional
        Figure size (width, height) in inches.
    palette : str or list, optional
        Color palette for hue.
    save_as : str, optional
        Filename to save the figure as. If None, the figure is not saved.
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create count plot
    sns.countplot(x=x, hue=hue, data=data, palette=palette, ax=ax)
    
    # Set labels and title
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    
    # Rotate x-axis labels if there are many categories
    if data[x].nunique() > 5:
        plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Save figure if filename provided
    if save_as:
        save_figure(fig, save_as)
    
    return fig

def plot_pair_plot(data, columns=None, hue=None, title=None, 
                  figsize=(12, 12), palette='viridis', save_as=None):
    """
    Create a pair plot for multiple variables.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The dataframe containing the data.
    columns : list, optional
        List of column names to include. If None, uses all numeric columns.
    hue : str, optional
        Column name for color encoding.
    title : str, optional
        Chart title.
    figsize : tuple, optional
        Figure size (width, height) in inches.
    palette : str or list, optional
        Color palette for hue.
    save_as : str, optional
        Filename to save the figure as. If None, the figure is not saved.
        
    Returns:
    --------
    seaborn.axisgrid.PairGrid
        The pair grid object.
    """
    if columns is None:
        # Use all numeric columns
        columns = data.select_dtypes(include=['number']).columns.tolist()
    
    # Create pair plot
    pair_grid = sns.pairplot(data[columns + [hue] if hue else columns], 
                            hue=hue, palette=palette, height=figsize[0]/len(columns))
    
    if title:
        pair_grid.fig.suptitle(title, y=1.02)
    
    # Save figure if filename provided
    if save_as:
        save_figure(pair_grid.fig, save_as)
    
    return pair_grid
