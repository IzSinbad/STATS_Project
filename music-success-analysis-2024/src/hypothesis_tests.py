"""
Hypothesis testing and statistical analysis for music streaming data.

This module provides functions for performing various statistical tests
and analyses on music streaming and listener preference data.
"""

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import seaborn as sns

def correlation_test(data, x, y, method='pearson'):
    """
    Perform correlation test between two variables.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The dataframe containing the data.
    x : str
        Column name for first variable.
    y : str
        Column name for second variable.
    method : str, optional
        Correlation method ('pearson', 'spearman', or 'kendall').
        
    Returns:
    --------
    corr_coef : float
        Correlation coefficient.
    p_value : float
        P-value for the correlation test.
    """
    if method == 'pearson':
        corr_coef, p_value = stats.pearsonr(data[x], data[y])
    elif method == 'spearman':
        corr_coef, p_value = stats.spearmanr(data[x], data[y])
    elif method == 'kendall':
        corr_coef, p_value = stats.kendalltau(data[x], data[y])
    else:
        raise ValueError("Method must be 'pearson', 'spearman', or 'kendall'")
    
    return corr_coef, p_value

def t_test_independent(data, value_col, group_col, group1, group2, equal_var=False):
    """
    Perform independent samples t-test.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The dataframe containing the data.
    value_col : str
        Column name with the values to compare.
    group_col : str
        Column name containing group identifiers.
    group1 : str
        Name of first group to compare.
    group2 : str
        Name of second group to compare.
    equal_var : bool, optional
        Whether to assume equal variances.
        
    Returns:
    --------
    t_stat : float
        T-statistic.
    p_value : float
        P-value.
    effect_size : float
        Cohen's d effect size.
    """
    # Extract data for each group
    group1_data = data[data[group_col] == group1][value_col].dropna()
    group2_data = data[data[group_col] == group2][value_col].dropna()
    
    # Perform t-test
    t_stat, p_value = stats.ttest_ind(group1_data, group2_data, equal_var=equal_var)
    
    # Calculate effect size (Cohen's d)
    n1, n2 = len(group1_data), len(group2_data)
    mean1, mean2 = group1_data.mean(), group2_data.mean()
    var1, var2 = group1_data.var(), group2_data.var()
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    # Cohen's d
    effect_size = abs(mean1 - mean2) / pooled_std
    
    return t_stat, p_value, effect_size

def anova_test(data, value_col, group_col):
    """
    Perform one-way ANOVA test.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The dataframe containing the data.
    value_col : str
        Column name with the values to compare.
    group_col : str
        Column name containing group identifiers.
        
    Returns:
    --------
    f_stat : float
        F-statistic.
    p_value : float
        P-value.
    groups_data : dict
        Dictionary with group data (means, counts, etc.).
    """
    # Group data
    groups = data[group_col].unique()
    groups_data = {}
    
    # Extract data for each group
    for group in groups:
        group_values = data[data[group_col] == group][value_col].dropna()
        groups_data[group] = {
            'values': group_values,
            'mean': group_values.mean(),
            'std': group_values.std(),
            'count': len(group_values)
        }
    
    # Prepare data for ANOVA
    anova_data = [groups_data[group]['values'] for group in groups]
    
    # Perform ANOVA
    f_stat, p_value = stats.f_oneway(*anova_data)
    
    return f_stat, p_value, groups_data

def chi_square_test(data, col1, col2):
    """
    Perform chi-square test of independence.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The dataframe containing the data.
    col1 : str
        First categorical column name.
    col2 : str
        Second categorical column name.
        
    Returns:
    --------
    chi2 : float
        Chi-square statistic.
    p_value : float
        P-value.
    dof : int
        Degrees of freedom.
    expected : numpy.ndarray
        Expected frequencies array.
    """
    # Create contingency table
    contingency_table = pd.crosstab(data[col1], data[col2])
    
    # Perform chi-square test
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
    
    return chi2, p_value, dof, expected

def linear_regression(data, formula, plot=False):
    """
    Perform linear regression analysis.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The dataframe containing the data.
    formula : str
        Formula for the regression (e.g., 'y ~ x1 + x2').
    plot : bool, optional
        Whether to create diagnostic plots.
        
    Returns:
    --------
    model : statsmodels.regression.linear_model.RegressionResultsWrapper
        Fitted regression model.
    """
    # Fit the model
    model = ols(formula, data=data).fit()
    
    if plot:
        # Create diagnostic plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Residuals vs Fitted
        sns.residplot(x=model.fittedvalues, y=model.resid, lowess=True, 
                     scatter_kws={'alpha': 0.5}, line_kws={'color': 'red', 'lw': 1}, ax=axes[0, 0])
        axes[0, 0].set_title('Residuals vs Fitted')
        axes[0, 0].set_xlabel('Fitted values')
        axes[0, 0].set_ylabel('Residuals')
        
        # QQ plot
        sm.qqplot(model.resid, fit=True, line='45', ax=axes[0, 1])
        axes[0, 1].set_title('Normal Q-Q')
        
        # Scale-Location
        standardized_resid = np.sqrt(np.abs(model.get_influence().resid_studentized_internal))
        sns.regplot(x=model.fittedvalues, y=standardized_resid, lowess=True,
                   scatter_kws={'alpha': 0.5}, line_kws={'color': 'red', 'lw': 1}, ax=axes[1, 0])
        axes[1, 0].set_title('Scale-Location')
        axes[1, 0].set_xlabel('Fitted values')
        axes[1, 0].set_ylabel('$\\sqrt{|Standardized Residuals|}$')
        
        # Leverage
        influence = model.get_influence()
        leverage = influence.hat_matrix_diag
        sns.regplot(x=leverage, y=standardized_resid, lowess=True,
                   scatter_kws={'alpha': 0.5}, line_kws={'color': 'red', 'lw': 1}, ax=axes[1, 1])
        axes[1, 1].set_title('Residuals vs Leverage')
        axes[1, 1].set_xlabel('Leverage')
        axes[1, 1].set_ylabel('Standardized Residuals')
        
        plt.tight_layout()
    
    return model

def multiple_comparison_test(data, value_col, group_col, alpha=0.05):
    """
    Perform multiple comparison test after ANOVA.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The dataframe containing the data.
    value_col : str
        Column name with the values to compare.
    group_col : str
        Column name containing group identifiers.
    alpha : float, optional
        Significance level.
        
    Returns:
    --------
    result : pandas.DataFrame
        DataFrame with pairwise comparison results.
    """
    # Group data
    groups = data[group_col].unique()
    group_data = [data[data[group_col] == group][value_col].dropna() for group in groups]
    
    # Perform Tukey's HSD test
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    tukey_result = pairwise_tukeyhsd(data[value_col].dropna(), data[group_col].dropna(), alpha=alpha)
    
    # Convert to DataFrame
    result = pd.DataFrame(data=tukey_result._results_table.data[1:], 
                         columns=tukey_result._results_table.data[0])
    
    return result

def mann_whitney_test(data, value_col, group_col, group1, group2):
    """
    Perform Mann-Whitney U test (non-parametric alternative to t-test).
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The dataframe containing the data.
    value_col : str
        Column name with the values to compare.
    group_col : str
        Column name containing group identifiers.
    group1 : str
        Name of first group to compare.
    group2 : str
        Name of second group to compare.
        
    Returns:
    --------
    u_stat : float
        U-statistic.
    p_value : float
        P-value.
    """
    # Extract data for each group
    group1_data = data[data[group_col] == group1][value_col].dropna()
    group2_data = data[data[group_col] == group2][value_col].dropna()
    
    # Perform Mann-Whitney U test
    u_stat, p_value = stats.mannwhitneyu(group1_data, group2_data, alternative='two-sided')
    
    return u_stat, p_value

def kruskal_wallis_test(data, value_col, group_col):
    """
    Perform Kruskal-Wallis H test (non-parametric alternative to ANOVA).
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The dataframe containing the data.
    value_col : str
        Column name with the values to compare.
    group_col : str
        Column name containing group identifiers.
        
    Returns:
    --------
    h_stat : float
        H-statistic.
    p_value : float
        P-value.
    """
    # Group data
    groups = data[group_col].unique()
    group_data = [data[data[group_col] == group][value_col].dropna() for group in groups]
    
    # Perform Kruskal-Wallis H test
    h_stat, p_value = stats.kruskal(*group_data)
    
    return h_stat, p_value

def shapiro_normality_test(data, column):
    """
    Perform Shapiro-Wilk test for normality.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The dataframe containing the data.
    column : str
        Column name to test for normality.
        
    Returns:
    --------
    w_stat : float
        W-statistic.
    p_value : float
        P-value.
    """
    # Extract data
    values = data[column].dropna()
    
    # Perform Shapiro-Wilk test
    w_stat, p_value = stats.shapiro(values)
    
    return w_stat, p_value

def levene_variance_test(data, value_col, group_col):
    """
    Perform Levene's test for equality of variances.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The dataframe containing the data.
    value_col : str
        Column name with the values to compare.
    group_col : str
        Column name containing group identifiers.
        
    Returns:
    --------
    w_stat : float
        W-statistic.
    p_value : float
        P-value.
    """
    # Group data
    groups = data[group_col].unique()
    group_data = [data[data[group_col] == group][value_col].dropna() for group in groups]
    
    # Perform Levene's test
    w_stat, p_value = stats.levene(*group_data)
    
    return w_stat, p_value

def partial_correlation(data, x, y, covariates):
    """
    Calculate partial correlation between two variables controlling for covariates.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The dataframe containing the data.
    x : str
        Column name for first variable.
    y : str
        Column name for second variable.
    covariates : list
        List of column names for covariates to control for.
        
    Returns:
    --------
    corr_coef : float
        Partial correlation coefficient.
    p_value : float
        P-value for the partial correlation.
    """
    # Extract data
    variables = [x, y] + covariates
    data_subset = data[variables].dropna()
    
    # Calculate correlation matrix
    corr_matrix = data_subset.corr()
    
    # Calculate partial correlation
    from scipy.stats import t
    
    # Indices of x and y in the correlation matrix
    idx_x = variables.index(x)
    idx_y = variables.index(y)
    
    # Inverse of correlation matrix
    corr_inv = np.linalg.inv(corr_matrix.values)
    
    # Partial correlation formula
    pcorr = -corr_inv[idx_x, idx_y] / np.sqrt(corr_inv[idx_x, idx_x] * corr_inv[idx_y, idx_y])
    
    # Calculate p-value
    n = len(data_subset)
    k = len(covariates)
    t_stat = pcorr * np.sqrt((n - k - 2) / (1 - pcorr**2))
    p_value = 2 * (1 - t.cdf(abs(t_stat), n - k - 2))
    
    return pcorr, p_value
