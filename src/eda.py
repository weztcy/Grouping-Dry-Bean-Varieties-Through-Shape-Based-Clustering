"""
eda.py - Module for statistical analysis, distribution examination, and correlation checks.
"""

import numpy as np
import pandas as pd


def compute_descriptive_stats(X):
    """
    Compute comprehensive descriptive statistics for each feature:
    mean, std, min, 25%, median (50%), 75%, max, skewness, kurtosis, IQR.
    """
    stats = X.describe().T
    stats['skewness'] = X.skew()
    stats['kurtosis'] = X.kurtosis()
    stats['iqr'] = stats['75%'] - stats['25%']
    stats['range'] = stats['max'] - stats['min']
    
    # Calculate outlier counts using 1.5 * IQR rule
    outlier_counts = {}
    for col in X.columns:
        q25 = stats.loc[col, '25%']
        q75 = stats.loc[col, '75%']
        iqr = stats.loc[col, 'iqr']
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr
        outliers = ((X[col] < lower_bound) | (X[col] > upper_bound)).sum()
        outlier_counts[col] = int(outliers)
        
    stats['outliers_1.5iqr'] = pd.Series(outlier_counts)
    stats['outlier_pct'] = (stats['outliers_1.5iqr'] / len(X)) * 100
    
    return stats


def compute_correlation_matrix(X, method='pearson'):
    """
    Compute correlation matrix among features.
    """
    return X.corr(method=method)


def check_multicollinearity(X, threshold=0.90):
    """
    Identify pairs of features with absolute correlation above threshold.
    """
    corr = X.corr().abs()
    high_corr_pairs = []
    
    cols = X.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            c_val = corr.iloc[i, j]
            if c_val >= threshold:
                high_corr_pairs.append({
                    'feature_1': cols[i],
                    'feature_2': cols[j],
                    'correlation': float(c_val)
                })
                
    high_corr_pairs.sort(key=lambda x: x['correlation'], reverse=True)
    return pd.DataFrame(high_corr_pairs)
