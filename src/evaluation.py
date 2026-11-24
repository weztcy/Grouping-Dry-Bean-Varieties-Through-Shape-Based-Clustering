"""
evaluation.py - Module for calculating internal clustering metrics and post-hoc external validation.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    homogeneity_completeness_v_measure
)


def compute_internal_metrics(X, labels, model_name=None, k=None, sample_size=None, random_state=42):
    """
    Compute internal clustering validation metrics:
    - Silhouette Score (-1 to 1, higher is better)
    - Calinski-Harabasz Index (higher is better)
    - Davies-Bouldin Index (lower is better)
    - Cluster sizes and distribution
    """
    # Filter out noise if DBSCAN (-1)
    mask = labels != -1
    n_clusters = len(np.unique(labels[mask])) if mask.sum() > 0 else 0
    noise_count = int(np.sum(labels == -1))
    noise_ratio = float(noise_count / len(labels)) if len(labels) > 0 else 0.0
    
    if n_clusters < 2:
        return {
            'model_name': model_name,
            'k': k,
            'n_clusters_found': n_clusters,
            'silhouette': np.nan,
            'calinski_harabasz': np.nan,
            'davies_bouldin': np.nan,
            'noise_count': noise_count,
            'noise_ratio': noise_ratio,
            'cluster_sizes': {}
        }
        
    X_eval = X[mask] if mask.sum() < len(labels) else X
    labels_eval = labels[mask] if mask.sum() < len(labels) else labels
    
    # Compute silhouette score (sample_size can be set if needed, or full)
    try:
        if sample_size and sample_size < len(X_eval):
            sil = silhouette_score(X_eval, labels_eval, sample_size=sample_size, random_state=random_state)
        else:
            sil = silhouette_score(X_eval, labels_eval)
    except Exception:
        sil = np.nan
        
    try:
        ch = calinski_harabasz_score(X_eval, labels_eval)
    except Exception:
        ch = np.nan
        
    try:
        db = davies_bouldin_score(X_eval, labels_eval)
    except Exception:
        db = np.nan
        
    unique_labels, counts = np.unique(labels, return_counts=True)
    cluster_sizes = {int(lbl): int(cnt) for lbl, cnt in zip(unique_labels, counts)}
    
    return {
        'model_name': model_name,
        'k': k,
        'n_clusters_found': n_clusters,
        'silhouette': float(sil),
        'calinski_harabasz': float(ch),
        'davies_bouldin': float(db),
        'noise_count': noise_count,
        'noise_ratio': noise_ratio,
        'cluster_sizes': cluster_sizes
    }


def compute_external_metrics(y_true, labels, model_name=None, k=None):
    """
    Perform post-hoc external validation comparing cluster assignments with known bean varieties.
    NOTE: y_true (Class) must NEVER be used in training or feature selection.
    """
    # Exclude noise points (-1) if any
    mask = labels != -1
    if mask.sum() == 0 or len(np.unique(labels[mask])) < 2:
        return {
            'model_name': model_name,
            'k': k,
            'ari': np.nan,
            'nmi': np.nan,
            'homogeneity': np.nan,
            'completeness': np.nan,
            'v_measure': np.nan,
            'purity': np.nan
        }
        
    y_eval = y_true[mask] if mask.sum() < len(labels) else y_true
    labels_eval = labels[mask] if mask.sum() < len(labels) else labels
    
    ari = adjusted_rand_score(y_eval, labels_eval)
    nmi = normalized_mutual_info_score(y_eval, labels_eval)
    homo, comp, vm = homogeneity_completeness_v_measure(y_eval, labels_eval)
    
    # Calculate overall cluster purity
    contingency = pd.crosstab(labels_eval, y_eval)
    purity = contingency.max(axis=1).sum() / len(y_eval)
    
    return {
        'model_name': model_name,
        'k': k,
        'ari': float(ari),
        'nmi': float(nmi),
        'homogeneity': float(homo),
        'completeness': float(comp),
        'v_measure': float(vm),
        'purity': float(purity)
    }


def generate_contingency_analysis(y_true, labels):
    """
    Build detailed cross-tabulation table with counts and percentage compositions.
    """
    df_temp = pd.DataFrame({'Cluster': labels, 'Class': y_true})
    
    # Raw counts
    counts_table = pd.crosstab(df_temp['Cluster'], df_temp['Class'], margins=True, margins_name='Total')
    
    # Row percentages (what % of each cluster consists of each variety)
    cluster_pct = pd.crosstab(df_temp['Cluster'], df_temp['Class'], normalize='index') * 100
    
    # Column percentages (how each variety is distributed across clusters)
    variety_pct = pd.crosstab(df_temp['Cluster'], df_temp['Class'], normalize='columns') * 100
    
    return {
        'counts': counts_table,
        'cluster_pct': cluster_pct,
        'variety_pct': variety_pct
    }


def compute_cluster_profiles(df_raw, labels):
    """
    Calculate summary statistics (mean, median, std) for each cluster across raw features.
    """
    df_temp = df_raw.copy()
    features = [c for c in df_temp.columns if c != 'Class']
    df_temp['Cluster'] = labels
    
    profiles_mean = df_temp.groupby('Cluster')[features].mean()
    profiles_median = df_temp.groupby('Cluster')[features].median()
    profiles_std = df_temp.groupby('Cluster')[features].std()
    
    return {
        'mean': profiles_mean,
        'median': profiles_median,
        'std': profiles_std
    }
