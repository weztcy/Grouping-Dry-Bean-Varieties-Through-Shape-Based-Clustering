"""
models.py - Module for executing clustering algorithms and stability tests.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import adjusted_rand_score


def run_kmeans_experiments(X, k_range=range(2, 13), random_state=42, n_init=20):
    """
    Run K-Means across a range of k values, recording models, labels, and inertia.
    """
    results = []
    models = {}
    
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=n_init, random_state=random_state)
        labels = km.fit_predict(X)
        models[k] = km
        results.append({
            'k': k,
            'inertia': float(km.inertia_),
            'labels': labels,
            'model': km
        })
        
    return results, models


def run_hierarchical_experiments(X, k_range=range(2, 13), linkage='ward'):
    """
    Run Agglomerative Hierarchical Clustering across a range of k values.
    """
    results = []
    models = {}
    
    for k in k_range:
        agg = AgglomerativeClustering(n_clusters=k, linkage=linkage)
        labels = agg.fit_predict(X)
        models[k] = agg
        results.append({
            'k': k,
            'labels': labels,
            'model': agg
        })
        
    return results, models


def run_gmm_experiments(X, k_range=range(2, 13), covariance_types=('full', 'tied'), random_state=42):
    """
    Run Gaussian Mixture Model across different components and covariance types.
    Records AIC, BIC, hard cluster assignments, and posterior probabilities.
    """
    results = []
    models = {}
    
    for cov_type in covariance_types:
        for k in k_range:
            gmm = GaussianMixture(n_components=k, covariance_type=cov_type, random_state=random_state)
            labels = gmm.fit_predict(X)
            probs = gmm.predict_proba(X)
            aic = gmm.aic(X)
            bic = gmm.bic(X)
            
            key = (cov_type, k)
            models[key] = gmm
            results.append({
                'k': k,
                'covariance_type': cov_type,
                'aic': float(aic),
                'bic': float(bic),
                'labels': labels,
                'probabilities': probs,
                'model': gmm
            })
            
    return results, models


def compute_k_distances(X, k_neighbors=32):
    """
    Compute k-distance for DBSCAN parameter tuning.
    """
    nn = NearestNeighbors(n_neighbors=k_neighbors)
    nn.fit(X)
    distances, _ = nn.kneighbors(X)
    k_dist = np.sort(distances[:, -1])
    return k_dist


def run_dbscan_experiments(X, eps_grid=(0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5), min_samples_grid=(15, 20, 30, 50)):
    """
    Run DBSCAN grid search over eps and min_samples.
    """
    results = []
    models = {}
    
    for eps in eps_grid:
        for ms in min_samples_grid:
            db = DBSCAN(eps=eps, min_samples=ms)
            labels = db.fit_predict(X)
            unique_labels = np.unique(labels)
            n_clusters = len(unique_labels[unique_labels != -1])
            noise_count = int(np.sum(labels == -1))
            noise_pct = (noise_count / len(labels)) * 100
            
            key = (eps, ms)
            models[key] = db
            results.append({
                'eps': eps,
                'min_samples': ms,
                'n_clusters': n_clusters,
                'noise_count': noise_count,
                'noise_pct': noise_pct,
                'labels': labels,
                'model': db
            })
            
    return results, models


def evaluate_cluster_stability(X, algorithm_fn, n_runs=5, sample_ratio=0.85, random_seed_base=42):
    """
    Evaluate cluster assignment stability using subsampling.
    Measures pairwise Adjusted Rand Index (ARI) on common subsampled instances.
    """
    n_samples = len(X)
    subsample_size = int(n_samples * sample_ratio)
    
    subsample_indices = []
    subsample_labels = []
    
    for i in range(n_runs):
        rng = np.random.RandomState(random_seed_base + i * 17)
        idx = rng.choice(n_samples, size=subsample_size, replace=False)
        idx_sorted = np.sort(idx)
        X_sub = X.iloc[idx_sorted] if hasattr(X, 'iloc') else X[idx_sorted]
        
        lbls = algorithm_fn(X_sub, random_state=random_seed_base + i * 17)
        subsample_indices.append(idx_sorted)
        subsample_labels.append(lbls)
        
    pairwise_aris = []
    for i in range(n_runs):
        for j in range(i + 1, n_runs):
            # Find intersecting indices
            common_idx, idx_i, idx_j = np.intersect1d(subsample_indices[i], subsample_indices[j], return_indices=True)
            if len(common_idx) > 10:
                ari = adjusted_rand_score(subsample_labels[i][idx_i], subsample_labels[j][idx_j])
                pairwise_aris.append(ari)
                
    return {
        'mean_stability_ari': float(np.mean(pairwise_aris)),
        'std_stability_ari': float(np.std(pairwise_aris)),
        'min_stability_ari': float(np.min(pairwise_aris)),
        'max_stability_ari': float(np.max(pairwise_aris)),
        'pairwise_aris': [float(x) for x in pairwise_aris]
    }
