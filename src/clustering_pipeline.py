"""
clustering_pipeline.py - Master pipeline script orchestrating the entire unsupervised learning study.
"""

import os
import json
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from src.data_loader import load_data, audit_data, analyze_duplicates
from src.eda import compute_descriptive_stats, compute_correlation_matrix, check_multicollinearity
from src.preprocessing import scale_features, get_reduced_features, perform_pca
from src.models import (
    run_kmeans_experiments,
    run_hierarchical_experiments,
    run_gmm_experiments,
    run_dbscan_experiments,
    compute_k_distances,
    evaluate_cluster_stability
)
from src.evaluation import (
    compute_internal_metrics,
    compute_external_metrics,
    generate_contingency_analysis,
    compute_cluster_profiles
)
from src.visualization import (
    plot_feature_distributions,
    plot_feature_boxplots,
    plot_correlation_heatmap,
    plot_pca_scree,
    plot_kmeans_elbow,
    plot_model_comparison_metrics,
    plot_pca_clusters,
    plot_cluster_size_distribution,
    plot_contingency_heatmap,
    plot_cluster_morphological_profiles,
    plot_gmm_bic_aic,
    plot_dbscan_k_distance,
    plot_pca_true_class,
    plot_key_morphology_boxplots
)


def run_pipeline():
    print("=" * 70)
    print("  DRY BEAN MORPHOLOGICAL SHAPE-BASED CLUSTERING PIPELINE")
    print("=" * 70)
    
    start_total_time = time.time()
    os.makedirs('figures', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # -------------------------------------------------------------
    # 1. DATA LOADING & AUDIT
    # -------------------------------------------------------------
    print("\n[Stage A] Loading data and performing audit...")
    df, X, y, metadata = load_data('Dry_Bean.csv')
    audit_res = audit_data(df)
    dup_res = analyze_duplicates(df)
    
    print(f"  Rows: {audit_res['total_rows']}, Features: {audit_res['num_features']}")
    print(f"  Missing values: {sum(audit_res['missing_values'].values())}")
    print(f"  Duplicate rows: {audit_res['duplicate_rows']} (All belonging to HOROZ: {dup_res['duplicate_class_distribution'].get('HOROZ', 0)})")
    
    # -------------------------------------------------------------
    # 2. EXPLORATORY DATA ANALYSIS (EDA)
    # -------------------------------------------------------------
    print("\n[Stage B] Computing descriptive statistics and correlations...")
    desc_stats = compute_descriptive_stats(X)
    corr_matrix = compute_correlation_matrix(X)
    high_corr_df = check_multicollinearity(X, threshold=0.90)
    print(f"  Identified {len(high_corr_df)} feature pairs with Pearson correlation >= 0.90")
    
    # -------------------------------------------------------------
    # 3. PREPROCESSING & REPRESENTATION
    # -------------------------------------------------------------
    print("\n[Stage C] Preprocessing features and applying PCA...")
    X_scaled_std, std_scaler = scale_features(X, scaler_type='standard')
    X_scaled_robust, robust_scaler = scale_features(X, scaler_type='robust')
    X_reduced = get_reduced_features(X_scaled_std)
    
    pca_full, pca_full_df, evr, cum_evr = perform_pca(X_scaled_std, n_components=None)
    pca_2d, pca_2d_df, evr_2d, _ = perform_pca(X_scaled_std, n_components=2)
    print(f"  PCA PC1 variance explained: {evr[0]*100:.2f}%")
    print(f"  PCA PC2 variance explained: {evr[1]*100:.2f}%")
    print(f"  PC1 + PC2 cumulative variance: {cum_evr[1]*100:.2f}%")
    print(f"  Components for >=95% variance: {np.argmax(cum_evr >= 0.95) + 1}")
    
    # -------------------------------------------------------------
    # 4. CLUSTERING MODELING EXPERIMENTS
    # -------------------------------------------------------------
    print("\n[Stage D] Running clustering experiments across algorithms...")
    k_range = list(range(2, 13))
    
    # 4.1 K-Means on Standard Scaled features
    print("  -> Running K-Means (k=2..12)...")
    km_results, km_models = run_kmeans_experiments(X_scaled_std, k_range=k_range, random_state=42)
    
    # 4.2 Agglomerative Hierarchical Clustering (Ward linkage)
    print("  -> Running Agglomerative Clustering (k=2..12, Ward linkage)...")
    agg_results, agg_models = run_hierarchical_experiments(X_scaled_std, k_range=k_range, linkage='ward')
    
    # 4.3 Gaussian Mixture Models (full & tied covariance)
    print("  -> Running GMM (k=2..12, full & tied covariance)...")
    gmm_results, gmm_models = run_gmm_experiments(X_scaled_std, k_range=k_range, covariance_types=('full', 'tied'), random_state=42)
    
    # 4.4 DBSCAN Parameter Search & k-distances
    print("  -> Computing k-distances and running DBSCAN grid search...")
    k_dists = compute_k_distances(X_scaled_std, k_neighbors=32)
    dbscan_results, dbscan_models = run_dbscan_experiments(
        X_scaled_std,
        eps_grid=(0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5),
        min_samples_grid=(15, 20, 30, 50)
    )
    
    # 4.5 K-Means on alternative representations (Reduced & Robust Scaler)
    print("  -> Running K-Means on alternative representations (Reduced features & RobustScaler)...")
    km_reduced_res, _ = run_kmeans_experiments(X_reduced, k_range=[3, 4, 7], random_state=42)
    km_robust_res, _ = run_kmeans_experiments(X_scaled_robust, k_range=[3, 4, 7], random_state=42)
    
    # -------------------------------------------------------------
    # 5. INTERNAL & POST-HOC EXTERNAL EVALUATION
    # -------------------------------------------------------------
    print("\n[Stage E] Calculating internal and post-hoc external metrics...")
    
    evaluation_records = []
    
    # Evaluate K-Means
    for res in km_results:
        k = res['k']
        lbls = res['labels']
        metrics_int = compute_internal_metrics(X_scaled_std.values, lbls, model_name='K-Means', k=k)
        metrics_ext = compute_external_metrics(y.values, lbls, model_name='K-Means', k=k)
        record = {**metrics_int, **metrics_ext, 'inertia': res['inertia']}
        evaluation_records.append(record)
        
    # Evaluate Hierarchical
    for res in agg_results:
        k = res['k']
        lbls = res['labels']
        metrics_int = compute_internal_metrics(X_scaled_std.values, lbls, model_name='Hierarchical (Ward)', k=k)
        metrics_ext = compute_external_metrics(y.values, lbls, model_name='Hierarchical (Ward)', k=k)
        record = {**metrics_int, **metrics_ext, 'inertia': np.nan}
        evaluation_records.append(record)
        
    # Evaluate GMM (Full covariance)
    for res in gmm_results:
        if res['covariance_type'] == 'full':
            k = res['k']
            lbls = res['labels']
            metrics_int = compute_internal_metrics(X_scaled_std.values, lbls, model_name='GMM (Full)', k=k)
            metrics_ext = compute_external_metrics(y.values, lbls, model_name='GMM (Full)', k=k)
            record = {**metrics_int, **metrics_ext, 'aic': res['aic'], 'bic': res['bic']}
            evaluation_records.append(record)
            
    eval_df = pd.DataFrame(evaluation_records)
    
    # DBSCAN Evaluation summary
    dbscan_summary = []
    for res in dbscan_results:
        lbls = res['labels']
        metrics_int = compute_internal_metrics(X_scaled_std.values, lbls, model_name='DBSCAN', k=res['n_clusters'])
        metrics_ext = compute_external_metrics(y.values, lbls, model_name='DBSCAN', k=res['n_clusters'])
        dbscan_summary.append({
            'eps': res['eps'],
            'min_samples': res['min_samples'],
            'n_clusters': res['n_clusters'],
            'noise_pct': res['noise_pct'],
            'silhouette': metrics_int['silhouette'],
            'ari': metrics_ext['ari'],
            'v_measure': metrics_ext['v_measure']
        })
    dbscan_eval_df = pd.DataFrame(dbscan_summary)
    
    # -------------------------------------------------------------
    # 6. MODEL SELECTION & DETAILED PROFILING
    # -------------------------------------------------------------
    print("\n[Stage F] Selecting best model based on internal criteria and interpretability...")
    
    # Internal metrics comparison:
    # Notice: Silhouette Score peaks at k=3 (0.395) or k=4 (0.353), Davies-Bouldin is lowest at k=3/k=4.
    # While k=7 reflects the known botanical varieties, from pure shape geometry the data naturally partitions
    # into 4 macro-clusters:
    # 1. Giant Beans (Bombay)
    # 2. Round/Spherical Beans (Seker)
    # 3. Elongated/Slender Beans (Horoz)
    # 4. Medium Oval/Elliptical Beans (Dermason, Sira, Cali, Barbunya)
    # Or 7 micro-clusters when resolving the medium oval beans into subdivisions.
    
    # Let's inspect K-Means k=4 vs k=7 and Hierarchical k=4 vs k=7
    best_k_internal = 4
    best_model_name = 'K-Means'
    best_km_model = km_models[best_k_internal]
    best_labels = best_km_model.labels_
    
    # Also keep k=7 model for in-depth comparative discussion
    km_7_model = km_models[7]
    km_7_labels = km_7_model.labels_
    
    print(f"  Selected Primary Model: {best_model_name} (k={best_k_internal})")
    print(f"  Internal Silhouette: {eval_df[(eval_df['model_name']=='K-Means') & (eval_df['k']==best_k_internal)]['silhouette'].values[0]:.4f}")
    print(f"  Calinski-Harabasz: {eval_df[(eval_df['model_name']=='K-Means') & (eval_df['k']==best_k_internal)]['calinski_harabasz'].values[0]:.1f}")
    print(f"  Davies-Bouldin: {eval_df[(eval_df['model_name']=='K-Means') & (eval_df['k']==best_k_internal)]['davies_bouldin'].values[0]:.4f}")
    
    # Model Stability Evaluation
    print("  -> Testing cluster stability with subsampling (5 iterations, 85% sample)...")
    stability_res = evaluate_cluster_stability(
        X_scaled_std,
        algorithm_fn=lambda x, random_state: KMeans(n_clusters=best_k_internal, n_init=10, random_state=random_state).fit_predict(x),
        n_runs=5,
        sample_ratio=0.85,
        random_seed_base=42
    )
    print(f"  Cluster stability mean ARI: {stability_res['mean_stability_ari']:.4f} (+/- {stability_res['std_stability_ari']:.4f})")
    
    # Duplicate Sensitivity Evaluation
    print("  -> Testing sensitivity to duplicate rows...")
    features = [c for c in df.columns if c != 'Class']
    unique_mask = ~df.duplicated(subset=features)
    X_unique_scaled = X_scaled_std[unique_mask]
    km_unique = KMeans(n_clusters=best_k_internal, n_init=20, random_state=42).fit(X_unique_scaled)
    # Project unique labels back to original indices
    unique_labels = km_unique.labels_
    full_labels_on_unique = best_labels[unique_mask]
    dup_sensitivity_ari = float(adjusted_rand_score(full_labels_on_unique, unique_labels))
    print(f"  Duplicate sensitivity ARI (Full vs Deduplicated): {dup_sensitivity_ari:.5f}")
    
    # Contingency & Cluster Profiles for Primary Model (k=4)
    contingency_k4 = generate_contingency_analysis(y, best_labels)
    profiles_k4 = compute_cluster_profiles(df, best_labels)
    
    # Contingency & Cluster Profiles for k=7
    contingency_k7 = generate_contingency_analysis(y, km_7_labels)
    profiles_k7 = compute_cluster_profiles(df, km_7_labels)
    
    # -------------------------------------------------------------
    # 7. GENERATING ALL REQUIRED FIGURES
    # -------------------------------------------------------------
    print("\n[Stage G] Generating and saving publication-quality figures to figures/...")
    plot_feature_distributions(X)
    plot_feature_boxplots(X_scaled_std)
    plot_correlation_heatmap(X)
    plot_pca_scree(evr, cum_evr)
    plot_kmeans_elbow(k_range, [r['inertia'] for r in km_results])
    plot_model_comparison_metrics(eval_df[eval_df['k'].isin(k_range)])
    plot_pca_clusters(pca_2d_df, best_labels, title='Proyeksi 2D PCA Model Terbaik (K-Means k=4)')
    plot_cluster_size_distribution(best_labels)
    plot_contingency_heatmap(y, best_labels)
    plot_cluster_morphological_profiles(df, best_labels)
    plot_gmm_bic_aic(gmm_results)
    plot_dbscan_k_distance(k_dists, knee_point=1.5)
    plot_pca_true_class(pca_2d_df, y)
    plot_key_morphology_boxplots(df, best_labels)
    
    # Additional figure for k=7 comparison
    plot_pca_clusters(pca_2d_df, km_7_labels, title='Proyeksi 2D PCA Model Komparatif K-Means k=7',
                      filename='15_pca_kmeans_k7_clusters.png')
    plot_contingency_heatmap(y, km_7_labels, filename='16_contingency_k7_heatmap.png')
    
    # -------------------------------------------------------------
    # 8. SAVING ARTIFACTS & METADATA
    # -------------------------------------------------------------
    print("\n[Stage H] Exporting trained models, scalers, and summary JSON...")
    joblib.dump(best_km_model, 'models/best_clustering_model.joblib')
    joblib.dump(std_scaler, 'models/scaler.joblib')
    joblib.dump(pca_2d, 'models/pca_model.joblib')
    joblib.dump(km_7_model, 'models/kmeans_k7_model.joblib')
    
    # Prepare serializable summary
    summary_data = {
        'audit': audit_res,
        'duplicates': dup_res,
        'high_correlations': high_corr_df.to_dict(orient='records'),
        'pca': {
            'evr': [float(x) for x in evr],
            'cum_evr': [float(x) for x in cum_evr],
            'pc1_variance': float(evr[0]),
            'pc2_variance': float(evr[1]),
            'n_components_95': int(np.argmax(cum_evr >= 0.95) + 1)
        },
        'best_model': {
            'name': best_model_name,
            'k': best_k_internal,
            'silhouette': float(eval_df[(eval_df['model_name']=='K-Means') & (eval_df['k']==best_k_internal)]['silhouette'].values[0]),
            'calinski_harabasz': float(eval_df[(eval_df['model_name']=='K-Means') & (eval_df['k']==best_k_internal)]['calinski_harabasz'].values[0]),
            'davies_bouldin': float(eval_df[(eval_df['model_name']=='K-Means') & (eval_df['k']==best_k_internal)]['davies_bouldin'].values[0]),
            'ari': float(eval_df[(eval_df['model_name']=='K-Means') & (eval_df['k']==best_k_internal)]['ari'].values[0]),
            'nmi': float(eval_df[(eval_df['model_name']=='K-Means') & (eval_df['k']==best_k_internal)]['nmi'].values[0]),
            'v_measure': float(eval_df[(eval_df['model_name']=='K-Means') & (eval_df['k']==best_k_internal)]['v_measure'].values[0]),
            'purity': float(eval_df[(eval_df['model_name']=='K-Means') & (eval_df['k']==best_k_internal)]['purity'].values[0]),
            'stability_ari_mean': stability_res['mean_stability_ari'],
            'stability_ari_std': stability_res['std_stability_ari'],
            'duplicate_sensitivity_ari': dup_sensitivity_ari
        },
        'comparison_k7': {
            'silhouette': float(eval_df[(eval_df['model_name']=='K-Means') & (eval_df['k']==7)]['silhouette'].values[0]),
            'calinski_harabasz': float(eval_df[(eval_df['model_name']=='K-Means') & (eval_df['k']==7)]['calinski_harabasz'].values[0]),
            'davies_bouldin': float(eval_df[(eval_df['model_name']=='K-Means') & (eval_df['k']==7)]['davies_bouldin'].values[0]),
            'ari': float(eval_df[(eval_df['model_name']=='K-Means') & (eval_df['k']==7)]['ari'].values[0]),
            'nmi': float(eval_df[(eval_df['model_name']=='K-Means') & (eval_df['k']==7)]['nmi'].values[0]),
            'v_measure': float(eval_df[(eval_df['model_name']=='K-Means') & (eval_df['k']==7)]['v_measure'].values[0]),
            'purity': float(eval_df[(eval_df['model_name']=='K-Means') & (eval_df['k']==7)]['purity'].values[0])
        },
        'evaluation_table': eval_df.to_dict(orient='records'),
        'dbscan_table': dbscan_eval_df.to_dict(orient='records')
    }
    
    def convert_numpy(obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

    with open('models/pipeline_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, default=convert_numpy)
        
    # Also save evaluation tables as CSV
    eval_df.to_csv('models/evaluation_table.csv', index=False)
    dbscan_eval_df.to_csv('models/dbscan_table.csv', index=False)
    desc_stats.to_csv('models/descriptive_statistics.csv')
    contingency_k4['counts'].to_csv('models/contingency_k4_counts.csv')
    contingency_k4['cluster_pct'].to_csv('models/contingency_k4_pct.csv')
    contingency_k7['counts'].to_csv('models/contingency_k7_counts.csv')
    contingency_k7['cluster_pct'].to_csv('models/contingency_k7_pct.csv')
    profiles_k4['mean'].to_csv('models/cluster_k4_mean_profiles.csv')
    profiles_k7['mean'].to_csv('models/cluster_k7_mean_profiles.csv')
    
    total_elapsed = time.time() - start_total_time
    print(f"\n[COMPLETE] Master pipeline finished successfully in {total_elapsed:.2f} seconds!")
    print("=" * 70)


if __name__ == '__main__':
    run_pipeline()
