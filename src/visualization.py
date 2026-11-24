"""
visualization.py - Module for generating high-resolution, publication-quality figures.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set global aesthetic style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['font.size'] = 10


def save_figure(fig, filename, output_dir='figures', dpi=300):
    """Utility to safely save figure with tight bounding box."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {filepath}")
    return filepath


def plot_feature_distributions(X, filename='01_feature_distribution_hist_kde.png', output_dir='figures'):
    """Plot histograms and KDEs for all 16 morphological features."""
    features = X.columns.tolist()
    fig, axes = plt.subplots(4, 4, figsize=(18, 14))
    axes = axes.flatten()
    
    for i, col in enumerate(features):
        sns.histplot(X[col], kde=True, ax=axes[i], color='#1f77b4', edgecolor='black', alpha=0.6, bins=30)
        axes[i].set_title(f"{col} (Skew: {X[col].skew():.2f})", fontsize=11, fontweight='bold')
        axes[i].set_xlabel('')
        axes[i].set_ylabel('Frekuensi')
        
    plt.suptitle("Distribusi dan Kepadatan Fitur Morfologi Biji Kacang (Histogram & KDE)", fontsize=16, fontweight='bold', y=0.99)
    plt.tight_layout()
    return save_figure(fig, filename, output_dir)


def plot_feature_boxplots(X_scaled_df, filename='02_feature_boxplots_outliers.png', output_dir='figures'):
    """Plot standardized boxplots for all 16 features to detect outliers."""
    fig, ax = plt.subplots(figsize=(16, 8))
    
    sns.boxplot(data=X_scaled_df, ax=ax, palette='Set3', flierprops={'marker': 'o', 'markersize': 3, 'alpha': 0.4})
    ax.tick_params(axis='x', rotation=45)
    ax.set_ylabel('Nilai Terstandarisasi (Z-Score)', fontsize=12, fontweight='bold')
    ax.set_title('Pemeriksaan Sebaran Data & Outlier Fitur Morfologi (Standard Scaled)', fontsize=15, fontweight='bold')
    ax.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.7)
    
    plt.tight_layout()
    return save_figure(fig, filename, output_dir)


def plot_correlation_heatmap(X, filename='03_correlation_heatmap.png', output_dir='figures'):
    """Plot correlation heatmap with annotations for all 16 features."""
    fig, ax = plt.subplots(figsize=(14, 12))
    
    corr = X.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0,
                annot=True, fmt='.2f', square=True, linewidths=0.5,
                cbar_kws={"shrink": 0.8, "label": "Koefisien Korelasi Pearson"}, ax=ax)
    
    ax.set_title('Matriks Korelasi 16 Fitur Morfologi Biji Kacang (Pearson)', fontsize=16, fontweight='bold', pad=15)
    plt.tight_layout()
    return save_figure(fig, filename, output_dir)


def plot_pca_scree(evr, cum_evr, filename='04_pca_scree_variance.png', output_dir='figures'):
    """Plot scree plot and cumulative explained variance ratio."""
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    n_comp = len(evr)
    x = range(1, n_comp + 1)
    
    # Bar plot for individual variance
    bars = ax1.bar(x, evr * 100, color='#3498db', alpha=0.7, label='Variansi Individual (%)')
    ax1.set_xlabel('Principal Component (Komponen Utama)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Variansi Dijelaskan (%)', fontsize=12, fontweight='bold', color='#2980b9')
    ax1.set_xticks(list(x))
    ax1.tick_params(axis='y', labelcolor='#2980b9')
    
    # Cumulative line
    ax2 = ax1.twinx()
    line = ax2.plot(x, cum_evr * 100, color='#e74c3c', marker='o', linewidth=2.5, label='Variansi Kumulatif (%)')
    ax2.set_ylabel('Variansi Kumulatif (%)', fontsize=12, fontweight='bold', color='#c0392b')
    ax2.tick_params(axis='y', labelcolor='#c0392b')
    ax2.set_ylim(0, 105)
    
    # Threshold lines
    ax2.axhline(90, color='gray', linestyle=':', label='Ambang 90%')
    ax2.axhline(95, color='black', linestyle='--', label='Ambang 95%')
    
    # Annotate PC1 and PC2
    ax1.text(1, (evr[0]*100) + 1, f"{evr[0]*100:.1f}%", ha='center', fontsize=9, fontweight='bold')
    ax1.text(2, (evr[1]*100) + 1, f"{evr[1]*100:.1f}%", ha='center', fontsize=9, fontweight='bold')
    ax2.text(2, (cum_evr[1]*100) - 5, f"PC1+PC2 = {cum_evr[1]*100:.1f}%", color='#c0392b', ha='center', fontweight='bold')
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right', frameon=True)
    
    plt.title('Scree Plot & Variansi Kumulatif PCA pada 16 Fitur Terstandarisasi', fontsize=15, fontweight='bold')
    plt.tight_layout()
    return save_figure(fig, filename, output_dir)


def plot_kmeans_elbow(k_range, inertias, filename='05_kmeans_elbow_inertia.png', output_dir='figures'):
    """Plot K-Means inertia elbow curve and first derivative."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Inertia curve
    ax1.plot(k_range, inertias, marker='o', color='#2ecc71', linewidth=2.5, markersize=8)
    ax1.set_xlabel('Jumlah Cluster (k)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Inertia (Within-Cluster Sum of Squares)', fontsize=12, fontweight='bold')
    ax1.set_title('Elbow Plot: Inertia vs k (K-Means)', fontsize=14, fontweight='bold')
    ax1.set_xticks(list(k_range))
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Delta inertia (derivative)
    deltas = -np.diff(inertias)
    ax2.bar(list(k_range)[:-1], deltas, color='#34495e', alpha=0.75, width=0.6)
    ax2.set_xlabel('Transisi k -> k+1', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Penurunan Inertia (Delta)', fontsize=12, fontweight='bold')
    ax2.set_title('Tingkat Penurunan Inertia per Kenaikan k', fontsize=14, fontweight='bold')
    ax2.set_xticks(list(k_range)[:-1])
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.suptitle("Analisis Titik Elbow K-Means pada Data Terstandarisasi", fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    return save_figure(fig, filename, output_dir)


def plot_model_comparison_metrics(eval_df, filename='06_model_silhouette_comparison.png', output_dir='figures'):
    """
    Plot comparison of Silhouette, Calinski-Harabasz, and Davies-Bouldin across models and k.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    palette = {'K-Means': '#2980b9', 'Hierarchical (Ward)': '#e67e22', 'GMM (Full)': '#27ae60'}
    
    # Silhouette Score
    sns.lineplot(data=eval_df, x='k', y='silhouette', hue='model_name', marker='o',
                 linewidth=2.5, markersize=8, palette=palette, ax=axes[0])
    axes[0].set_title('Silhouette Score (Lebih Tinggi Lebih Baik)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Jumlah Cluster (k)', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Silhouette Score', fontsize=11, fontweight='bold')
    axes[0].set_xticks(eval_df['k'].unique())
    axes[0].grid(True, linestyle='--', alpha=0.6)
    
    # Calinski-Harabasz
    sns.lineplot(data=eval_df, x='k', y='calinski_harabasz', hue='model_name', marker='s',
                 linewidth=2.5, markersize=8, palette=palette, ax=axes[1])
    axes[1].set_title('Calinski-Harabasz Index (Lebih Tinggi Lebih Baik)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Jumlah Cluster (k)', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Calinski-Harabasz Score', fontsize=11, fontweight='bold')
    axes[1].set_xticks(eval_df['k'].unique())
    axes[1].grid(True, linestyle='--', alpha=0.6)
    
    # Davies-Bouldin
    sns.lineplot(data=eval_df, x='k', y='davies_bouldin', hue='model_name', marker='^',
                 linewidth=2.5, markersize=8, palette=palette, ax=axes[2])
    axes[2].set_title('Davies-Bouldin Index (Lebih Rendah Lebih Baik)', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Jumlah Cluster (k)', fontsize=11, fontweight='bold')
    axes[2].set_ylabel('Davies-Bouldin Score', fontsize=11, fontweight='bold')
    axes[2].set_xticks(eval_df['k'].unique())
    axes[2].grid(True, linestyle='--', alpha=0.6)
    
    # Clean up legends
    for ax in axes:
        ax.legend(title='Algoritma', frameon=True)
        
    plt.suptitle("Perbandingan Metrik Evaluasi Internal Antara K-Means, Hierarchical, dan GMM", fontsize=15, fontweight='bold', y=1.03)
    plt.tight_layout()
    return save_figure(fig, filename, output_dir)


def plot_pca_clusters(pca_2d_df, labels, title='Proyeksi 2D PCA Cluster Biji Kacang',
                      filename='07_pca_clusters_scatter.png', output_dir='figures'):
    """Plot 2D PCA scatter plot colored by cluster assignments."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    plot_df = pd.DataFrame({
        'PC1': pca_2d_df['PC1'],
        'PC2': pca_2d_df['PC2'],
        'Cluster': [f"Cluster {lbl}" if lbl != -1 else "Noise" for lbl in labels]
    })
    
    clusters_sorted = sorted(plot_df['Cluster'].unique())
    palette = sns.color_palette('tab10', n_colors=len(clusters_sorted))
    
    sns.scatterplot(data=plot_df, x='PC1', y='PC2', hue='Cluster', hue_order=clusters_sorted,
                    palette=palette, alpha=0.65, s=25, edgecolor=None, ax=ax)
    
    # Plot cluster centroids in PCA space
    for c in clusters_sorted:
        if c == 'Noise':
            continue
        c_mean = plot_df[plot_df['Cluster'] == c][['PC1', 'PC2']].mean()
        ax.scatter(c_mean['PC1'], c_mean['PC2'], s=150, c='black', marker='X', edgecolors='white', linewidths=1.5, zorder=10)
        ax.text(c_mean['PC1'], c_mean['PC2'] + 0.3, c, fontsize=11, fontweight='bold', ha='center',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='black'))
        
    ax.set_title(title, fontsize=15, fontweight='bold')
    ax.set_xlabel('Principal Component 1 (Ukuran & Skala Keseluruhan)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Principal Component 2 (Kebundaran vs Kelonjongan)', fontsize=12, fontweight='bold')
    ax.legend(title='Kelompok', bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
    
    plt.tight_layout()
    return save_figure(fig, filename, output_dir)


def plot_cluster_size_distribution(labels, filename='08_cluster_size_distribution.png', output_dir='figures'):
    """Plot bar chart of cluster sizes with counts and percentage share."""
    fig, ax = plt.subplots(figsize=(9, 5))
    
    unique_lbls, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    names = [f"Cluster {lbl}" if lbl != -1 else "Noise" for lbl in unique_lbls]
    pcts = (counts / total) * 100
    
    bars = ax.bar(names, counts, color=sns.color_palette('tab10', n_colors=len(names)), edgecolor='black', alpha=0.85)
    
    for bar, count, pct in zip(bars, counts, pcts):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, yval + (total * 0.01),
                f"{count:,}\n({pct:.1f}%)", ha='center', va='bottom', fontsize=10, fontweight='bold')
                
    ax.set_ylabel('Jumlah Observasi (Biji Kacang)', fontsize=12, fontweight='bold')
    ax.set_title('Distribusi Ukuran & Proporsi Setiap Cluster', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(counts) * 1.15)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    return save_figure(fig, filename, output_dir)


def plot_contingency_heatmap(y_true, labels, filename='09_contingency_class_vs_cluster_heatmap.png', output_dir='figures'):
    """Plot post-hoc cross-tabulation heatmap between Cluster and known Class varieties."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    df_temp = pd.DataFrame({'Cluster': [f"Cluster {l}" if l != -1 else "Noise" for l in labels], 'Class': y_true})
    
    # Raw counts
    counts = pd.crosstab(df_temp['Cluster'], df_temp['Class'])
    sns.heatmap(counts, annot=True, fmt='d', cmap='Blues', linewidths=0.5, cbar=True, ax=ax1)
    ax1.set_title('Jumlah Biji: Cluster vs Varietas Sebenarnya (Post-hoc)', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Varietas Biji (Ground Truth Class)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Cluster Hasil Pemodelan', fontsize=11, fontweight='bold')
    
    # Percentage composition per cluster
    pct = pd.crosstab(df_temp['Cluster'], df_temp['Class'], normalize='index') * 100
    sns.heatmap(pct, annot=True, fmt='.1f', cmap='YlGnBu', linewidths=0.5, cbar=True, ax=ax2)
    ax2.set_title('Komposisi Persentase Varietas per Cluster (%)', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Varietas Biji (Ground Truth Class)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('')
    
    plt.suptitle("Evaluasi Eksternal Post-Hoc: Kesesuaian Cluster dengan Varietas Biji Kacang", fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    return save_figure(fig, filename, output_dir)


def plot_cluster_morphological_profiles(df_raw, labels, filename='10_cluster_morphological_profiles.png', output_dir='figures'):
    """Plot standardized z-score profile heatmap of all features for each cluster."""
    features = [c for c in df_raw.columns if c != 'Class']
    df_temp = df_raw[features].copy()
    
    # Calculate z-scores relative to dataset overall mean/std
    df_z = (df_temp - df_temp.mean()) / df_temp.std()
    df_z['Cluster'] = [f"Cluster {l}" if l != -1 else "Noise" for l in labels]
    
    profile_z = df_z.groupby('Cluster').mean()
    
    fig, ax = plt.subplots(figsize=(16, 7))
    sns.heatmap(profile_z, cmap='coolwarm', center=0, annot=True, fmt='.2f',
                linewidths=0.8, cbar_kws={'label': 'Rata-rata Z-Score Morfologi'}, ax=ax)
    
    ax.set_title('Profil Morfologi Standar Setiap Cluster (Z-Score Relatif Terhadap Populasi)', fontsize=15, fontweight='bold', pad=15)
    ax.set_xlabel('Fitur Morfologi', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cluster Teridentifikasi', fontsize=12, fontweight='bold')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=11)
    
    plt.tight_layout()
    return save_figure(fig, filename, output_dir)


def plot_gmm_bic_aic(gmm_results, filename='11_gmm_bic_aic_selection.png', output_dir='figures'):
    """Plot BIC and AIC scores across components and covariance types."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    df_gmm = pd.DataFrame(gmm_results)
    
    # BIC
    sns.lineplot(data=df_gmm, x='k', y='bic', hue='covariance_type', marker='o',
                 linewidth=2.5, markersize=8, ax=ax1)
    ax1.set_title('Bayesian Information Criterion (BIC) - Lebih Rendah Lebih Baik', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Jumlah Komponen Campuran (k)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Skor BIC', fontsize=11, fontweight='bold')
    ax1.set_xticks(df_gmm['k'].unique())
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # AIC
    sns.lineplot(data=df_gmm, x='k', y='aic', hue='covariance_type', marker='s',
                 linewidth=2.5, markersize=8, ax=ax2)
    ax2.set_title('Akaike Information Criterion (AIC) - Lebih Rendah Lebih Baik', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Jumlah Komponen Campuran (k)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Skor AIC', fontsize=11, fontweight='bold')
    ax2.set_xticks(df_gmm['k'].unique())
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.suptitle("Seleksi Model Gaussian Mixture Model (GMM) Berdasarkan Kriteria Informasi", fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    return save_figure(fig, filename, output_dir)


def plot_dbscan_k_distance(k_distances, knee_point=1.5, filename='12_dbscan_k_distance_elbow.png', output_dir='figures'):
    """Plot sorted k-nearest neighbor distances to detect density threshold (eps) for DBSCAN."""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    x = range(len(k_distances))
    ax.plot(x, k_distances, color='#8e44ad', linewidth=2, label='Jarak k-Tetangga Terdekat (k=32)')
    
    if knee_point is not None:
        ax.axhline(knee_point, color='red', linestyle='--', label=f'Titik Potong Indikatif (eps ≈ {knee_point})')
        
    ax.set_title('Kurva Urutan Jarak k-Nearest Neighbors (k-Distance Plot untuk DBSCAN)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Indeks Titik Biji Kacang (Diurutkan Berdasarkan Jarak)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Jarak ke Tetangga ke-32 (Euclidean)', fontsize=11, fontweight='bold')
    ax.legend(frameon=True)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    return save_figure(fig, filename, output_dir)


def plot_pca_true_class(pca_2d_df, y_true, filename='13_pca_true_class_scatter.png', output_dir='figures'):
    """Plot 2D PCA scatter plot colored by known Ground Truth Class (for post-hoc comparison)."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    plot_df = pd.DataFrame({
        'PC1': pca_2d_df['PC1'],
        'PC2': pca_2d_df['PC2'],
        'Class': y_true
    })
    
    classes_sorted = sorted(plot_df['Class'].unique())
    palette = sns.color_palette('tab10', n_colors=len(classes_sorted))
    
    sns.scatterplot(data=plot_df, x='PC1', y='PC2', hue='Class', hue_order=classes_sorted,
                    palette=palette, alpha=0.65, s=25, edgecolor=None, ax=ax)
    
    # Plot class centroids in PCA space
    for c in classes_sorted:
        c_mean = plot_df[plot_df['Class'] == c][['PC1', 'PC2']].mean()
        ax.scatter(c_mean['PC1'], c_mean['PC2'], s=150, c='black', marker='o', edgecolors='white', linewidths=1.5, zorder=10)
        ax.text(c_mean['PC1'], c_mean['PC2'] + 0.3, c, fontsize=11, fontweight='bold', ha='center',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='black'))
        
    ax.set_title('Sebaran Varietas Biji Kacang Asli pada Ruang 2D PCA (Post-Hoc Benchmark)', fontsize=15, fontweight='bold')
    ax.set_xlabel('Principal Component 1 (Ukuran & Skala Keseluruhan)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Principal Component 2 (Kebundaran vs Kelonjongan)', fontsize=12, fontweight='bold')
    ax.legend(title='Varietas Asli', bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
    
    plt.tight_layout()
    return save_figure(fig, filename, output_dir)


def plot_key_morphology_boxplots(df_raw, labels, filename='14_cluster_morphology_boxplots.png', output_dir='figures'):
    """Plot boxplots of key distinguishing morphological features across clusters."""
    key_features = ['Area', 'MajorAxisLength', 'MinorAxisLength', 'AspectRatio', 'Solidity', 'Compactness']
    df_temp = df_raw[key_features].copy()
    df_temp['Cluster'] = [f"C{l}" if l != -1 else "Noise" for l in labels]
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()
    
    for i, feat in enumerate(key_features):
        sns.boxplot(data=df_temp, x='Cluster', y=feat, hue='Cluster', legend=False, ax=axes[i], palette='Set2')
        axes[i].set_title(f'Distribusi {feat} per Cluster', fontsize=12, fontweight='bold')
        axes[i].set_xlabel('')
        axes[i].grid(True, linestyle='--', alpha=0.6)
        
    plt.suptitle("Pemeriksaan Perbedaan Fitur Morfologi Kunci Antar-Cluster", fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    return save_figure(fig, filename, output_dir)
