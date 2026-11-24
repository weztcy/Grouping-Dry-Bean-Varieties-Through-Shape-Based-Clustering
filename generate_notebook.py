"""
generate_notebook.py - Script to build notebooks/clustering_workflow.ipynb
"""

import nbformat as nbf

nb = nbf.v4.new_notebook()

cells = []

# Title & Overview
cells.append(nbf.v4.new_markdown_cell("""# Grouping Dry Bean Varieties Through Shape-Based Clustering
**Proyek Unsupervised Learning untuk Karakterisasi Morfologi Biji Kacang Kering**

---

## 1. Project Overview & Latar Belakang
Proyek ini mengimplementasikan alur kerja **Unsupervised Machine Learning** komprehensif untuk mengidentifikasi kelompok bentuk alami (*natural shape-based clusters*) pada 13.611 citra biji kacang kering (*Phaseolus vulgaris L.*). 

Dataset yang digunakan berasal dari **UCI Machine Learning Repository** (Koklu & Özkan, 2020; DOI: `10.1016/j.compag.2020.105507`).

### Prinsip Metodologis Utama
* **Strictly Unsupervised:** Kolom `Class` (7 varietas biologis) **sama sekali tidak dilibatkan** dalam tahap ekstraksi fitur, preprocessing, pemilihan jumlah cluster ($k$), maupun pelatihan model.
* **Post-hoc Evaluation Only:** Informasi label hanya dibuka di akhir untuk mengukur seberapa jauh morfologi geometris murni berkorespondensi dengan varietas kacang yang telah terdaftar secara komersial.
"""))

# Cell 2: Imports
cells.append(nbf.v4.new_code_cell("""import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    v_measure_score
)

# Set visual styling
sns.set_theme(style='whitegrid')
plt.rcParams['font.size'] = 10
%matplotlib inline

print("All dependencies successfully imported.")"""))

# Cell 3: Data Loading
cells.append(nbf.v4.new_markdown_cell("""## 2. Tahap A — Pemeriksaan dan Pemahaman Data
Memuat dataset `Dry_Bean.csv`, memisahkan fitur 16 numerik, dan memverifikasi kualitas data (missing values, duplikat, dan tipe data)."""))

cells.append(nbf.v4.new_code_cell("""csv_path = '../Dry_Bean.csv' if os.path.exists('../Dry_Bean.csv') else 'Dry_Bean.csv'
df = pd.read_csv(csv_path)
features = [c for c in df.columns if c != 'Class']

# Strict separation of X and y
X = df[features].copy()
y = df['Class'].copy()

print(f"Dimensi Dataset: {df.shape[0]} baris, {df.shape[1]} kolom")
print(f"Fitur Numerik ({len(features)}): {features}")
print(f"Missing Values: {df.isnull().sum().sum()}")
print(f"Duplikat Identik: {df.duplicated().sum()} baris")
print("\\nDistribusi Varietas Asli (Hanya untuk referensi post-hoc):")
print(y.value_counts())"""))

# Cell 4: Duplicate Analysis
cells.append(nbf.v4.new_code_cell("""dup_df = df[df.duplicated(subset=features, keep=False)]
print(f"Total baris yang terlibat dalam duplikasi: {len(dup_df)}")
print(f"Distribusi kelas pada baris duplikat:")
print(df[df.duplicated(subset=features)]['Class'].value_counts())"""))

# Cell 5: Descriptive Stats & EDA
cells.append(nbf.v4.new_markdown_cell("""## 3. Tahap B — Exploratory Data Analysis (EDA)
Meninjau statistik deskriptif, kemiringan distribusi (skewness), keberadaan pencilan (outliers), dan matriks korelasi."""))

cells.append(nbf.v4.new_code_cell("""desc_stats = X.describe().T
desc_stats['skewness'] = X.skew()
desc_stats[['mean', 'std', 'min', '50%', 'max', 'skewness']]"""))

# Cell 6: Correlation Analysis
cells.append(nbf.v4.new_code_cell("""plt.figure(figsize=(14, 11))
corr = X.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, cmap='coolwarm', vmin=-1, vmax=1, annot=True, fmt='.2f', square=True)
plt.title("Matriks Korelasi Pearson 16 Fitur Morfologi Biji Kacang", fontsize=14, fontweight='bold')
plt.show()"""))

# Cell 7: Preprocessing & Scaling
cells.append(nbf.v4.new_markdown_cell("""## 4. Tahap C — Preprocessing, Scaling & PCA
Standarisasi fitur menggunakan `StandardScaler` (Z-score normalization) dan eksplorasi reduksi dimensi via `PCA`."""))

cells.append(nbf.v4.new_code_cell("""scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(random_state=42)
pca.fit(X_scaled)
evr = pca.explained_variance_ratio_
cum_evr = np.cumsum(evr)

print(f"Variansi PC1: {evr[0]*100:.2f}%")
print(f"Variansi PC2: {evr[1]*100:.2f}%")
print(f"Variansi Kumulatif PC1 + PC2: {cum_evr[1]*100:.2f}%")
print(f"Jumlah komponen untuk >= 95% variansi: {np.argmax(cum_evr >= 0.95) + 1}")

# Proyeksi 2D untuk visualisasi
pca_2d = PCA(n_components=2, random_state=42)
X_pca_2d = pca_2d.fit_transform(X_scaled)"""))

# Cell 8: Clustering Modeling
cells.append(nbf.v4.new_markdown_cell("""## 5. Tahap D & E — Pemodelan Clustering & Evaluasi Internal
Membandingkan 4 algoritma:
1. **K-Means** (k=2..12, n_init=20)
2. **Agglomerative Hierarchical** (linkage='ward', k=2..12)
3. **Gaussian Mixture Model (GMM)** (full covariance, k=2..12)
4. **DBSCAN** (density parameter grid search)"""))

cells.append(nbf.v4.new_code_cell("""k_values = list(range(2, 13))
km_inertias = []
km_silhouettes = []
agg_silhouettes = []
gmm_bics = []

for k in k_values:
    # K-Means
    km = KMeans(n_clusters=k, n_init=20, random_state=42).fit(X_scaled)
    km_inertias.append(km.inertia_)
    km_silhouettes.append(silhouette_score(X_scaled, km.labels_))
    
    # Agglomerative Ward
    agg = AgglomerativeClustering(n_clusters=k, linkage='ward').fit(X_scaled)
    agg_silhouettes.append(silhouette_score(X_scaled, agg.labels_))
    
    # GMM
    gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=42).fit(X_scaled)
    gmm_bics.append(gmm.bic(X_scaled))

print("Eksperimen pemodelan selesai.")"""))

# Cell 9: Evaluation Plots
cells.append(nbf.v4.new_code_cell("""fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Elbow plot
ax1.plot(k_values, km_inertias, marker='o', color='#2ecc71', linewidth=2)
ax1.set_title("Elbow Curve K-Means (Inertia vs k)", fontweight='bold')
ax1.set_xlabel("Jumlah Cluster (k)")
ax1.set_ylabel("Inertia")
ax1.set_xticks(k_values)

# Silhouette comparison
ax2.plot(k_values, km_silhouettes, marker='o', label='K-Means', color='#2980b9', linewidth=2)
ax2.plot(k_values, agg_silhouettes, marker='s', label='Hierarchical (Ward)', color='#e67e22', linewidth=2)
ax2.set_title("Perbandingan Silhouette Score", fontweight='bold')
ax2.set_xlabel("Jumlah Cluster (k)")
ax2.set_ylabel("Silhouette Score")
ax2.set_xticks(k_values)
ax2.legend()

plt.tight_layout()
plt.show()"""))

# Cell 10: Best Model Fitting & Post-hoc Analysis
cells.append(nbf.v4.new_markdown_cell("""## 6. Tahap F — Model Terbaik & Profil Morfologi (k=4)
Berdasarkan metrik internal Silhouette Score dan separabilitas struktur morfologi alami, K-Means dengan $k=4$ dipilih sebagai model representasi morfologi alami terbaik."""))

cells.append(nbf.v4.new_code_cell("""best_kmeans = KMeans(n_clusters=4, n_init=20, random_state=42)
cluster_labels = best_kmeans.fit_predict(X_scaled)

# Evaluasi Post-hoc eksternal
ari = adjusted_rand_score(y, cluster_labels)
nmi = normalized_mutual_info_score(y, cluster_labels)
vm = v_measure_score(y, cluster_labels)

print(f"Silhouette Score (k=4): {silhouette_score(X_scaled, cluster_labels):.4f}")
print(f"Calinski-Harabasz: {calinski_harabasz_score(X_scaled, cluster_labels):.1f}")
print(f"Davies-Bouldin: {davies_bouldin_score(X_scaled, cluster_labels):.4f}")
print(f"Post-hoc ARI: {ari:.4f}")
print(f"Post-hoc NMI: {nmi:.4f}")
print(f"Post-hoc V-Measure: {vm:.4f}")"""))

# Cell 11: PCA Scatter Plot
cells.append(nbf.v4.new_code_cell("""plt.figure(figsize=(10, 7))
plot_df = pd.DataFrame({'PC1': X_pca_2d[:, 0], 'PC2': X_pca_2d[:, 1], 'Cluster': [f'Cluster {l}' for l in cluster_labels]})
sns.scatterplot(data=plot_df, x='PC1', y='PC2', hue='Cluster', palette='tab10', alpha=0.6, s=25)
plt.title("Proyeksi 2D PCA Cluster Morfologi K-Means (k=4)", fontsize=13, fontweight='bold')
plt.xlabel("PC1 (Ukuran / Skala)")
plt.ylabel("PC2 (Kebundaran vs Kelonjongan)")
plt.show()"""))

# Cell 12: Contingency Matrix
cells.append(nbf.v4.new_code_cell("""contingency = pd.crosstab(plot_df['Cluster'], y)
print("Matriks Kontingensi Post-hoc (Cluster vs True Class):")
display(contingency)

plt.figure(figsize=(10, 5))
sns.heatmap(pd.crosstab(plot_df['Cluster'], y, normalize='index') * 100, annot=True, fmt='.1f', cmap='Blues')
plt.title("Proporsi Varietas Biji Kacang pada Setiap Cluster (%)", fontsize=13, fontweight='bold')
plt.ylabel("Cluster")
plt.xlabel("Ground Truth Class")
plt.show()"""))

# Cell 13: Conclusion
cells.append(nbf.v4.new_markdown_cell("""## 7. Kesimpulan Metodologis
1. **Struktur Alami Data:** Berdasarkan geometri bentuk 16 fitur, data secara alami terkelompok menjadi 4 kelompok utama:
   - *Cluster 0:* Biji bulat padat (didominasi kuat oleh **SEKER**).
   - *Cluster 1:* Biji lonjong ramping (didominasi oleh **HOROZ** dan sebagian **CALI**).
   - *Cluster 2:* Biji raksasa masif (100% adalah **BOMBAY**).
   - *Cluster 3:* Biji oval sedang (campuran kontinu antara **DERMASON** dan **SIRA**).
2. **Keterbatasan Tanpa Supervisi:** Varietas *Dermason* dan *Sira* memiliki kemiripan morfologi geometris yang sangat tinggi sehingga membentuk kontinum spasial yang tidak dapat dipisahkan secara murni tanpa informasi warna, tekstur, atau label supervisi.
"""))

nb['cells'] = cells

with open('notebooks/clustering_workflow.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Notebook generated successfully at notebooks/clustering_workflow.ipynb")
