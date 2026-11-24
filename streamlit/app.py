"""
app.py - Interactive Streamlit Dashboard for Dry Bean Morphological Clustering.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(
    page_title="Dry Bean Clustering Dashboard",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1e3d59;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #555555;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        border-left: 5px solid #17b978;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        white-space: pre-wrap;
        background-color: #f1f3f5;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1e3d59 !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_cached_data():
    csv_path = 'Dry_Bean.csv' if os.path.exists('Dry_Bean.csv') else 'dataset/Dry_Bean.csv'
    df = pd.read_csv(csv_path)
    features = [c for c in df.columns if c != 'Class']
    X = df[features]
    y = df['Class']
    return df, X, y, features


@st.cache_resource
def load_cached_artifacts():
    scaler = joblib.load('models/scaler.joblib') if os.path.exists('models/scaler.joblib') else None
    model_k4 = joblib.load('models/best_clustering_model.joblib') if os.path.exists('models/best_clustering_model.joblib') else None
    model_k7 = joblib.load('models/kmeans_k7_model.joblib') if os.path.exists('models/kmeans_k7_model.joblib') else None
    pca_model = joblib.load('models/pca_model.joblib') if os.path.exists('models/pca_model.joblib') else None
    
    summary = {}
    if os.path.exists('models/pipeline_summary.json'):
        with open('models/pipeline_summary.json', 'r') as f:
            summary = json.load(f)
            
    return scaler, model_k4, model_k7, pca_model, summary


def main():
    st.markdown('<div class="main-title">🌱 Pengelompokan Varietas Biji Kacang Kering (Dry Bean)</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Analisis Struktur Kelompok Alami Morfologi Berbasis Unsupervised Learning</div>', unsafe_allow_html=True)
    
    df, X, y, features = load_cached_data()
    scaler, model_k4, model_k7, pca_model, summary = load_cached_artifacts()
    
    # Precompute scaled and PCA if models exist
    if scaler is not None and pca_model is not None:
        X_scaled = scaler.transform(X)
        pca_coords = pca_model.transform(X_scaled)
        pca_df = pd.DataFrame(pca_coords, columns=['PC1', 'PC2'])
        if model_k4 is not None:
            labels_k4 = model_k4.labels_
        else:
            labels_k4 = np.zeros(len(df))
    else:
        labels_k4 = np.zeros(len(df))
        pca_df = pd.DataFrame(np.zeros((len(df), 2)), columns=['PC1', 'PC2'])

    # Sidebar
    st.sidebar.image("https://img.icons8.com/color/96/000000/soy.png", width=80)
    st.sidebar.title("Navigasi Kontrol")
    st.sidebar.markdown("""
    **Dataset:** UCI Dry Bean Dataset  
    **Observasi:** 13.611 baris  
    **Fitur Morfologi:** 16 numerik  
    **Varietas:** 7 kelas (Post-hoc)
    """)
    st.sidebar.divider()
    
    selected_view = st.sidebar.radio(
        "Pilih Modul Dashboard:",
        [
            "1. Ringkasan & Arsitektur Project",
            "2. Eksplorasi Data & Korelasi (EDA)",
            "3. Peta Kluster & Proyeksi PCA",
            "4. Profil Morfologi Setiap Cluster",
            "5. Evaluasi Eksternal (Post-Hoc Class)",
            "6. Prediksi & Simulasi Biji Baru"
        ]
    )
    
    # -------------------------------------------------------------
    # TAB 1: RINGKASAN & ARSITEKTUR
    # -------------------------------------------------------------
    if "1. Ringkasan" in selected_view:
        st.header("📋 Ringkasan Eksekutif & Konteks Penelitian")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Observasi", f"{len(df):,}")
        with col2:
            st.metric("Fitur Morfologi", f"{len(features)}")
        with col3:
            st.metric("Model Utama Terpilih", "K-Means (k=4)")
        with col4:
            st.metric("Silhouette Score (k=4)", "0.3533")
            
        st.markdown("""
        ### Latar Belakang & Filosofi Unsupervised Learning
        Proyek ini bertujuan menemukan **kelompok alami (natural clustering)** pada biji kacang kering (*Phaseolus vulgaris L.*) 
        hanya berdasarkan pengukuran geometris dan morfologi citra komputer, **tanpa supervisi label**.
        
        > **Aturan Metodologis Utama:**
        > Kolom `Class` (7 varietas terdaftar: *SEKER, BARBUNYA, BOMBAY, CALI, DERMASON, HOROZ, SIRA*) sama sekali **tidak pernah digunakan** 
        > dalam pemilihan fitur, scaling, atau pembentukan cluster. Evaluasi kesesuaian biologis hanya dilakukan secara **post-hoc** 
        > setelah pemodelan selesai sepenuhnya.
        """)
        
        st.subheader("Karakteristik 16 Fitur Morfologi")
        col_dim, col_shape = st.columns(2)
        with col_dim:
            st.markdown(r"""
            **Fitur Dimensi & Ukuran:**
            - `Area` ($A$): Luas piksel area biji.
            - `Perimeter` ($P$): Panjang keliling kontur biji.
            - `MajorAxisLength` ($L$): Panjang garis terpanjang (sumbu mayor).
            - `MinorAxisLength` ($l$): Lebar biji tegak lurus sumbu mayor.
            - `ConvexArea` ($C$): Luas poligon cembung terkecil yang memuat biji.
            - `EquivDiameter` ($E_d$): Diameter lingkaran berluas setara ($2\sqrt{A/\pi}$).
            """)
        with col_shape:
            st.markdown(r"""
            **Fitur Bentuk & Rasio Morfologi:**
            - `AspectRatio` ($K$): Rasio panjang/lebar ($L/l$).
            - `Eccentricity` ($Ec$): Eksentrisitas elips ekuivalen.
            - `Extent` ($Ex$): Rasio luas biji terhadap *bounding box*.
            - `Solidity` ($S$): Rasio luas biji terhadap *convex area* ($A/C$).
            - `Roundness` ($R$): Kebundaran kontur ($4\pi A / P^2$).
            - `Compactness` ($CO$): Kepadatan struktur ($E_d / L$).
            - `ShapeFactor1 - 4`: Faktor bentuk geometris non-linear.
            """)
            
        st.divider()
        st.subheader("Arsitektur Hasil Pipeline Unsupervised Learning")
        st.image("figures/06_model_silhouette_comparison.png", caption="Perbandingan Metrik Evaluasi Internal (K-Means, Hierarchical Ward, GMM)")

    # -------------------------------------------------------------
    # TAB 2: EXPLORATORY DATA ANALYSIS (EDA)
    # -------------------------------------------------------------
    elif "2. Eksplorasi Data" in selected_view:
        st.header("🔍 Exploratory Data Analysis & Analisis Korelasi")
        
        eda_tab1, eda_tab2, eda_tab3 = st.tabs(["Distribusi & Outlier", "Matriks Korelasi & Multikolinearitas", "Scatter Hubungan Fitur"])
        
        with eda_tab1:
            st.subheader("Distribusi 16 Fitur Morfologi")
            st.image("figures/01_feature_distribution_hist_kde.png", caption="Histogram & KDE 16 Fitur Morfologi")
            st.divider()
            st.subheader("Pemeriksaan Outlier Fitur Terstandarisasi")
            st.image("figures/02_feature_boxplots_outliers.png", caption="Boxplot Terstandarisasi (Z-Score)")
            st.info("Catatan: Nilai ekstrem pada `Area` dan `ConvexArea` didorong oleh varietas raksasa (BOMBAY) yang ukurannya 3-4x lebih besar dari varietas normal.")
            
        with eda_tab2:
            st.subheader("Matriks Korelasi Pearson 16 Fitur")
            st.image("figures/03_correlation_heatmap.png", caption="Heatmap Korelasi Pearson")
            
            st.markdown(r"""
            **Temuan Multikolinearitas Sangat Tinggi:**
            - `Area` $\leftrightarrow$ `ConvexArea`: $r = 0.9999$ (hampir identik sempurna)
            - `Compactness` $\leftrightarrow$ `ShapeFactor3`: $r = 0.9987$
            - `Perimeter` $\leftrightarrow$ `EquivDiameter`: $r = 0.9914$
            - `AspectRatio` $\leftrightarrow$ `Compactness`: $r = -0.9877$ (kebundaran vs kelonjongan)
            """)
            
        with eda_tab3:
            st.subheader("Eksplorasi Scatterplot Interaktif")
            c1, c2 = st.columns(2)
            with c1:
                feat_x = st.selectbox("Pilih Sumbu X:", features, index=0)
            with c2:
                feat_y = st.selectbox("Pilih Sumbu Y:", features, index=4)
                
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(data=df.sample(2500, random_state=42), x=feat_x, y=feat_y, alpha=0.5, color='#2980b9', s=30, ax=ax)
            ax.set_title(f"Hubungan antara {feat_x} dan {feat_y}", fontsize=13, fontweight='bold')
            st.pyplot(fig)

    # -------------------------------------------------------------
    # TAB 3: PETA KLUSTER & PROYEKSI PCA
    # -------------------------------------------------------------
    elif "3. Peta Kluster" in selected_view:
        st.header("🗺️ Visualisasi Ruang PCA & Pembentukan Cluster")
        
        c1, c2 = st.columns([2, 1])
        with c1:
            st.image("figures/07_pca_clusters_scatter.png", caption="Proyeksi 2D PCA Cluster Terpilih (K-Means k=4)")
        with c2:
            st.image("figures/08_cluster_size_distribution.png", caption="Distribusi Ukuran Cluster")
            st.markdown("""
            **Interpretasi Ruang PCA:**
            - **PC1 (55.47% variansi):** Memetakan **Ukuran/Skala** (*Area, Perimeter, Length*).
            - **PC2 (26.43% variansi):** Memetakan **Kebundaran vs Kelonjongan** (*Compactness* vs *AspectRatio*).
            - Total Variansi Terjelaskan PC1 + PC2 = **81.90%**.
            """)
            
        st.divider()
        st.subheader("Perbandingan Model: K-Means k=4 (Struktur Alami) vs k=7 (Kompabilitas Varietas)")
        col_a, col_b = st.columns(2)
        with col_a:
            st.image("figures/07_pca_clusters_scatter.png", caption="Model Alami K-Means (k=4) - Silhouette 0.3533")
        with col_b:
            st.image("figures/15_pca_kmeans_k7_clusters.png", caption="Model Rinci K-Means (k=7) - Silhouette 0.3347")

    # -------------------------------------------------------------
    # TAB 4: PROFIL MORFOLOGI SETIAP CLUSTER
    # -------------------------------------------------------------
    elif "4. Profil Morfologi" in selected_view:
        st.header("📊 Profil Morfologi & Karakteristik Geometris Cluster")
        
        st.image("figures/10_cluster_morphological_profiles.png", caption="Heatmap Profil Z-Score Fitur per Cluster")
        st.divider()
        st.image("figures/14_cluster_morphology_boxplots.png", caption="Sebaran Fitur Morfologi Kunci Antar-Cluster")
        
        st.markdown("""
        ### Deskripsi Biologis & Fisik Cluster Alami (k=4):
        1. **Cluster 0 (Biji Bundar & Padat):**
           - *Karakteristik:* Area kecil-sedang, `Roundness` sangat tinggi, `Compactness` tinggi, `AspectRatio` rendah (~1.2 - 1.3).
           - *Korespondensi Biologis:* Sangat didominasi varietas **SEKER** (97.9% kemurnian).
        2. **Cluster 1 (Biji Lonjong / Slender Sedang):**
           - *Karakteristik:* `AspectRatio` tinggi (~1.9 - 2.2), `MajorAxisLength` panjang, `MinorAxisLength` sempit, `Eccentricity` tinggi.
           - *Korespondensi Biologis:* Sangat didominasi varietas **HOROZ** dan sebagian **CALI**.
        3. **Cluster 2 (Biji Raksasa / Giant Beans):**
           - *Karakteristik:* `Area` ekstrem (>150.000 px), `MajorAxisLength` dan `Perimeter` sangat masif, terisolasi jauh di kuadran kanan PC1.
           - *Korespondensi Biologis:* 100% adalah varietas **BOMBAY**.
        4. **Cluster 3 (Biji Oval Standar / Medium Oval):**
           - *Karakteristik:* Area sedang (~40.000 - 55.000 px), bentuk oval moderat.
           - *Korespondensi Biologis:* Campuran alami antara **DERMASON** dan **SIRA** yang memiliki morfologi tumpang tindih.
        """)

    # -------------------------------------------------------------
    # TAB 5: EVALUASI EKSTERNAL (POST-HOC CLASS)
    # -------------------------------------------------------------
    elif "5. Evaluasi Eksternal" in selected_view:
        st.header("🎯 Validasi Eksternal Post-Hoc terhadap Varietas Biologis")
        
        st.warning("⚠️ **Pengingat Metodologis:** Informasi label varietas ini TIDAK PERNAH digunakan dalam proses clustering. Evaluasi ini dilakukan semata-mata untuk menguji apakah kluster alami yang terbentuk mencerminkan pemisahan varietas nyata.")
        
        col1, col2 = st.columns(2)
        with col1:
            st.image("figures/09_contingency_class_vs_cluster_heatmap.png", caption="Matriks Kontingensi Cluster (k=4) vs True Class")
        with col2:
            st.image("figures/13_pca_true_class_scatter.png", caption="Sebaran Varietas Asli pada Ruang 2D PCA")
            
        st.divider()
        st.subheader("Matriks Evaluasi Eksternal Post-Hoc")
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1:
            st.metric("Adjusted Rand Index (ARI)", "0.5015")
        with m_col2:
            st.metric("Normalized Mutual Info (NMI)", "0.5898")
        with m_col3:
            st.metric("V-Measure", "0.5898")
        with m_col4:
            st.metric("Purity Kluster", "66.52%")
            
        st.markdown("""
        **Analisis Tumpang Tindih Morfologis:**
        - Varietas dengan morfologi unik (*Bombay* = raksasa, *Seker* = bulat bola, *Horoz* = sangat lonjong) terpisah dengan sangat akurat secara murni tanpa bantuan label.
        - Varietas *Dermason* dan *Sira* secara morfologis sangat mirip (keduanya biji putih oval berukuran sedang-kecil), sehingga algoritma tanpa label menggabungkannya ke dalam satu spektrum morfologi kontinu. Ini bukan kegagalan algoritma, melainkan fakta biologis bahwa perbedaan keduanya lebih pada warna/tekstur daripada geometri luar.
        """)

    # -------------------------------------------------------------
    # TAB 6: PREDIKSI & SIMULASI BIJI BARU
    # -------------------------------------------------------------
    elif "6. Prediksi & Simulasi" in selected_view:
        st.header("🔬 Simulasi & Prediksi Kluster Biji Kacang Baru")
        st.markdown("Masukkan nilai pengukuran morfologi biji kacang untuk mengidentifikasi kelompok bentuk alami dan varietas yang paling mirip:")
        
        # Preset buttons for ease of testing
        st.subheader("Atau Pilih Contoh Varietas Representatif:")
        preset_cols = st.columns(7)
        preset_choice = None
        for i, v_name in enumerate(['SEKER', 'BARBUNYA', 'BOMBAY', 'CALI', 'DERMASON', 'HOROZ', 'SIRA']):
            if preset_cols[i].button(v_name):
                sample_row = df[df['Class'] == v_name].iloc[0]
                preset_choice = sample_row
                
        # Form inputs
        with st.form("bean_form"):
            col_a, col_b, col_c, col_d = st.columns(4)
            
            def_val = lambda col, d: float(preset_choice[col]) if preset_choice is not None else float(d)
            
            with col_a:
                in_area = st.number_input("Area (piksel)", value=def_val('Area', 45000.0), step=1000.0)
                in_perim = st.number_input("Perimeter", value=def_val('Perimeter', 800.0), step=10.0)
                in_major = st.number_input("MajorAxisLength", value=def_val('MajorAxisLength', 300.0), step=5.0)
                in_minor = st.number_input("MinorAxisLength", value=def_val('MinorAxisLength', 190.0), step=5.0)
            with col_b:
                in_aspect = st.number_input("AspectRatio", value=def_val('AspectRatio', 1.55), step=0.05)
                in_ecc = st.number_input("Eccentricity", value=def_val('Eccentricity', 0.76), step=0.01)
                in_carea = st.number_input("ConvexArea", value=def_val('ConvexArea', 45500.0), step=1000.0)
                in_eqdiam = st.number_input("EquivDiameter", value=def_val('EquivDiameter', 239.0), step=5.0)
            with col_c:
                in_extent = st.number_input("Extent", value=def_val('Extent', 0.75), step=0.01)
                in_solidity = st.number_input("Solidity", value=def_val('Solidity', 0.988), step=0.001)
                in_roundness = st.number_input("Roundness", value=def_val('Roundness', 0.88), step=0.01)
                in_compactness = st.number_input("Compactness", value=def_val('Compactness', 0.80), step=0.01)
            with col_d:
                in_sf1 = st.number_input("ShapeFactor1", value=def_val('ShapeFactor1', 0.0066), format="%.6f")
                in_sf2 = st.number_input("ShapeFactor2", value=def_val('ShapeFactor2', 0.0017), format="%.6f")
                in_sf3 = st.number_input("ShapeFactor3", value=def_val('ShapeFactor3', 0.64), step=0.01)
                in_sf4 = st.number_input("ShapeFactor4", value=def_val('ShapeFactor4', 0.996), format="%.4f")
                
            submit_btn = st.form_submit_button("Analisis Morfologi & Prediksi Kluster", use_container_width=True)
            
        if submit_btn and scaler is not None and model_k4 is not None and pca_model is not None:
            input_vector = np.array([[
                in_area, in_perim, in_major, in_minor, in_aspect, in_ecc, in_carea, in_eqdiam,
                in_extent, in_solidity, in_roundness, in_compactness, in_sf1, in_sf2, in_sf3, in_sf4
            ]])
            
            input_scaled = scaler.transform(input_vector)
            cluster_pred = model_k4.predict(input_scaled)[0]
            input_pca = pca_model.transform(input_scaled)[0]
            
            cluster_descriptions = {
                0: ("Cluster 0: Biji Bundar / Bulat (Spherical)", "Sangat mirip morfologi **SEKER** (tinggi roundness & compactness).", "#1f77b4"),
                1: ("Cluster 1: Biji Lonjong Ramping (Elongated/Slender)", "Sangat mirip morfologi **HOROZ** atau **CALI** (tinggi aspect ratio & length).", "#ff7f0e"),
                2: ("Cluster 2: Biji Raksasa / Masif (Giant Beans)", "Identik dengan morfologi **BOMBAY** (area masif > 150k px).", "#2ca02c"),
                3: ("Cluster 3: Biji Oval Sedang (Medium Oval)", "Mirip morfologi **DERMASON** atau **SIRA** (ukuran sedang, bentuk simetris oval).", "#d62728")
            }
            
            title_desc, bio_desc, clr = cluster_descriptions.get(cluster_pred, (f"Cluster {cluster_pred}", "", "#333333"))
            
            st.success(f"### Hasil Prediksi: {title_desc}")
            st.markdown(f"**Indikasi Kesesuaian Biologis:** {bio_desc}")
            
            # Show position on PCA map
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.scatterplot(data=pca_df.sample(2000, random_state=42), x='PC1', y='PC2',
                            hue=[f"Cluster {l}" for l in labels_k4[:2000]], alpha=0.3, palette='tab10', ax=ax)
            ax.scatter(input_pca[0], input_pca[1], color='red', marker='*', s=350, edgecolors='black', linewidth=2, label='Biji Input')
            ax.set_title("Posisi Biji Input pada Ruang Morfologi PCA", fontsize=12, fontweight='bold')
            ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
            st.pyplot(fig)


if __name__ == '__main__':
    main()
