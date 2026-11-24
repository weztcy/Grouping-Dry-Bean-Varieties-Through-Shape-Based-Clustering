# Portfolio Project Data — Grouping Dry Bean Varieties Through Shape-Based Clustering

- **Category:** `machine-learning` (`Machine Learning`)
- **Accent Color:** `indigo`
- **Target Assets Folder:** `public/images/projects/grouping_dry_bean_varieties/`
- **Required Local Images:** `banner.webp`, `image01.webp` .. `image18.webp`

## TypeScript Data (`ProjectCaseStudy`)

```typescript
{
  "id": "grouping-dry-bean-varieties-through-shape-based-clustering",
  "code": "PRO-27 // ML-12",
  "title": "Grouping Dry Bean Varieties Through Shape-Based Clustering",
  "category": "machine-learning",
  "categoryLabel": "Unsupervised Learning",
  "year": "2024",
  "status": "Completed",
  "accentColor": "indigo",
  "repoUrl": "https://github.com/weztcy/Grouping-Dry-Bean-Varieties-Through-Shape-Based-Clustering",
  "license": "MIT License",
  "banner": "/images/projects/grouping_dry_bean_varieties/banner.webp",
  "summary": "Strictly unsupervised shape-based clustering of 13,611 dry bean specimens across 16 geometric features using K-Means, Ward, GMM, and DBSCAN.",
  "projectDescription": "This project discovers natural morphological groups across 13,611 dry bean specimens from 16 digital image geometric features without label supervision. Four clustering paradigms are benchmarked via internal metrics, bootstrap stability, and post-hoc biological validation.",
  "architectureSummary": "Dry_Bean.csv (16 Features, Class Held Out) -> StandardScaler & PCA -> Multi-Algorithm Clustering (K-Means, Ward, GMM, DBSCAN) -> Internal Selection (k=4) & Post-Hoc Validation -> Streamlit Dashboard",
  "techStack": [
    "Python",
    "Scikit-Learn",
    "Pandas",
    "NumPy",
    "SciPy",
    "Matplotlib",
    "Seaborn",
    "Streamlit",
    "Joblib",
    "Jupyter Notebook"
  ],
  "telemetryMatrix": {
    "projectType": "Unsupervised Shape Clustering Pipeline",
    "targetPlatform": "Python CLI, Notebook & Streamlit",
    "primaryDomain": "Agricultural Computer Vision",
    "architecturePipeline": "StandardScaler -> PCA -> K-Means",
    "testingAndAudit": "Verified",
    "projectStatus": "Completed",
    "projectYear": "2024"
  },
  "mlProjectInfo": {
    "projectType": "Unsupervised Clustering & Profiling Pipeline",
    "learningParadigm": "Unsupervised Learning",
    "modelDomain": "Morphological Seed Phenotyping",
    "projectStatus": "Completed",
    "developmentYear": "2024",
    "capabilityBadges": [
      "Strictly Unsupervised Clustering",
      "Multi-Paradigm Benchmarking",
      "Post-Hoc Validation"
    ]
  },
  "mlProblemContext": "Manual and supervised seed sorting forces administrative variety labels onto specimens even when physical shapes overlap continuously. Automated optical sorting needs unsupervised methods to discover natural morphological archetypes purely from 16 geometric features and diagnose physical separability limits.",
  "mlDatasetOverview": {
    "profileBadge": "13,611 Specimens x 16 Features",
    "source": "UCI Machine Learning Repository (Dry Bean Dataset, ID 602, DOI: 10.24432/C50S4B)",
    "modality": "Tabular Morphological Geometry",
    "scale": "13,611 Bean Specimens across 7 Varieties",
    "inputShape": "13,611 x 16 Numerical Features",
    "dataFormat": "CSV",
    "description": "Extracted from high-resolution digital images of 7 registered dry bean varieties (DERMASON, SIRA, SEKER, HOROZ, CALI, BARBUNYA, BOMBAY) with 12 dimensional metrics and 4 shape factors. The Class column is strictly isolated from preprocessing and clustering.",
    "provenanceNote": "Koklu & Ozkan (2020), Computers and Electronics in Agriculture (DOI: 10.1016/j.compag.2020.105507)."
  },
  "mlFeaturesAndDataCharacteristics": {
    "spaceBadge": "16D Geometric Feature Space",
    "spaceNote": "12 size/dimension metrics and 4 dimensionless shape factors with zero missing values",
    "objectiveOutput": {
      "label": "Target Output",
      "name": "Unsupervised Cluster Assignment (k = 4 Primary, k = 7 Comparative)",
      "type": "Discrete Cluster Partition",
      "distribution": "k=4 sizes: C0 2,221 (16.3%), C1 5,874 (43.2%), C2 524 (3.8%), C3 4,992 (36.7%)"
    },
    "inputSignals": [
      {
        "name": "Area & ConvexArea",
        "type": "Numerical (Integer Pixels)",
        "importanceOrRole": "Primary PC1 Scale Driver (r = 0.99994)",
        "description": "Pixel area inside bean contour and smallest enclosing convex polygon; isolates giant BOMBAY specimens (+4.10 Z-score in Cluster 2)."
      },
      {
        "name": "MajorAxisLength & MinorAxisLength",
        "type": "Numerical (Float64)",
        "importanceOrRole": "Principal Axial Dimensions",
        "description": "Longest contour axis and perpendicular width defining bean scale and slenderness along PC1 and PC2."
      },
      {
        "name": "Perimeter & EquivDiameter",
        "type": "Numerical (Float64)",
        "importanceOrRole": "Boundary Size Descriptors (r = 0.99138)",
        "description": "Total contour circumference and diameter of an equal-area circle driving global size variance."
      },
      {
        "name": "AspectRatio & Eccentricity",
        "type": "Numerical (Float64)",
        "importanceOrRole": "Primary Elongation Drivers (r = 0.92429)",
        "description": "Major-to-minor axis ratio and ellipse eccentricity separating elongated HOROZ/CALI/BARBUNYA beans from spherical SEKER."
      },
      {
        "name": "Roundness & Compactness",
        "type": "Numerical (Float64)",
        "importanceOrRole": "Primary PC2 Spherical Drivers",
        "description": "Contour circularity and equivalent-diameter compactness peaking in Cluster 0 (SEKER: Roundness 0.945, Compactness 0.897)."
      },
      {
        "name": "Solidity, Extent & ShapeFactor1-4",
        "type": "Numerical (Float64)",
        "importanceOrRole": "Micro-Geometry & Contour Regularity",
        "description": "Convexity ratio, bounding-box fill ratio, and non-linear shape ratios capturing curvature and boundary smoothness on PC3 and PC4."
      }
    ],
    "dataCharacteristicsSummary": "The dataset contains zero missing or infinite values and 68 identical HOROZ feature rows whose retention preserves cluster topology (ARI 0.99905). Strong multicollinearity links size features such as Area and ConvexArea (r = 0.99994), while PC1 and PC2 capture 81.90% of total geometric variance."
  },
  "mlLearningApproach": {
    "paradigm": "Strictly Unsupervised Shape-Based Clustering",
    "methodologyDescription": "Four distinct clustering paradigms—K-Means, Agglomerative Hierarchical Ward, Gaussian Mixture Models (full and tied covariance), and DBSCAN—were evaluated over k = 2 to 12 on Z-score standardized features, isolating the Class label strictly for post-hoc biological validation.",
    "justification": "K-Means outperformed Hierarchical Ward, GMM, and DBSCAN across Silhouette, Calinski-Harabasz, and Davies-Bouldin indices at k = 4 while avoiding DBSCAN density-chaining collapse across the continuous morphological manifold."
  },
  "mlDataPreparation": [
    {
      "step": "01",
      "title": "Strict Label Isolation & Data Quality Audit",
      "description": "Separated the 16 numerical features from Class, verified zero missing or infinite values, and validated via sensitivity analysis (ARI = 0.99905) that retaining 68 duplicate HOROZ measurements preserves cluster topology.",
      "badge": "Data Integrity"
    },
    {
      "step": "02",
      "title": "Multicollinearity & Outlier Diagnostics",
      "description": "Identified 21 feature pairs with Pearson |r| >= 0.90 and verified that extreme positive Z-scores (> +3 sigma) on Area and ConvexArea reflect genuine giant BOMBAY morphology rather than sensor errors.",
      "badge": "EDA & Collinearity"
    },
    {
      "step": "03",
      "title": "Z-Score Feature Standardization",
      "description": "Applied StandardScaler across all 16 features without label information and benchmarked against RobustScaler and an 8-feature non-redundant subset, confirming StandardScaler preserves BOMBAY cluster separation.",
      "badge": "StandardScaler"
    },
    {
      "step": "04",
      "title": "Principal Component Analysis (PCA)",
      "description": "Evaluated full 16-component variance (95.02% explained by 4 components) and fitted a 2D PCA projector (PC1 = 55.47% scale, PC2 = 26.43% shape; 81.90% cumulative) for cluster mapping and inference.",
      "badge": "PCA Projection"
    }
  ],
  "mlDataPreparationNote": "All scalers, PCA transformers, and clustering models were fit strictly on the 16 unlabeled numerical features; the Class column was never accessed during scaling, feature selection, k-selection, or model fitting.",
  "mlModelArchitectureAndTraining": {
    "algorithmName": "K-Means Clustering (k = 4 Primary, k = 7 Comparative)",
    "architectureSummary": "Centroid-based Euclidean partitioning with k-means++ initialization on 16 Z-score standardized morphological features, benchmarked against Agglomerative Ward linkage, full/tied covariance Gaussian Mixture Models, and DBSCAN across k = 2 to 12.",
    "configurationBadge": "k-means++ | n_init=20",
    "hyperparameters": [
      {
        "name": "n_clusters (k)",
        "value": "4 (Primary) / 7 (Comparative)",
        "description": "k=4 captures natural macro-archetypes; k=7 evaluates micro-separation aligned to 7 registered varieties."
      },
      {
        "name": "init",
        "value": "k-means++",
        "description": "Spread-out centroid seeding to improve convergence speed and partition consistency."
      },
      {
        "name": "n_init",
        "value": "20",
        "description": "Number of independent centroid initializations evaluated per model fit."
      },
      {
        "name": "max_iter",
        "value": "300",
        "description": "Maximum Lloyd iterations per restart run."
      },
      {
        "name": "random_state",
        "value": "42",
        "description": "Fixed seed ensuring deterministic centroid initialization and reproducibility."
      }
    ],
    "objectiveCriterion": "Within-Cluster Sum of Squares (Inertia)",
    "optimizationSolver": "Lloyd Expectation-Maximization (k-means++)",
    "trainingProcess": "K-Means models were fit across k = 2 to 12 using k-means++ initialization, 20 restarts, up to 300 iterations, and fixed seed 42 on 13,611 standardized samples. Cluster stability was audited via 5-iteration 85% subsampling bootstrap (ARI 0.9979) alongside k = 7 comparative modeling."
  },
  "mlEvaluationStrategy": {
    "validationMethod": "Internal Validity, Bootstrap Stability & Post-Hoc External Audit",
    "description": "Unsupervised partitions were selected using Silhouette Score, Calinski-Harabasz Index, Davies-Bouldin Index, and 5-iteration 85% subsampling ARI stability, followed by post-hoc contingency validation against held-out Class labels.",
    "metrics": [
      {
        "label": "SILHOUETTE (K=4)",
        "value": "0.3416",
        "benchmark": "Target > 0.30"
      },
      {
        "label": "CALINSKI-HARABASZ",
        "value": "8380.56",
        "benchmark": "Target > 8000"
      },
      {
        "label": "DAVIES-BOULDIN",
        "value": "0.9272",
        "benchmark": "Target < 1.00"
      },
      {
        "label": "STABILITY ARI",
        "value": "0.9979",
        "benchmark": "Target > 0.95"
      },
      {
        "label": "POST-HOC ARI (K=7)",
        "value": "0.6688",
        "benchmark": "Baseline: 0.4416"
      },
      {
        "label": "CLUSTER PURITY (K=7)",
        "value": "79.84%",
        "benchmark": "Baseline: 55.51%"
      }
    ]
  },
  "mlPerformanceVisualizations": [
    {
      "title": "Streamlit Executive Summary & Methodology Overview",
      "badge": "Streamlit UI",
      "imageUrl": "/images/projects/grouping_dry_bean_varieties/image01.webp",
      "telemetryNote": "Summarizes 13,611 observations, 16 geometric features, K-Means k=4 selection, and label-free modeling rules."
    },
    {
      "title": "Streamlit EDA — Pearson Correlation Matrix Tab",
      "badge": "Correlation UI",
      "imageUrl": "/images/projects/grouping_dry_bean_varieties/image02.webp",
      "telemetryNote": "Displays the 16x16 Pearson correlation matrix identifying 21 collinear feature pairs (|r| >= 0.90)."
    },
    {
      "title": "Streamlit EDA — Interactive Feature Scatterplot Tab",
      "badge": "Interactive EDA",
      "imageUrl": "/images/projects/grouping_dry_bean_varieties/image03.webp",
      "telemetryNote": "Plots 2,500 sampled specimens across Area and AspectRatio to inspect morphological dispersion."
    },
    {
      "title": "Streamlit Cluster Map & 2D PCA Projection",
      "badge": "PCA Dashboard",
      "imageUrl": "/images/projects/grouping_dry_bean_varieties/image04.webp",
      "telemetryNote": "Visualizes K-Means k=4 PCA cluster boundaries (81.90% variance) alongside cluster population counts."
    },
    {
      "title": "Streamlit Cluster Morphological Z-Score Profile",
      "badge": "Profile UI",
      "imageUrl": "/images/projects/grouping_dry_bean_varieties/image05.webp",
      "telemetryNote": "Presents standardized feature Z-score deviations differentiating Clusters 0 through 3."
    },
    {
      "title": "Streamlit Post-Hoc External Validation View",
      "badge": "External Audit",
      "imageUrl": "/images/projects/grouping_dry_bean_varieties/image06.webp",
      "telemetryNote": "Compares k=4 contingency heatmaps directly against the ground-truth 7-variety PCA scatter."
    },
    {
      "title": "Streamlit New Bean Prediction & Simulation Engine",
      "badge": "Inference UI",
      "imageUrl": "/images/projects/grouping_dry_bean_varieties/image07.webp",
      "telemetryNote": "Interactive 16-feature specimen input form with 7 variety presets for live cluster assignment."
    },
    {
      "title": "Morphological Feature Distributions (Histogram & KDE)",
      "badge": "Feature EDA",
      "imageUrl": "/images/projects/grouping_dry_bean_varieties/image08.webp",
      "telemetryNote": "Shows positive skewness on size features (Area skew +2.95) and negative skewness on Solidity (-2.55)."
    },
    {
      "title": "Internal Metric Comparison & K-Means Elbow Curves",
      "badge": "Model Selection",
      "imageUrl": "/images/projects/grouping_dry_bean_varieties/image09.webp",
      "telemetryNote": "K-Means outperforms Ward and GMM across k=2..12 with sharp inertia delta drops through k=4."
    },
    {
      "title": "Standardized Feature Outliers & PCA Scree Plot",
      "badge": "Variance & Outliers",
      "imageUrl": "/images/projects/grouping_dry_bean_varieties/image10.webp",
      "telemetryNote": "Highlights giant BOMBAY Z-scores (> +3 sigma) and confirms 4 PCs capture 95.02% cumulative variance."
    },
    {
      "title": "Gaussian Mixture Model BIC & AIC Selection",
      "badge": "GMM Diagnostics",
      "imageUrl": "/images/projects/grouping_dry_bean_varieties/image11.webp",
      "telemetryNote": "Full covariance GMM consistently beats tied covariance across k=2..12 on both BIC and AIC."
    },
    {
      "title": "Key Morphological Feature Boxplots Across Clusters",
      "badge": "Cluster Profiles",
      "imageUrl": "/images/projects/grouping_dry_bean_varieties/image12.webp",
      "telemetryNote": "Contrasts Area, Axis Lengths, AspectRatio, Solidity, and Compactness across Clusters C0-C3."
    },
    {
      "title": "Cluster Size Distribution & DBSCAN k-Distance Graph",
      "badge": "Density Analysis",
      "imageUrl": "/images/projects/grouping_dry_bean_varieties/image13.webp",
      "telemetryNote": "Shows balanced k=4 cluster populations (524 to 5,874 beans) and k=32 distance curve explaining DBSCAN chaining."
    },
    {
      "title": "Comparative K-Means (k=7) vs Ground-Truth PCA",
      "badge": "PCA Comparison",
      "imageUrl": "/images/projects/grouping_dry_bean_varieties/image14.webp",
      "telemetryNote": "Maps k=7 K-Means clusters alongside true varieties, showing BOMBAY, SEKER, DERMASON, and HOROZ alignment."
    },
    {
      "title": "Primary K-Means (k=4) PCA & Correlation Heatmap",
      "badge": "Feature & Cluster Map",
      "imageUrl": "/images/projects/grouping_dry_bean_varieties/image15.webp",
      "telemetryNote": "Pairs 2D PCA cluster separation with the 16x16 Pearson correlation matrix linking size and shape axes."
    },
    {
      "title": "Cluster Standardized Morphological Z-Score Heatmap",
      "badge": "Phenotypic Matrix",
      "imageUrl": "/images/projects/grouping_dry_bean_varieties/image16.webp",
      "telemetryNote": "Quantifies Cluster 2 (+4.10 Area), Cluster 0 (+1.57 Compactness), and Cluster 3 (+0.90 AspectRatio) signatures."
    },
    {
      "title": "Post-Hoc Contingency Heatmap — Primary Model (k=4)",
      "badge": "Contingency k=4",
      "imageUrl": "/images/projects/grouping_dry_bean_varieties/image17.webp",
      "telemetryNote": "Demonstrates 99.6% BOMBAY purity in Cluster 2 and 84.7% SEKER purity in Cluster 0 without labels."
    },
    {
      "title": "Post-Hoc Contingency Heatmap — Comparative Model (k=7)",
      "badge": "Contingency k=7",
      "imageUrl": "/images/projects/grouping_dry_bean_varieties/image18.webp",
      "telemetryNote": "Achieves 79.84% overall purity with 100.0% BOMBAY, 93.9% HOROZ, 92.2% DERMASON, and 92.2% SEKER clusters."
    }
  ],
  "mlVisualizationTelemetryNote": "Across 5 subsampling bootstrap iterations (85% population), K-Means k=4 preserved partition consistency with 0.9979 (+/- 0.0008) pairwise ARI and 0.99905 duplicate sensitivity ARI.",
  "mlModelInterpretationAndLimitations": {
    "interpretationSummary": "K-Means k = 4 separates specimens into Giant (Bombay, 99.62% purity), Spherical (Seker, 84.69%), Medium Oval (Dermason and Sira), and Elongated Kidney (Horoz, Cali, Barbunya) groups. Overlapping pairs share identical binary silhouettes, proving color or texture features are required to separate them.",
    "keyDrivers": [
      {
        "factor": "Area & MajorAxisLength (PC1: 55.47% Variance)",
        "impact": "Drives macro-scale separation, isolating giant BOMBAY beans (+4.10 Z-score) at 99.62% purity (k=4) and 100.0% purity (k=7)."
      },
      {
        "factor": "Roundness & Compactness vs AspectRatio (PC2: 26.43% Variance)",
        "impact": "Separates spherical SEKER specimens (Roundness 0.945, Compactness 0.897) from elongated HOROZ/CALI/BARBUNYA kidney beans (AspectRatio 1.805)."
      },
      {
        "factor": "Continuous Density Manifold",
        "impact": "Prevents DBSCAN from separating non-BOMBAY varieties because no zero-density gap exists between oval and elongated beans."
      }
    ],
    "limitations": [
      {
        "title": "Absence of Color & Surface Texture Features",
        "description": "Binary silhouette metrics cannot separate mottled BARBUNYA from white CALI kidney beans or distinguish geometrically overlapping DERMASON and SIRA varieties."
      },
      {
        "title": "DBSCAN Density-Chaining Collapse",
        "description": "Continuous density across medium-sized beans causes DBSCAN to discard 24.90%–40.36% as noise at small eps (0.8) or collapse into 1–2 clusters at larger eps (>= 1.2)."
      },
      {
        "title": "K-Means Spherical Cluster Assumption",
        "description": "Euclidean centroid partitioning assumes isotropic cluster geometry, slightly limiting fit on curved kidney-shaped varieties."
      }
    ],
    "futureImprovements": [
      {
        "title": "Multispectral Color & GLCM Texture Integration",
        "target": "Separate visually distinct mottled varieties such as BARBUNYA vs CALI using RGB/HSV and surface texture descriptors."
      },
      {
        "title": "Non-Linear Manifold Embeddings (UMAP / Spectral)",
        "target": "Capture curved morphological boundaries across elongated kidney-bean sub-populations."
      },
      {
        "title": "Conveyor Optical Sorter Edge Deployment",
        "target": "Integrate serialized StandardScaler and K-Means artifacts into real-time industrial seed sorting pipelines."
      }
    ],
    "discoveryNote": "Shape-only unsupervised clustering cleanly isolates extreme morphological archetypes (BOMBAY, SEKER, HOROZ, DERMASON) while objectively pinpointing where color sensing is physically mandatory."
  },
  "mlModelArtifacts": [
    {
      "name": "best_clustering_model.joblib",
      "format": "JOBLIB"
    },
    {
      "name": "kmeans_k7_model.joblib",
      "format": "JOBLIB"
    },
    {
      "name": "scaler.joblib",
      "format": "JOBLIB"
    },
    {
      "name": "pca_model.joblib",
      "format": "JOBLIB"
    },
    {
      "name": "pipeline_summary.json",
      "format": "JSON"
    },
    {
      "name": "evaluation_table.csv",
      "format": "CSV"
    }
  ],
  "mlMethodologySpecs": {
    "learningParadigm": "Unsupervised Learning",
    "modelArchitecture": "K-Means & PCA Pipeline",
    "coreFramework": "Scikit-Learn",
    "validationMethod": "Internal & Post-Hoc Audit"
  }
}
```
