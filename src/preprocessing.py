"""
preprocessing.py - Module for scaling, feature selection, and dimensionality reduction.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA


def get_scaler(scaler_type='standard'):
    """
    Factory function for feature scalers.
    """
    if scaler_type == 'standard':
        return StandardScaler()
    elif scaler_type == 'robust':
        return RobustScaler()
    elif scaler_type == 'minmax':
        return MinMaxScaler()
    else:
        raise ValueError(f"Unknown scaler type: {scaler_type}. Choose 'standard', 'robust', or 'minmax'.")


def scale_features(X, scaler=None, scaler_type='standard'):
    """
    Scale feature matrix X strictly without any label information.
    """
    if scaler is None:
        scaler = get_scaler(scaler_type)
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
        
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    return X_scaled_df, scaler


def get_reduced_features(X_scaled_df):
    """
    Create a curated non-redundant feature representation to eliminate near-duplicate collinear features.
    
    Exclusions:
    - ConvexArea (0.9999 correlation with Area) -> keep Area
    - EquivDiameter (0.9914 correlation with Perimeter, 0.985 with Area) -> redundant with Area & Perimeter
    - ShapeFactor3 (0.9987 correlation with Compactness) -> keep Compactness
    - Eccentricity (0.924 correlation with AspectRatio, 0.970 with Compactness) -> keep AspectRatio
    - MajorAxisLength (0.977 correlation with Perimeter) -> capture via MinorAxisLength and AspectRatio
    
    Retained core 8 features:
    ['Area', 'Perimeter', 'MinorAxisLength', 'AspectRatio', 'Solidity', 'Roundness', 'Compactness', 'ShapeFactor1']
    """
    selected_cols = [
        'Area', 'Perimeter', 'MinorAxisLength', 'AspectRatio',
        'Solidity', 'Roundness', 'Compactness', 'ShapeFactor1'
    ]
    return X_scaled_df[selected_cols].copy()


def perform_pca(X_scaled_df, n_components=None, random_state=42):
    """
    Apply Principal Component Analysis on scaled features.
    Returns:
    - pca model
    - transformed coordinates as DataFrame
    - explained variance ratio
    - cumulative explained variance
    """
    pca = PCA(n_components=n_components, random_state=random_state)
    pca_transformed = pca.fit_transform(X_scaled_df)
    
    num_comp = pca_transformed.shape[1]
    col_names = [f'PC{i+1}' for i in range(num_comp)]
    pca_df = pd.DataFrame(pca_transformed, columns=col_names, index=X_scaled_df.index)
    
    evr = pca.explained_variance_ratio_
    cum_evr = np.cumsum(evr)
    
    return pca, pca_df, evr, cum_evr
