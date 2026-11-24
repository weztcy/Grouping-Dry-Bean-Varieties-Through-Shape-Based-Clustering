"""
data_loader.py - Module for loading, validating, and auditing the Dry Bean Dataset.
"""

import os
import json
import numpy as np
import pandas as pd


def load_data(filepath='Dry_Bean.csv'):
    """
    Load dataset from CSV and metadata from JSON if available.
    Ensures strict separation of features X (16 numerical) and target y (Class).
    """
    if not os.path.exists(filepath):
        alt_path = os.path.join('dataset', filepath)
        if os.path.exists(alt_path):
            filepath = alt_path
        else:
            raise FileNotFoundError(f"Dataset file not found at {filepath} or {alt_path}")
            
    df = pd.read_csv(filepath)
    
    # Verify expected column structure
    expected_features = [
        'Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength', 'AspectRatio',
        'Eccentricity', 'ConvexArea', 'EquivDiameter', 'Extent', 'Solidity',
        'Roundness', 'Compactness', 'ShapeFactor1', 'ShapeFactor2',
        'ShapeFactor3', 'ShapeFactor4'
    ]
    target_col = 'Class'
    
    assert all(col in df.columns for col in expected_features), "Missing expected features in CSV"
    assert target_col in df.columns, "Missing 'Class' target column in CSV"
    
    # Strictly isolate X and y
    X = df[expected_features].copy()
    y = df[target_col].copy()
    
    # Load metadata if present
    metadata = {}
    meta_paths = ['metadata.json', os.path.join('dataset', 'metadata.json')]
    for mp in meta_paths:
        if os.path.exists(mp):
            with open(mp, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            break
            
    return df, X, y, metadata


def audit_data(df):
    """
    Perform rigorous audit of data quality: missing values, infs, types, duplicates, and distributions.
    """
    features = [c for c in df.columns if c != 'Class']
    
    audit_results = {
        'total_rows': int(df.shape[0]),
        'total_cols': int(df.shape[1]),
        'num_features': int(len(features)),
        'missing_values': {str(k): int(v) for k, v in df.isnull().sum().to_dict().items()},
        'infinite_values': int(np.isinf(df[features].values).sum()),
        'duplicate_rows': int(df.duplicated().sum()),
        'duplicate_feature_rows': int(df.duplicated(subset=features).sum()),
        'class_distribution': {str(k): int(v) for k, v in df['Class'].value_counts().to_dict().items()},
        'dtypes': {str(k): str(v) for k, v in df.dtypes.to_dict().items()}
    }
    
    return audit_results


def analyze_duplicates(df):
    """
    Analyze the duplicate rows in detail.
    """
    features = [c for c in df.columns if c != 'Class']
    dup_mask = df.duplicated(subset=features, keep=False)
    dup_df = df[dup_mask]
    
    unique_dup_count = df.duplicated(subset=features).sum()
    class_dup_counts = df[df.duplicated(subset=features)]['Class'].value_counts().to_dict()
    
    return {
        'total_rows_involved': int(len(dup_df)),
        'unique_duplicated_rows': int(unique_dup_count),
        'duplicate_class_distribution': class_dup_counts,
        'example_duplicate_indices': dup_df.index[:10].tolist()
    }
