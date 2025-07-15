import numpy as np
from scipy.spatial.distance import cdist

def normalize_features(features):
    """
    Normalizes feature vectors using L2 normalization.
    This ensures each feature vector has a magnitude of 1.
    
    Args:
        features (np.ndarray): A numpy array of features with shape (..., feature_dim).
    
    Returns:
        np.ndarray: The L2-normalized features.
    """
    # Calculate the L2 norm along the last dimension (the feature vector dimension)
    norms = np.linalg.norm(features, axis=-1, keepdims=True)
    
    # Add a small epsilon to avoid division by zero for-zero vectors
    norms = np.maximum(norms, 1e-6)
    
    normalized_features = features / norms
    return normalized_features

def calculate_chamfer_distance(features1, features2):
    """
    Calculates the Chamfer Distance between two individual descriptors (sets of views).
    It expects pre-normalized features.
    """
    # 'cosine' distance on L2-normalized vectors is a reliable similarity measure
    dists = cdist(features1, features2, 'cosine')
    
    min_dists_1 = np.min(dists, axis=1)
    min_dists_2 = np.min(dists, axis=0)
    
    chamfer_dist = np.sum(min_dists_1) + np.sum(min_dists_2)
    return chamfer_dist

def calculate_lfd_distance(model_A_features, model_B_features):
    """
    Calculates the final distance between two models by finding the best
    alignment among their 10 LightField Descriptors.
    
    Args:
        model_A_features (np.array): Shape (10, 10, feature_dim)
        model_B_features (np.array): Shape (10, 10, feature_dim)
    """
    # ✨ 1. L2-normalize the entire feature sets first for efficiency
    model_A_features_norm = normalize_features(model_A_features)
    model_B_features_norm = normalize_features(model_B_features)
    
    min_overall_distance = float('inf')
    
    # 2. Compare each of Model A's 10 descriptors to each of Model B's 10 descriptors
    for i in range(model_A_features_norm.shape[0]):
        for j in range(model_B_features_norm.shape[0]):
            descriptor_A = model_A_features_norm[i]
            descriptor_B = model_B_features_norm[j]
            
            # This function now receives normalized features
            dist = calculate_chamfer_distance(descriptor_A, descriptor_B)
            
            if dist < min_overall_distance:
                min_overall_distance = dist
                
    return min_overall_distance