import numpy as np
from scipy.spatial.distance import cdist
import os

def normalize_features(features):
    """
    Normalizes feature vectors using L2 normalization. This is crucial for
    making cosine distance a meaningful measure of similarity.
    
    Args:
        features (np.ndarray): An array of features with shape (num_views, feature_dim).
    
    Returns:
        np.ndarray: The L2-normalized features.
    """
    # Calculate the L2 norm along the last dimension (the feature vector itself)
    norms = np.linalg.norm(features, axis=-1, keepdims=True)
    # Use a small epsilon to prevent division by zero for zero-vectors
    return features / np.maximum(norms, 1e-12)


def calculate_chamfer_distance(descriptor_A, descriptor_B):
    """
    Calculates the Chamfer Distance between two descriptors (sets of view features).
    This is the standard method for comparing two unordered point sets.
    
    Args:
        descriptor_A (np.ndarray): Shape (num_views, feature_dim) for Model A.
        descriptor_B (np.ndarray): Shape (num_views, feature_dim) for Model B.
        
    Returns:
        float: The Chamfer Distance between the two descriptors.
    """
    # --- Step 1: L2-normalize features to use cosine distance ---
    # Cosine distance on L2-normalized vectors is a reliable similarity measure.
    desc_A_norm = normalize_features(descriptor_A)
    desc_B_norm = normalize_features(descriptor_B)
    
    # --- Step 2: Compute the pairwise distance matrix ---
    # `dists[i, j]` will be the cosine distance between view `i` of A and view `j` of B.
    # The shape will be (num_views_A, num_views_B).
    dists = cdist(desc_A_norm, desc_B_norm, 'cosine')
    
    # --- Step 3: Calculate the two terms of the Chamfer Distance ---
    # For each view in A, find the distance to the closest view in B.
    min_dists_A_to_B = np.min(dists, axis=1)
    
    # For each view in B, find the distance to the closest view in A.
    min_dists_B_to_A = np.min(dists, axis=0)
    
    # The total Chamfer Distance is the sum of these minimum distances.
    # It's a measure of the total "error" in matching the two sets of views.
    chamfer_dist = np.sum(min_dists_A_to_B) + np.sum(min_dists_B_to_A)
    
    return chamfer_dist

if __name__ == '__main__':
    print("--- Chamfer Distance Calculation Demo ---")
    
    # --- Setup: Create dummy data for demonstration ---
    # This reflects the new feature shape from your pipeline.
    num_views = 150
    feature_dim = 1536 # DINOv2-giant feature dimension
    
    # Create a descriptor for Model A
    descriptor_A = np.random.rand(num_views, feature_dim)
    
    # Descriptor B is slightly different from A
    descriptor_B = descriptor_A + 0.2 * np.random.rand(num_views, feature_dim)
    
    # Descriptor C is identical to A
    descriptor_C = descriptor_A.copy()
    
    # --- Calculate and Print Distances ---
    dist_AB = calculate_chamfer_distance(descriptor_A, descriptor_B)
    print(f"\nDistance between different descriptors (A vs B): {dist_AB:.4f}")
    print("(This value should be significantly greater than zero)")

    dist_AC = calculate_chamfer_distance(descriptor_A, descriptor_C)
    print(f"\nDistance between identical descriptors (A vs C): {dist_AC:.4f}")
    print("(This value should be very close to zero)")