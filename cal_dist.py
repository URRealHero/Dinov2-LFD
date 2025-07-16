import numpy as np
from scipy.spatial.distance import cdist
import os

def load_alignments(filepath='data/align10.txt'):
    """
    Loads the alignment data from the binary 'align10.txt' file.
    
    The original C code reads this as a flat array of unsigned chars and uses it
    as a 2D array. Based on the loops `for (j=0; j<60; j++)` and `for(m=0; m<CAMNUM; m++)`
    where CAMNUM is 10, we interpret this as a 60x10 matrix of alignment indices.
    The C code actually reads 60 * 20 bytes, but only uses the first 10 columns.
    
    Args:
        filepath (str): The path to the align10.txt file.
        
    Returns:
        np.ndarray: A NumPy array of shape (60, 10) containing the alignment indices.
                    Returns None if the file is not found.
    """
    if not os.path.exists(filepath):
        print(f"❌ Error: Alignment file not found at '{filepath}'")
        print("Please ensure 'align10.txt' is in a 'data' subdirectory.")
        return None
        
    # The C code reads 60 * 20 `unsigned char` values.
    # We load it and reshape, then take only the first 10 columns used in the matching loop.
    try:
        alignments_full = np.fromfile(filepath, dtype=np.uint8).reshape(60, 20)
        # The inner loop in C is `m < CAMNUM` where CAMNUM=10, so we only need 10 columns.
        alignments = alignments_full[:, :10].astype(int)
        print(f"✅ Successfully loaded alignments from '{filepath}' with shape {alignments.shape}.")
        return alignments
    except Exception as e:
        print(f"❌ Error reading or reshaping alignment file: {e}")
        return None


def normalize_features(features):
    """
    Normalizes feature vectors using L2 normalization.
    
    Args:
        features (np.ndarray): An array of features with shape (..., feature_dim).
    
    Returns:
        np.ndarray: The L2-normalized features.
    """
    norms = np.linalg.norm(features, axis=-1, keepdims=True)
    # Use a small epsilon to prevent division by zero
    return features / np.maximum(norms, 1e-12)


def calculate_lfd_distance(model_A_features, model_B_features, alignments=None):
    """
    Calculates the final LFD distance between two models by finding the best
    rotational alignment among their 10 LightField Descriptors.
    
    This function faithfully reproduces the set-to-set comparison logic from the
    original C code, but uses cosine distance on deep features.
    
    Args:
        model_A_features (np.ndarray): Shape (10, 10, feature_dim) for Model A.
        model_B_features (np.ndarray): Shape (10, 10, feature_dim) for Model B.
        alignments (np.ndarray): The (60, 10) alignment table from align10.txt.
        
    Returns:
        float: The final minimum distance between the two models.
    """
    if alignments is None:
        alignments = load_alignments()
        

    # --- Step 1: L2-normalize all features once for efficiency ---
    # This prepares them for cosine distance calculation.
    model_A_norm = normalize_features(model_A_features)
    model_B_norm = normalize_features(model_B_features)
    
    min_overall_distance = float('inf')
    
    # --- Step 2: Iterate through all 10x10 pairs of descriptors ---
    # This is the outer loop from the C code, comparing each camera rig ("angle")
    # from the source model to each rig from the destination model.
    num_descriptors = model_A_norm.shape[0]
    for i in range(num_descriptors):
        for j in range(num_descriptors):
            descriptor_A = model_A_norm[i]  # Shape: (10, 768)
            descriptor_B = model_B_norm[j]  # Shape: (10, 768)
            
            # --- Step 3: Pre-compute the pairwise distance matrix for this descriptor pair ---
            # This creates a 10x10 matrix where `pairwise_dists[v1, v2]` is the
            # cosine distance between view v1 of descriptor A and view v2 of descriptor B.
            pairwise_dists = cdist(descriptor_A, descriptor_B, 'cosine')

            # --- Step 4: Perform the rotational alignment search ---
            # This is the core logic from `MatchLF`. It finds the best rotational
            # alignment out of the 60 possibilities.
            min_alignment_dist = float('inf')
            
            # The C code compares every alignment `align10[k]` to a fixed reference `align10[0]`.
            reference_alignment = alignments[0]
            
            for k in range(alignments.shape[0]): # Loop through all 60 alignments
                current_alignment = alignments[k]
                current_sum_dist = 0
                
                # Sum the distances for the 10 views based on the fixed pairing
                for m in range(10):
                    # Get the indices for this specific pairing
                    # This maps a view from descriptor A to a view from descriptor B
                    idx_A = current_alignment[m]
                    idx_B = reference_alignment[m]
                    
                    # Add the pre-computed distance for this specific pair
                    current_sum_dist += pairwise_dists[idx_A, idx_B]
                
                # Check if this alignment gives a smaller total distance
                if current_sum_dist < min_alignment_dist:
                    min_alignment_dist = current_sum_dist
            
            # --- Step 5: Update the overall minimum distance ---
            # The final distance is the minimum found across all 100 descriptor pairings.
            if min_alignment_dist < min_overall_distance:
                min_overall_distance = min_alignment_dist
                
    return min_overall_distance

if __name__ == '__main__':
    print("--- LFD Aligned Distance Calculation Demo ---")
    
    # --- Setup: Create dummy data for demonstration ---
    
    # Create a dummy data directory and align10.txt file
    if not os.path.exists('data'):
        os.makedirs('data')
    
    # Load the alignments
    alignments = load_alignments('data/align10.txt')
    
    if alignments is not None:
        # Create two dummy feature sets for two models
        # Shape: (num_descriptors, num_views_per_descriptor, feature_dimension)
        feature_dim = 768
        model_A = np.random.rand(10, 10, feature_dim).astype(np.float32)
        
        # Model B will be a slightly perturbed version of Model A for a non-zero distance
        model_B = model_A + 0.1 * np.random.rand(10, 10, feature_dim).astype(np.float32)
        
        # Create a third model identical to A for a near-zero distance
        model_C = model_A.copy()

        print("\nCalculating distance between Model A and Model B (should be > 0)...")
        distance_AB = calculate_lfd_distance(model_A, model_B, alignments)
        print(f"Final Distance (A vs B): {distance_AB:.4f}")

        print("\nCalculating distance between Model A and Model C (should be ~ 0)...")
        distance_AC = calculate_lfd_distance(model_A, model_C, alignments)
        print(f"Final Distance (A vs C): {distance_AC:.4f}")

