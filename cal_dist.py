# cal_dist.py
import numpy as np
from scipy.spatial.distance import cdist

def calculate_chamfer_distance(features1, features2):
    """
    Calculates the Chamfer Distance between two individual descriptors (sets of views).
    """
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
        model_A_features (np.array): Shape (10, 10, 768)
        model_B_features (np.array): Shape (10, 10, 768)
    """
    min_overall_distance = float('inf')
    
    # Compare each of Model A's 10 descriptors to each of Model B's 10 descriptors
    for i in range(model_A_features.shape[0]):
        for j in range(model_B_features.shape[0]):
            descriptor_A = model_A_features[i]
            descriptor_B = model_B_features[j]
            
            dist = calculate_chamfer_distance(descriptor_A, descriptor_B)
            
            if dist < min_overall_distance:
                min_overall_distance = dist
                
    return min_overall_distance