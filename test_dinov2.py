# test.py
import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import numpy as np
import trimesh
import pyrender
import torch

# Import our custom modules
import encode
import cal_dist

if __name__ == '__main__':
    # --- Configuration ---
    MODEL_FILE = 'assets/model1.glb'
    CAMERA_DIR = 'data'
    NUM_TESTS = 1
    DIFFERENT_MODEL_FILE = 'assets/model2.glb'

    if not os.path.exists(MODEL_FILE):
        print(f"❌ Error: Model file not found at '{MODEL_FILE}'")
        exit()

    # --- 1. Initial Setup ---
    print("--- Initializing Test Environment ---")
    processor, model, device = encode.load_dinov2_model()
    # processor, model, device = encode.load_clip_model()
    renderer = pyrender.OffscreenRenderer(256, 256)
    camera_rigs = encode.load_camera_positions(CAMERA_DIR) # Note: rigs, not positions
    
    # --- 2. Generate Baseline Features ---
    print(f"\n🔄 Encoding baseline features for '{MODEL_FILE}'...")
    original_mesh = trimesh.load(MODEL_FILE, force='mesh')
    baseline_features = encode.generate_features(
        original_mesh, camera_rigs, renderer, model_assets=(processor, model, device)
    )
    print(f"✅ Baseline features generated with shape {baseline_features.shape}.")
    
    # print(f"The maximum of the feature vector is {baseline_features.max()}, The minimum of the feature vector is {baseline_features.min()}")

    # --- 3. Run Rotation Tests ---
    total_dissimilarity = 0
    for i in range(NUM_TESTS):
        print(f"\n--- 🔄 Rotation Test {i + 1}/{NUM_TESTS} ---")

        # a. Generate and apply random rotation
        axis = np.random.rand(3) - 0.5
        axis /= np.linalg.norm(axis)
        angle_rad = np.deg2rad(np.random.uniform(0, 360))
        transform = trimesh.transformations.rotation_matrix(angle_rad, axis)
        
        rotated_mesh = original_mesh.copy()
        rotated_mesh.apply_transform(transform)

        # c. Generate features for the rotated mesh
        rotated_features = encode.generate_features(
            rotated_mesh, camera_rigs, renderer, model_assets=(processor, model, device)
        )
        
        # d. Calculate the dissimilarity using the LFD method
        dissimilarity = cal_dist.calculate_lfd_distance(baseline_features, rotated_features)
        total_dissimilarity += dissimilarity
        
        print(f"✅ Dissimilarity Score: {dissimilarity:.4f}")

    # --- 4. Cleanup and Final Report ---

    print("\n--- ✅ Test Complete ---")
    print(f"Average dissimilarity across {NUM_TESTS} random rotations: {total_dissimilarity / NUM_TESTS:.4f}")
    
    
    print("\n--- Testing Two Different Models Distance ---")
    diff_mesh = trimesh.load(DIFFERENT_MODEL_FILE, force='mesh')
    diff_features = encode.generate_features(
        diff_mesh, camera_rigs, renderer, model_assets=(processor, model, device)
    )
    
    diff_dissimilarity = cal_dist.calculate_lfd_distance(baseline_features, diff_features)
    renderer.delete()
    print(f"Dissimilarity between '{MODEL_FILE}' and '{DIFFERENT_MODEL_FILE}': {diff_dissimilarity:.4f}")
    print("All tests completed successfully!")
    
    save_path = 'assets'
    encode.save_features(
        baseline_features, os.path.join(save_path, 'baseline_features.npy')
    )
    encode.save_features(
        diff_features, os.path.join(save_path, 'diff_features.npy')
    )