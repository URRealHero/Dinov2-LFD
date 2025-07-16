import os
import numpy as np
import trimesh
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import tempfile
import shutil

# Import our custom modules
# NOTE: This script assumes 'encode.py' and 'cal_dist.py' are updated accordingly.
import encode
import cal_dist


# --- Phase 1 Worker: Renders images and returns their directory ---
def render_worker(object_path, camera_views, quality):
    """
    A helper function that runs in a separate process to RENDER an object's views.
    It does NOT load the feature extractor model, keeping its memory footprint low.
    """
    try:
        print(f"RENDER WORKER (PID: {os.getpid()}) started for: {os.path.basename(object_path)}")
        
        # This function should exist in your 'encode.py' module.
        # It handles running Blender for one object and saving the images to a new temp directory.
        image_dir = encode.render_views_to_tempdir(object_path, camera_views, quality)
        
        print(f"✅ RENDER WORKER (PID: {os.getpid()}) finished: {os.path.basename(object_path)}")
        return object_path, image_dir
        
    except Exception as e:
        print(f"!!!!!!!!!!!!!! ERROR in render worker for {os.path.basename(object_path)} !!!!!!!!!!!!!!")
        import traceback
        traceback.print_exc()
        return object_path, None


if __name__ == '__main__':
    # --- 1. Main Configuration ---
    parser = argparse.ArgumentParser(description="Run 3D model similarity tests in parallel.")
    parser.add_argument('--models', nargs='+', default=['assets/model1.glb', 'assets/model2.glb'],
                        help="List of model files to process. The first model will be used for the rotation invariance test.")
    parser.add_argument('--model_type', type=str, default='dinov2',
                        choices=['dinov1', 'dinov2', 'clip', 'sscd', 'sam2'],
                        help="The feature extractor model to use.")
    parser.add_argument('--quality', type=str, default='FAST', choices=['FAST', 'HIGH'],
                        help="Rendering quality ('FAST' uses EEVEE, 'HIGH' uses CYCLES).")
    parser.add_argument('--views', type=int, default=50, help="Number of views to render per object.")
    parser.add_argument('--workers', type=int, default=4, help="Number of parallel processes to run.")
    parser.add_argument('--output_dir', type=str, default='assets/results_features', help="Directory to save the final feature descriptors.")
    
    # Add paths for models that require them
    parser.add_argument('--sscd_path', type=str, default='checkpoints/sscd_imagenet_mixup.torchscript.pt', help="Path to the SSCD model file.")
    parser.add_argument('--sam2_config', type=str, default='configs/sam2.1_hiera_l.yaml', help="Path to the SAM2 model config file.")
    parser.add_argument('--sam2_checkpoint', type=str, default='checkpoints/sam2.1_hiera_large.pt', help="Path to the SAM2 model checkpoint.")
    
    args = parser.parse_args()

    # ---(Optional) Download Blender ---
    if not os.path.exists(encode.BLENDER_EXECUTABLE_PATH):
        print("Blender executable not found. Please ensure Blender is installed and set up correctly.")
        encode._install_blender()

    # --- 2. Setup ---
    print("--- Initializing Test Environment ---")
    
    MODEL_ASSET_INFO = {
        'dinov1': {'loader_func_name': 'load_dinov1_model'},
        'dinov2': {'loader_func_name': 'load_dinov2_model'},
        'clip': {'loader_func_name': 'load_clip_model'},
        'sscd': {'loader_func_name': 'load_sscd_model', 'args': [args.sscd_path]},
        'sam2': {'loader_func_name': 'load_sam2_model', 'args': [args.sam2_config, args.sam2_checkpoint]},
    }
    
    # This will be the master list of all models to be processed
    models_to_process = list(args.models)
    rotated_models_temp_dir = tempfile.mkdtemp()

    # --- 2a. Rotation Invariance Check ---
    try:
        if models_to_process:
            NUM_ROT = 2
            base_model_path = models_to_process[0]
            print(f"\n--- 🔄 Generating {NUM_ROT} random rotations for '{os.path.basename(base_model_path)}' to test invariance ---")
            base_mesh = trimesh.load(base_model_path, force='mesh')
            
            for i in range(NUM_ROT):
                transform = trimesh.transformations.random_rotation_matrix()
                rotated_mesh = base_mesh.copy()
                rotated_mesh.apply_transform(transform)
                
                original_name, original_ext = os.path.splitext(os.path.basename(base_model_path))
                rotated_name = f"{original_name}_ROT_{i+1}{original_ext}"
                rotated_path = os.path.join(rotated_models_temp_dir, rotated_name)
                
                rotated_mesh.export(rotated_path)
                print(f"  -> Created temporary rotated model: {rotated_name}")
                models_to_process.append(rotated_path)
        
        # --- 3. PHASE 1: PARALLEL RENDERING ---
        # Each worker runs Blender to render views. This is CPU-intensive.
        # No large AI models are loaded here, saving VRAM.
        print(f"\n--- Starting PHASE 1: Parallel Rendering with {args.workers} worker(s) ---")
        camera_views = encode.generate_spherical_views(num_views=args.views)
        start_time = time.time()
        
        rendered_data = {} # Stores {model_path: image_directory}
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_to_path = {
                executor.submit(render_worker, path, camera_views, args.quality): path
                for path in models_to_process
            }
            
            for future in as_completed(future_to_path):
                path, image_dir = future.result()
                if image_dir:
                    rendered_data[path] = image_dir
        
        print(f"--- Phase 1 completed in {time.time() - start_time:.2f} seconds. ---")

        # --- 4. PHASE 2: CENTRALIZED ENCODING ---
        # The main process now loads the model ONCE and processes all rendered images.
        # This is VRAM-efficient and fast on a GPU.
        print("\n--- Starting PHASE 2: Centralized Encoding ---")
        print(f"--- 🧠 Loading '{args.model_type}' model into VRAM... ---")
        model_loader = getattr(encode, MODEL_ASSET_INFO[args.model_type]['loader_func_name'])
        model_assets = model_loader(*MODEL_ASSET_INFO[args.model_type].get('args', []))

        results = {}
        print("--- ⚙️  Encoding rendered images to generate descriptors... ---")
        for path, image_dir in rendered_data.items():
            # This function should exist in 'encode.py'.
            # It loads images from a directory and uses the model to get a descriptor.
            descriptor = encode.create_descriptor_from_images(image_dir, model_type=args.model_type, model_assets=model_assets)
            results[path] = descriptor
            # Clean up the temporary images for this model immediately after use
            shutil.rmtree(image_dir)
        print("--- ✅ Encoding complete. ---")

        # --- 5. Save Features & Run Comparisons ---
        if len(results) < 2:
            print("⚠️ Need at least two successfully processed models to compare.")
        else:
            # --- 5a. Save all features ONCE to a persistent directory ---
            os.makedirs(args.output_dir, exist_ok=True)
            print(f"\n--- 💾 Saving feature descriptors to '{args.output_dir}/' ---")
            for path, descriptor in results.items():
                base_name = os.path.basename(path)
                file_name, _ = os.path.splitext(base_name)
                save_path = os.path.join(args.output_dir, f"{file_name}.npy")
                np.save(save_path, descriptor)
            print("--- ✅ All features saved. ---")

            # --- 5b. Calculate and Print Dissimilarity Matrix ---
            print("\n--- Pairwise Dissimilarity Matrix (Chamfer Distance) ---")
            model_paths = sorted(list(results.keys()))
            
            col_width = max(len(os.path.basename(p)) for p in model_paths) + 2
            header = f"| {'Model':<{col_width}} |"
            for path in model_paths:
                header += f" {os.path.basename(path):<{col_width}} |"
            print(header)
            print("-" * len(header))

            for i in range(len(model_paths)):
                row_str = f"| {os.path.basename(model_paths[i]):<{col_width}} |"
                for j in range(len(model_paths)):
                    if i == j:
                        distance_str = "0.0000"
                    else:
                        path_A, path_B = model_paths[i], model_paths[j]
                        distance = cal_dist.calculate_chamfer_distance(results[path_A], results[path_B])
                        distance_str = f"{distance:.4f}"
                    
                    row_str += f" {distance_str:<{col_width}} |"
                print(row_str)

    finally:
        # --- 6. Final Cleanup ---
        print(f"\n--- 🧹 Cleaning up temporary directory: {rotated_models_temp_dir} ---")
        shutil.rmtree(rotated_models_temp_dir)

    print("\nAll tests completed.")