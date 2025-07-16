import os
import numpy as np
import trimesh
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse

# Import our custom modules
import encode
import cal_dist

def process_single_object(object_path, camera_views, model_type, model_assets_info, quality):
    """
    A helper function that runs in a separate process to encode a single object.
    This is what each parallel worker will execute.
    """
    # This function now gracefully handles being called from a parallel worker
    try:
        print(f"WORKER (PID: {os.getpid()}) started for: {os.path.basename(object_path)}")
        
        # Each worker process must load its own model assets
        # getattr dynamically gets the loading function (e.g., encode.load_dinov2_model)
        model_loader = getattr(encode, model_assets_info['loader_func_name'])
        # The '*' unpacks the arguments for the loader function (e.g., paths for sscd or sam2)
        model_assets = model_loader(*model_assets_info.get('args', []))
    
        descriptor = encode.generate_descriptor(
            object_path, camera_views, model_type, model_assets, quality=quality
        )
        
        print(f"✅ WORKER (PID: {os.getpid()}) finished: {os.path.basename(object_path)}")
        return object_path, descriptor
        
    except Exception as e:
        print(f"!!!!!!!!!!!!!! ERROR in worker for {os.path.basename(object_path)} !!!!!!!!!!!!!!")
        # Print the full exception to help with debugging
        import traceback
        traceback.print_exc()
        return object_path, None

if __name__ == '__main__':
    # --- 1. Main Configuration ---
    parser = argparse.ArgumentParser(description="Run 3D model similarity tests in parallel.")
    parser.add_argument('--models', nargs='+', default=['assets/model1.glb', 'assets/model2.glb'],
                        help="List of model files to process.")
    parser.add_argument('--model_type', type=str, default='dinov2',
                        choices=['dinov1', 'dinov2', 'clip', 'sscd', 'sam2'],
                        help="The feature extractor model to use.")
    parser.add_argument('--quality', type=str, default='FAST', choices=['FAST', 'HIGH'],
                        help="Rendering quality ('FAST' uses EEVEE, 'HIGH' uses CYCLES).")
    parser.add_argument('--views', type=int, default=50, help="Number of views to render per object.")
    parser.add_argument('--workers', type=int, default=1, help="Number of parallel processes to run.")
    
    # Add paths for models that require them
    parser.add_argument('--sscd_path', type=str, default='checkpoints/sscd_imagenet_mixup.torchscript.pt', help="Path to the SSCD model file.")
    parser.add_argument('--sam2_config', type=str, default='configs/sam2.1_hiera_l.yaml', help="Path to the SAM2 model config file.")
    parser.add_argument('--sam2_checkpoint', type=str, default='checkpoints/sam2.1_hiera_large.pt', help="Path to the SAM2 model checkpoint.")
    
    args = parser.parse_args()

    # ---(Optional) Download Blender ---
    if not os.path.exists(encode.BLENDER_EXECUTABLE_PATH):
        print("Blender executable not found. Please ensure Blender is installed and set up correctly.")
        print(f"Expected path: {encode.BLENDER_EXECUTABLE_PATH}")
        encode._install_blender()
        print("Blender installation complete. Please re-run the script.")

    # --- 2. Setup ---
    print("--- Initializing Test Environment ---")
    
    # This dictionary makes the script easily expandable.
    # To add a new model, just add its configuration here.
    MODEL_ASSET_INFO = {
        'dinov1': {'loader_func_name': 'load_dinov1_model'},
        'dinov2': {'loader_func_name': 'load_dinov2_model'},
        'clip': {'loader_func_name': 'load_clip_model'},
        'sscd': {'loader_func_name': 'load_sscd_model', 'args': [args.sscd_path]},
        'sam2': {'loader_func_name': 'load_sam2_model', 'args': [args.sam2_config, args.sam2_checkpoint]},
    }
    
    # Validate that necessary paths are provided if a specific model is chosen
    if args.model_type == 'sscd' and not os.path.exists(args.sscd_path):
        raise FileNotFoundError(f"SSCD model not found. Please provide the path via --sscd_path")
    if args.model_type == 'sam2' and (not os.path.exists(args.sam2_config) or not os.path.exists(args.sam2_checkpoint)):
        raise FileNotFoundError(f"SAM2 config or checkpoint not found. Please provide paths via --sam2_config and --sam2_checkpoint")

    # Generate camera views once in the main process
    camera_views = encode.generate_spherical_views(num_views=args.views)
    
    results = {}
    start_time = time.time()
    
    # --- 3. Process All Objects in Parallel ---
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        # Submit all rendering jobs to the pool
        future_to_path = {
            executor.submit(process_single_object, path, camera_views, args.model_type, MODEL_ASSET_INFO[args.model_type], args.quality): path
            for path in args.models
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_path):
            path = future_to_path[future]
            try:
                path, descriptor = future.result()
                if descriptor is not None:
                    results[path] = descriptor
            except Exception as exc:
                print(f"❌ An exception occurred in the main process for {os.path.basename(path)}: {exc}")

    total_time = time.time() - start_time
    print(f"\n--- All parallel jobs completed in {total_time:.2f} seconds. ---")

    # --- 4. Run Pairwise Comparisons on the Results ---
    if len(results) < 2:
        print("⚠️ Need at least two successfully rendered models to compare.")
    else:
        print("\n--- Pairwise Dissimilarity Matrix (Chamfer Distance) ---")
        model_paths = sorted(list(results.keys()))
        
        # Print a formatted table header
        header = "| {:<25} |".format("Model")
        for path in model_paths:
            header += " {:<25} |".format(os.path.basename(path))
        print(header)
        print("-" * len(header))

        # Calculate and print the distance matrix
        for i in range(len(model_paths)):
            row_str = "| {:<25} |".format(os.path.basename(model_paths[i]))
            for j in range(len(model_paths)):
                if i == j:
                    distance_str = "0.00"
                else:
                    path_A = model_paths[i]
                    path_B = model_paths[j]
                    distance = cal_dist.calculate_chamfer_distance(results[path_A], results[path_B])
                    distance_str = f"{distance:.4f}"
                
                row_str += " {:<25} |".format(distance_str)
            print(row_str)

    print("\nAll tests completed.")