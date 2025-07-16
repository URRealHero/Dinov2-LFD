import os
import time
import argparse
import shutil
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

# Import the refactored functions from your encode.py module
import encode

def find_model_paths(input_dir, output_filename, resume=True):
    """
    Scans the input directory for model_normalized.glb files.
    If resume is True, it skips models that already have an output file.
    """
    print(f"🔍 Scanning for models in '{input_dir}'...")
    model_paths_to_process = []
    total_models_found = 0
    skipped_count = 0

    # Walk through the directory structure
    for root, dirs, files in os.walk(input_dir):
        if "model_normalized.glb" in files:
            total_models_found += 1
            model_path = os.path.join(root, "model_normalized.glb")
            output_path = os.path.join(root, output_filename)

            # Check if we should skip this model
            if resume and os.path.exists(output_path):
                skipped_count += 1
                continue
            
            model_paths_to_process.append(model_path)

    print(f"Found {total_models_found} total models.")
    if resume:
        print(f"Skipping {skipped_count} that already have features.")
    print(f"➡️  Found {len(model_paths_to_process)} new models to process.")
    
    return model_paths_to_process

if __name__ == '__main__':
    # --- 1. Configuration ---
    parser = argparse.ArgumentParser(
        description="Process a large dataset of 3D models in batches to generate DINOv2 features."
    )
    parser.add_argument(
        '--input_dir', type=str, default='/workspace/Objaversexl_sketchfab/raw',
        help="Root directory containing {model_uid}/model_normalized.glb files."
    )
    parser.add_argument(
        '--workers', type=int, default=8,  # Reduced default for stability
        help="Number of parallel processes to run."
    )
    parser.add_argument(
        '--views', type=int, default=50,
        help="Number of views to render per object."
    )
    parser.add_argument(
        '--batch_size', type=int, default=1000,
        help="Number of models to process in each batch to manage disk space."
    )
    parser.add_argument(
        '--output_filename', type=str, default='dinov2_feature.npy',
        help="Filename for the saved feature descriptor."
    )
    parser.add_argument(
        '--quality', type=str, default='FAST', choices=['FAST', 'HIGH'],
        help="Rendering quality ('FAST' uses EEVEE, 'HIGH' uses CYCLES)."
    )
    parser.add_argument(
        '--no-resume', dest='resume', action='store_false',
        help="Flag to disable resuming and re-process all models."
    )
    args = parser.parse_args()

    # --- 2. Discover All Models ---
    all_model_paths = find_model_paths(args.input_dir, args.output_filename, args.resume)
    
    if not all_model_paths:
        print("✅ No new models to process. Exiting.")
        exit(0)

    # --- 3. Load AI Model Once ---
    print("\n--- 🧠 Loading DINOv2 model into VRAM once for the entire run... ---")
    model_assets = encode.load_dinov2_model()
    print("--- ✅ DINOv2 model loaded. ---\n")
    
    # --- 4. Process All Models in Batches ---
    overall_start_time = time.time()
    total_processed_count = 0
    num_batches = (len(all_model_paths) + args.batch_size - 1) // args.batch_size

    for i in range(num_batches):
        batch_start_idx = i * args.batch_size
        batch_end_idx = batch_start_idx + args.batch_size
        batch_paths = all_model_paths[batch_start_idx:batch_end_idx]

        print("─" * 80)
        print(f"📦 Processing Batch {i+1} / {num_batches} (Models {batch_start_idx+1} to {min(batch_end_idx, len(all_model_paths))})")
        print("─" * 80)

        # --- PHASE 1 (BATCH): PARALLEL RENDERING ---
        print(f"--- 🚀 Starting Rendering for batch with {args.workers} workers... ---")
        batch_render_start_time = time.time()
        camera_views = encode.generate_spherical_views(num_views=args.views)
        rendered_data = {}

        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_to_path = {
                executor.submit(encode.render_views_to_tempdir, path, camera_views, args.quality): path
                for path in batch_paths
            }
            for future in as_completed(future_to_path):
                original_path = future_to_path[future]
                try:
                    image_dir = future.result()
                    if image_dir: rendered_data[original_path] = image_dir
                except Exception as exc:
                    print(f"❌ Render error for {os.path.basename(os.path.dirname(original_path))}: {exc}")
        
        print(f"--- ✅ Rendering for batch complete in {time.time() - batch_render_start_time:.2f}s. ---")

        # --- PHASE 2 (BATCH): SEQUENTIAL ENCODING & CLEANUP ---
        if not rendered_data:
            print("⚠️ No models were successfully rendered in this batch. Skipping to next batch.")
            continue
        
        print("--- ⚙️  Encoding rendered images for the batch... ---")
        batch_processed_count = 0
        for model_path, image_dir in rendered_data.items():
            try:
                descriptor = encode.create_descriptor_from_images(image_dir, 'dinov2', model_assets)
                if descriptor is not None:
                    output_path = os.path.join(os.path.dirname(model_path), args.output_filename)
                    np.save(output_path, descriptor)
                    uid = os.path.basename(os.path.dirname(model_path))
                    # A more concise log message
                    # print(f"Saved: {uid}") 
                    batch_processed_count += 1
                else:
                    print(f"⚠️ Failed to generate descriptor for {uid}, skipping.")
            except Exception as e:
                uid = os.path.basename(os.path.dirname(model_path))
                print(f"❌ Encoding error for {uid}: {e}")
            finally:
                # IMPORTANT: Clean up the image directory immediately after use
                shutil.rmtree(image_dir)
        
        total_processed_count += batch_processed_count
        print(f"--- ✅ Batch {i+1} complete. Processed {batch_processed_count} models. ---")

    # --- 5. Final Summary ---
    total_time = time.time() - overall_start_time
    print("\n" + "✨" * 20)
    print("✨ All Batches Complete! ✨")
    print("✨" * 20)
    print(f"Successfully generated features for {total_processed_count} / {len(all_model_paths)} models.")
    print(f"Total time elapsed: {total_time:.2f} seconds ({total_time/3600:.2f} hours).")