import os
import time
import argparse
import shutil
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

# Import the refactored functions from your encode.py module
import encode

# --- MODIFIED ---: The function now takes task_id and num_tasks to divide the workload
def find_model_paths(input_dir, output_dir, output_filename, resume, task_id, num_tasks):
    """
    Scans for all models and returns a unique chunk for the given task ID.
    """
    print(f"TASK {task_id}/{num_tasks}: 🔍 Scanning for all models in '{input_dir}'...")
    
    all_found_paths = []
    # First, gather all potential models without checking the resume status
    for root, dirs, files in os.walk(input_dir):
        if "model_normalized.glb" in files:
            all_found_paths.append(os.path.join(root, "model_normalized.glb"))

    print(f"TASK {task_id}/{num_tasks}: Found {len(all_found_paths)} total models in source.")

    # Now, select a unique chunk for this specific array task
    chunk_paths = []
    for i, model_path in enumerate(all_found_paths):
        # This math assigns every Nth file to this task
        if i % num_tasks == (task_id - 1):
            chunk_paths.append(model_path)

    print(f"TASK {task_id}/{num_tasks}: This task is responsible for {len(chunk_paths)} models.")

    # If resuming, filter out already completed models from this task's chunk
    if not resume:
        print(f"TASK {task_id}/{num_tasks}: Resume is disabled, processing all {len(chunk_paths)} models in its chunk.")
        return chunk_paths
    
    model_paths_to_process = []
    skipped_count = 0
    for model_path in chunk_paths:
        uid = os.path.basename(os.path.dirname(model_path))
        output_path = os.path.join(output_dir, uid, output_filename)
        if os.path.exists(output_path):
            skipped_count += 1
            continue
        model_paths_to_process.append(model_path)

    print(f"TASK {task_id}/{num_tasks}: Skipping {skipped_count} completed models.")
    print(f"TASK {task_id}/{num_tasks}: ➡️ Found {len(model_paths_to_process)} new models to process.")
    
    return model_paths_to_process

if __name__ == '__main__':
    # --- 1. Configuration ---
    parser = argparse.ArgumentParser(
        description="Process a chunk of a 3D model dataset to generate DINOv2 features."
    )
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    
    # --- ADDED ---: Arguments to accept SLURM array variables
    parser.add_argument('--task_id', type=int, required=True, help="SLURM array task ID")
    parser.add_argument('--num_tasks', type=int, required=True, help="Total number of SLURM array tasks")

    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--views', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--output_filename', type=str, default='dinov2_feature.npy')
    parser.add_argument('--no-resume', dest='resume', action='store_false')
    
    args = parser.parse_args()

    # --- 2. Discover This Task's Models ---
    # --- MODIFIED ---: Pass the task arguments to the discovery function
    all_model_paths = find_model_paths(
        args.input_dir, args.output_dir, args.output_filename, args.resume, 
        args.task_id, args.num_tasks
    )
    
    if not all_model_paths:
        print(f"✅ TASK {args.task_id}/{args.num_tasks}: No new models to process. Exiting.")
        exit(0)

    # The rest of the script (loading model, processing batches) remains the same
    # ... (code from Step 3 onward is unchanged) ...
    # --- 3. Load AI Model Once ---
    print(f"\n--- 🧠 TASK {args.task_id}: Loading DINOv2 model into VRAM... ---")
    model_assets = encode.load_dinov2_model()
    print(f"--- ✅ TASK {args.task_id}: DINOv2 model loaded. ---\n")
    
    # --- 4. Process All Models in Batches ---
    overall_start_time = time.time()
    total_processed_count = 0
    num_batches = (len(all_model_paths) + args.batch_size - 1) // args.batch_size

    for i in range(num_batches):
        batch_start_idx = i * args.batch_size
        batch_end_idx = batch_start_idx + args.batch_size
        batch_paths = all_model_paths[batch_start_idx:batch_end_idx]

        print("─" * 80)
        print(f"📦 TASK {args.task_id}: Processing Batch {i+1} / {num_batches}")
        
        # --- PHASE 1 (BATCH): PARALLEL RENDERING ---
        print(f"--- 🚀 TASK {args.task_id}: Starting Rendering with {args.workers} workers... ---")
        batch_render_start_time = time.time()
        camera_views = encode.generate_spherical_views(num_views=args.views)
        rendered_data = {}

        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_to_path = {
                executor.submit(encode.render_views_to_tempdir, path, camera_views): path
                for path in batch_paths
            }
            for future in as_completed(future_to_path):
                original_path = future_to_path[future]
                try:
                    image_dir = future.result()
                    if image_dir: rendered_data[original_path] = image_dir
                except Exception as exc:
                    uid = os.path.basename(os.path.dirname(original_path))
                    print(f"❌ Render error for {uid}: {exc}")
        
        print(f"--- ✅ TASK {args.task_id}: Rendering for batch complete in {time.time() - batch_render_start_time:.2f}s. ---")

        # --- PHASE 2 (BATCH): SEQUENTIAL ENCODING & CLEANUP ---
        if not rendered_data:
            print(f"⚠️ TASK {args.task_id}: No models were successfully rendered in this batch. Skipping.")
            continue
        
        print(f"--- ⚙️ TASK {args.task_id}: Encoding rendered images for the batch... ---")
        batch_processed_count = 0
        for model_path, image_dir in rendered_data.items():
            uid = os.path.basename(os.path.dirname(model_path))
            try:
                descriptor = encode.create_descriptor_from_images(image_dir, 'dinov2', model_assets)
                if descriptor is not None:
                    output_folder = os.path.join(args.output_dir, uid)
                    os.makedirs(output_folder, exist_ok=True)
                    output_path = os.path.join(output_folder, args.output_filename)
                    np.save(output_path, descriptor)
                    batch_processed_count += 1
                else:
                    print(f"⚠️ Failed to generate descriptor for {uid}, skipping.")
            except Exception as e:
                print(f"❌ Encoding error for {uid}: {e}")
            finally:
                if os.path.exists(image_dir):
                    shutil.rmtree(image_dir)
        
        total_processed_count += batch_processed_count
        print(f"--- ✅ TASK {args.task_id}: Batch {i+1} complete. Processed {batch_processed_count} models. ---")

    # --- 5. Final Summary ---
    total_time = time.time() - overall_start_time
    print("\n" + "✨" * 20)
    print(f"✨ TASK {args.task_id}: All Batches Complete! ✨")
    print(f"Successfully processed {total_processed_count} models in this task.")
    print(f"Total time elapsed for this task: {total_time:.2f} seconds.")