import os
import time
import argparse
import shutil
import numpy as np
import multiprocessing as mp
mp.set_start_method("spawn", force=True)
from concurrent.futures import ProcessPoolExecutor, as_completed

# Import the refactored functions from your encode.py module
import encode

def find_model_paths(input_dir, output_dir, output_filename, resume, task_id, num_tasks):
    """
    Scans for all ShapeNet models and returns a unique chunk for the given task ID.
    This version is adapted for the ShapeNetCore structure: {input_dir}/{category_uid}/{model_uid}/...
    """
    print(f"TASK {task_id}/{num_tasks}: 🔍 Scanning for all models in '{input_dir}'...")

    # Stores tuples of (full_model_path, relative_path_for_output)
    all_found_models = []
    # --- MODIFIED ---: Search for '.obj' files and preserve the category/model structure.
    search_file = "model_normalized.obj"
    
    for root, dirs, files in os.walk(input_dir):
        if search_file in files:
            full_path = os.path.join(root, search_file)
            # This captures the '{category_uid}/{model_uid}' part of the path
            relative_path = os.path.relpath(root, input_dir)
            all_found_models.append((full_path, relative_path))

    print(f"TASK {task_id}/{num_tasks}: Found {len(all_found_models)} total models in source.")

    # Distribute the models among the tasks
    chunk_models = []
    for i, model_info in enumerate(all_found_models):
        if i % num_tasks == (task_id - 1):
            chunk_models.append(model_info)

    print(f"TASK {task_id}/{num_tasks}: This task is responsible for {len(chunk_models)} models.")

    if not resume:
        print(f"TASK {task_id}/{num_tasks}: Resume is disabled, processing all {len(chunk_models)} models in its chunk.")
        return chunk_models

    # If resuming, filter out already processed models
    models_to_process = []
    skipped_count = 0
    for full_path, relative_path in chunk_models:
        # --- MODIFIED ---: Construct the output path using the relative path
        output_path = os.path.join(output_dir, relative_path, output_filename)
        if os.path.exists(output_path):
            skipped_count += 1
            continue
        models_to_process.append((full_path, relative_path))

    print(f"TASK {task_id}/{num_tasks}: Skipping {skipped_count} completed models.")
    print(f"TASK {task_id}/{num_tasks}: ➡️ Found {len(models_to_process)} new models to process.")

    return models_to_process

if __name__ == '__main__':
    # --- 1. Configuration ---
    parser = argparse.ArgumentParser(
        description="Process a chunk of a ShapeNet dataset to generate image-based features."
    )
    # --- MODIFIED ---: Renamed for clarity, but function is the same.
    parser.add_argument('--input_dir', type=str, required=True, help="Root directory of the ShapeNet dataset.")
    parser.add_argument('--output_dir', type=str, required=True, help="Root directory to save feature files.")
    
    parser.add_argument(
        '--model_name', 
        type=str, 
        required=True, 
        choices=['dinov2', 'inceptionv3', 'dinov1', 'clip', 'sscd', 'sam2'],
        help="Name of the feature model to use."
    )
    parser.add_argument('--output_filename', type=str, default=None, help="Optional: override the default output filename.")

    # --- SLURM arguments ---
    parser.add_argument('--task_id', type=int, required=True, help="SLURM array task ID")
    parser.add_argument('--num_tasks', type=int, required=True, help="Total number of SLURM array tasks")

    # --- Processing arguments ---
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--views', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--no-resume', dest='resume', action='store_false')

    args = parser.parse_args()
    
    if args.output_filename is None:
        output_filename = f"{args.model_name}_feature.npy"
    else:
        output_filename = args.output_filename
    print(f"Using model '{args.model_name}'. Output files will be named '{output_filename}'.")

    # --- 2. Discover This Task's Models ---
    models_to_process = find_model_paths(
        args.input_dir, args.output_dir, output_filename, args.resume,
        args.task_id, args.num_tasks
    )

    if not models_to_process:
        print(f"✅ TASK {args.task_id}/{args.num_tasks}: No new models to process. Exiting.")
        exit(0)

    # --- 3. Load AI Model Once ---
    print(f"\n--- 🧠 TASK {args.task_id}: Loading {args.model_name.upper()} model into VRAM... ---")
    
    if args.model_name == 'dinov2':
        model_assets = encode.load_dinov2_model()
    elif args.model_name == 'inceptionv3':
        model_assets = encode.load_inceptionv3_model()
    elif args.model_name == 'dinov1':
        model_assets = encode.load_dinov1_model()
    elif args.model_name == 'clip':
        model_assets = encode.load_clip_model()
    # Add other models here if needed
    # elif args.model_name == 'sam2': ...
    else:
        raise ValueError(f"Model '{args.model_name}' is not supported by this script.")

    print(f"--- ✅ TASK {args.task_id}: {args.model_name.upper()} model loaded. ---\n")

    # --- 4. Process All Models in Batches ---
    overall_start_time = time.time()
    total_processed_count = 0
    num_batches = (len(models_to_process) + args.batch_size - 1) // args.batch_size

    for i in range(num_batches):
        batch_start_idx = i * args.batch_size
        batch_end_idx = batch_start_idx + args.batch_size
        batch_model_info = models_to_process[batch_start_idx:batch_end_idx]

        print("─" * 80)
        print(f"📦 TASK {args.task_id}: Processing Batch {i+1} / {num_batches}")

        # --- PHASE 1 (BATCH): PARALLEL RENDERING ---
        print(f"--- 🚀 TASK {args.task_id}: Starting Rendering with {args.workers} workers... ---")
        batch_render_start_time = time.time()
        camera_views = encode.generate_spherical_views(num_views=args.views)
        rendered_data = {}

        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            # --- MODIFIED ---: Submit the full path for rendering, but keep track of the tuple.
            future_to_model_info = {
                executor.submit(encode.render_views_to_tempdir, full_path, camera_views): (full_path, rel_path)
                for full_path, rel_path in batch_model_info
            }
            for future in as_completed(future_to_model_info):
                original_model_info = future_to_model_info[future]
                try:
                    image_dir = future.result()
                    if image_dir: rendered_data[original_model_info] = image_dir
                except Exception as exc:
                    # Log using the relative path for easy identification
                    print(f"❌ Render error for {original_model_info[1]}: {exc}")

        print(f"--- ✅ TASK {args.task_id}: Rendering for batch complete in {time.time() - batch_render_start_time:.2f}s. ---")

        # --- PHASE 2 (BATCH): SEQUENTIAL ENCODING & CLEANUP ---
        if not rendered_data:
            print(f"⚠️ TASK {args.task_id}: No models were successfully rendered in this batch. Skipping.")
            continue

        print(f"--- ⚙️ TASK {args.task_id}: Encoding rendered images for the batch with {args.model_name.upper()}... ---")
        batch_processed_count = 0
        # --- MODIFIED ---: Unpack the tuple when iterating
        for (model_path, relative_path), image_dir in rendered_data.items():
            try:
                descriptor = encode.create_descriptor_from_images(
                    image_dir, args.model_name, model_assets
                )
                if descriptor is not None:
                    # --- MODIFIED ---: Create the full category/model output path
                    output_folder = os.path.join(args.output_dir, relative_path)
                    os.makedirs(output_folder, exist_ok=True)
                    output_path = os.path.join(output_folder, output_filename)
                    np.save(output_path, descriptor)
                    batch_processed_count += 1
                else:
                    print(f"⚠️ Failed to generate descriptor for {relative_path}, skipping.")
            except Exception as e:
                print(f"❌ Encoding error for {relative_path}: {e}")
            finally:
                if os.path.exists(image_dir):
                    shutil.rmtree(image_dir)

        total_processed_count += batch_processed_count
        print(f"--- ✅ TASK {args.task_id}: Batch {i+1} complete. Processed {batch_processed_count} models. ---")

    # --- 5. Final Summary ---
    total_time = time.time() - overall_start_time
    print("\n" + "✨" * 20)
    print(f"✨ TASK {args.task_id}: All Batches Complete! ✨")
    print(f"Successfully processed {total_processed_count} models in this task using {args.model_name.upper()}.")
    print(f"Total time elapsed for this task: {total_time:.2f} seconds.")