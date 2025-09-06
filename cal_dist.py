#!/usr/bin/env python3
"""
Fast, parallel 3D model retrieval using pre-computed feature databases.

This script is optimized for multiple metric types:
- Single-Vector (Uni3D, ULIP): Uses fast batched cosine similarity.
- Multi-View (DINOv2, InceptionV3): Uses Chamfer Distance to compare sets of
  view-based features. The calculation is GPU-accelerated with chunked GEMMs.
- LFD: Uses a custom, pre-computed distance handler.

Usage:
  python retrieve_chamfer.py --query_list queries.txt --metric dinov2 \
    --precomputed_db_dir /path/to/db --output dinov2_results.json \
    --gpu_ids 0,1
"""

import argparse
import json
import time
from pathlib import Path
from typing import List
import heapq
import tempfile

import numpy as np
import torch
from tqdm import tqdm

# --- Assumed project utilities ---
from utils.batched_cos import batched_cosine_similarity
from lfd_metric_fast import FastLFDMetric

# --- Use torch's multiprocessing for CUDA context safety ---
try:
    mp = torch.multiprocessing.get_context("spawn")
    print("[INFO] Using torch.multiprocessing with 'spawn' context.")
except ImportError:
    import multiprocessing as mp
    print("[WARN] torch.multiprocessing not found. Falling back to standard multiprocessing.")

# --- Define supported metrics ---
SINGLE_VECTOR_METRICS = {"uni3d", "ulip"}
MULTI_VIEW_METRICS = {"dinov2", "inceptionv3"}
SUPPORTED_METRICS = SINGLE_VECTOR_METRICS | MULTI_VIEW_METRICS | {"lfd"}


# ===================================================================
# Utilities
# ===================================================================

def torch_normalize_features(features: torch.Tensor) -> torch.Tensor:
    """L2-normalizes a tensor of feature vectors."""
    return features / features.norm(dim=-1, keepdim=True).clamp_min_(1e-12)


@torch.no_grad()
def chamfer_to_all_models_chunked(
    q_desc: torch.Tensor,             # (m, D), L2-normalized on device
    db_desc: torch.Tensor,            # (N, D), L2-normalized (CPU or same device)
    offsets: List[dict],              # [{'start': int, 'len': int}, ...] length = num_models
    chunk_size: int,                  # number of DB views per chunk
) -> torch.Tensor:
    """
    Computes mean Chamfer distance from a query (multi-view) to every model in the DB,
    using chunked GEMMs to control memory. Returns a tensor of shape (num_models,)
    with lower = better.

    Chamfer (mean version to avoid size bias):
      CD(A,B) = mean_i min_j d(A_i,B_j) + mean_j min_i d(A_i,B_j)
    where d is cosine distance on L2-normalized features.
    """
    device = q_desc.device
    m = q_desc.shape[0]
    num_models = len(offsets)

    # Accumulators
    # A->B: per-model, for each query row, running min across the DB (store m values per model)
    a2b_rowmins = torch.full((num_models, m), float("inf"), device=device)
    # B->A: per-model, accumulate sum of per-column mins; divide by L at the end
    b2a_sum = torch.zeros(num_models, device=device, dtype=q_desc.dtype)

    N = db_desc.shape[0]

    # Optional speed feature
    torch.backends.cuda.matmul.allow_tf32 = True

    for g0 in range(0, N, chunk_size):
        g1 = min(g0 + chunk_size, N)
        db_chunk = db_desc[g0:g1]
        if db_chunk.device != device:
            db_chunk = db_chunk.to(device, non_blocking=True)

        # (m, D) @ (Nc, D)^T -> (m, Nc)
        sim_chunk = q_desc @ db_chunk.T
        d_chunk = 1.0 - sim_chunk  # cosine distance

        # Update per-model accumulators for models overlapping this chunk
        # Linear scan over offsets is fine for ~60k models.
        for k, of in enumerate(offsets):
            s, L = of["start"], of["len"]
            e = s + L
            if e <= g0 or s >= g1:
                continue  # no overlap with current chunk

            # Overlap range inside [g0, g1)
            lo = max(s, g0)
            hi = min(e, g1)

            # Local indices inside the chunk
            c0 = lo - g0
            c1 = hi - g0

            sub = d_chunk[:, c0:c1]  # (m, L_chunk)

            # A->B term: for each query view (row), min over this slice; running min across chunks
            a2b_rowmins[k] = torch.minimum(a2b_rowmins[k], sub.min(dim=1).values)

            # B->A term: for each slice view (column), min over query rows, then sum into accumulator
            b2a_sum[k] += sub.min(dim=0).values.sum()

        # Free temporaries
        del db_chunk, sim_chunk, d_chunk

    # Finalize means
    a2b = a2b_rowmins.mean(dim=1)
    L_vec = torch.tensor([of["len"] for of in offsets], device=device, dtype=b2a_sum.dtype).clamp_min_(1)
    b2a = b2a_sum / L_vec

    chamfer = a2b + b2a  # lower is better
    return chamfer


def load_queries_from_file(path: Path) -> List[Path]:
    """Reads a text file where each line is a path to a query feature."""
    with open(path) as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    return [Path(p) for p in lines]


# ===================================================================
# Main Worker Function
# ===================================================================

def worker_retrieval(
    proc_id: int,
    gpu_id: int,
    query_chunk: List[Path],
    args: argparse.Namespace,
    temp_dir: Path,
):
    """The core retrieval function that runs on a single GPU process."""
    device = f"cuda:{gpu_id}"
    worker_name = f"[Worker-{proc_id} | GPU:{gpu_id}]"
    print(f"{worker_name} Started. Processing {len(query_chunk)} queries.")

    metric = args.metric
    db_dir = Path(args.precomputed_db_dir)

    # --- Initialize metric-specific components ONCE per worker ---
    metric_handler = None
    db_matrix_path, db_meta_path = None, None
    db_tensor, db_meta, db_offsets = None, None, None

    if metric in SINGLE_VECTOR_METRICS:
        db_matrix_path = db_dir / f"{metric}_db.npy"
        db_meta_path = db_dir / f"{metric}_db_meta.json"
        if not (db_matrix_path.exists() and db_meta_path.exists()):
            raise FileNotFoundError(f"{worker_name} Database files not found for {metric}")

    elif metric in MULTI_VIEW_METRICS:
        db_matrix_path = db_dir / f"{metric}_db_perview.npy"
        db_meta_path = db_dir / f"{metric}_db_perview_meta.json"
        if not (db_matrix_path.exists() and db_meta_path.exists()):
            raise FileNotFoundError(f"{worker_name} Per-view database files not found for {metric}")

        print(f"{worker_name} Loading per-view database (features + meta)...")
        # Load DB features as float32 on CPU first to allow chunked streaming if needed.
        db_np = np.load(db_matrix_path, mmap_mode="r")
        db_tensor = torch.from_numpy(db_np)  # CPU tensor (zero-copy via memmap)
        # L2-normalize once (in chunks to avoid big peak memory), then keep on CPU for streaming
        with torch.no_grad():
            # Normalize in-place by chunks to keep memory modest
            N, D = db_tensor.shape
            step = max(100_000, min(N, args.db_chunk_views))
            for i0 in range(0, N, step):
                i1 = min(i0 + step, N)
                sl = db_tensor[i0:i1]
                sl /= sl.norm(dim=-1, keepdim=True).clamp_min_(1e-12)

        with open(db_meta_path, 'r') as f:
            db_meta = json.load(f)
        db_offsets = db_meta['offsets']
        print(f"{worker_name} DB ready on CPU. Views: {db_tensor.shape[0]}, Dim: {db_tensor.shape[1]}")

    elif metric == "lfd":
        lfd_db_paths = {"meta": db_dir / "lfd_db_meta.json"}  # Simplified for brevity
        if not lfd_db_paths["meta"].exists():
            raise FileNotFoundError(f"{worker_name} LFD pre-computed files not found in {db_dir}")
        metric_handler = FastLFDMetric(lfd_db_paths, device=device)

    # --- Process the assigned chunk of queries ---
    results = {}
    pbar = tqdm(query_chunk, desc=worker_name, position=proc_id, leave=False)
    for q_path in pbar:
        q_path_str = str(q_path)
        if not q_path.exists():
            print(f"{worker_name} [WARN] Query path does not exist, skipping: {q_path_str}")
            results[q_path_str] = []
            continue

        best = []
        try:
            if metric in SINGLE_VECTOR_METRICS:
                q_vec = np.load(q_path)
                # For cosine similarity, higher is better.
                best = batched_cosine_similarity(q_vec, db_matrix_path, db_meta_path, args.topk, device)

            elif metric in MULTI_VIEW_METRICS:
                with torch.no_grad():
                    q_desc = torch.from_numpy(np.load(q_path)).to(device, non_blocking=True).float()
                    q_desc = torch_normalize_features(q_desc)

                    # Chunked GEMM over DB views; DB stays on CPU (pinned) and streams to GPU.
                    # Convert DB to pinned memory for faster H2D copies.
                    if db_tensor.device.type == "cpu":
                        db_tensor_pinned = db_tensor.pin_memory()
                    else:
                        db_tensor_pinned = db_tensor  # already on device (not our default path)

                    chamfers = chamfer_to_all_models_chunked(
                        q_desc=q_desc,
                        db_desc=db_tensor_pinned,
                        offsets=db_offsets,
                        chunk_size=args.db_chunk_views,
                    )  # (num_models,) lower is better

                    # Keep top-k smallest distances
                    # Use torch.topk on negative to get smallest efficiently
                    k = min(args.topk, chamfers.numel())
                    scores, idxs = torch.topk(-chamfers, k)  # largest negative == smallest positive
                    scores = (-scores).tolist()
                    idxs = idxs.tolist()

                    for score, model_idx in zip(scores, idxs):
                        model_info = db_meta['items'][model_idx]
                        best.append({
                            "dataset": model_info["dataset"],
                            "category_id": model_info["category_id"],
                            "model_id": model_info["model_id"],
                            "score": float(score),  # mean-Chamfer distance (lower is better)
                            "relpath": model_info["relpath"]
                        })

            elif metric == "lfd":
                # LFD returns distances, lower is better.
                best = metric_handler.distance(q_path, args.topk)

        except Exception as e:
            print(f"{worker_name} [ERROR] Calculation failed for query {q_path_str}: {e}")
            import traceback
            traceback.print_exc()

        results[q_path_str] = best

    # --- Save partial results ---
    out_path = temp_dir / f"results_{proc_id}.json"
    with open(out_path, "w") as f:
        json.dump(results, f)

    print(f"{worker_name} Finished. Partial results saved.")


def main():
    ap = argparse.ArgumentParser(description="Parallel Fast Retrieval using Chamfer Distance for Multi-View Features.")
    ap.add_argument("--query_list", required=True, help="Path to a .txt file listing query feature paths.")
    ap.add_argument("--metric", choices=SUPPORTED_METRICS, required=True, help="Metric to use for retrieval.")
    ap.add_argument("--precomputed_db_dir", required=True, help="Directory with pre-computed DB files.")
    ap.add_argument("--topk", type=int, default=10, help="Number of top results to save.")
    ap.add_argument("--output", required=True, help="Path for the final output JSON file.")
    ap.add_argument("--gpu_ids", type=str, default="0", help="Comma-separated list of GPU IDs to use (e.g., '0,1,2,3').")
    ap.add_argument("--db_chunk_views", type=int, default=200_000, help="Number of DB views per chunk for multi-view metrics.")
    args = ap.parse_args()

    start_time = time.time()
    out_path = Path(args.output)
    gpu_ids = [int(gid) for gid in args.gpu_ids.split(',')]
    num_gpus = len(gpu_ids)

    print(f"[INFO] Starting parallel retrieval on {num_gpus} GPUs: {gpu_ids}")

    print(f"[INFO] Loading queries from: {args.query_list}")
    all_query_paths = load_queries_from_file(Path(args.query_list))
    if not all_query_paths:
        print("[ERROR] No queries found. Exiting.")
        return
    print(f"[INFO] Found {len(all_query_paths)} total queries.")

    queries_per_gpu = [[] for _ in range(num_gpus)]
    for i, q_path in enumerate(all_query_paths):
        queries_per_gpu[i % num_gpus].append(q_path)

    processes = []
    with tempfile.TemporaryDirectory(prefix="retrieval_") as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        print(f"[INFO] Using temporary directory for partial results: {temp_dir}")

        for i in range(num_gpus):
            if not queries_per_gpu[i]:
                continue
            p = mp.Process(target=worker_retrieval, args=(i, gpu_ids[i], queries_per_gpu[i], args, temp_dir))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

        print("\n[INFO] All workers finished. Merging results...")
        merged_results = {}
        for i in range(num_gpus):
            partial_file = temp_dir / f"results_{i}.json"
            if partial_file.exists():
                with open(partial_file, 'r') as f:
                    merged_results.update(json.load(f))

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(merged_results, f, indent=2)

    total_time = time.time() - start_time
    print(f"\n[✓] Retrieval finished in {total_time:.2f} seconds.")
    print(f"    - Processed {len(all_query_paths)} queries.")
    print(f"    - Final results saved to: {out_path}")


if __name__ == "__main__":
    main()
