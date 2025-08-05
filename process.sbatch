#!/bin/bash
#SBATCH --job-name=dino_array_extraction
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16         # Requesting 16 CPUs for the workers
#SBATCH --mem=32G                  # Memory for the main process and workers
#SBATCH --time=8:00:00              # Adjust time for processing 1/100th of the data
#SBATCH --gres=gpu:1                 # Request one GPU per task
#SBATCH --constraint=gpu40           # Request a node with a 40GB A100 GPU

# --- MODIFIED: Define the job array to run 100 parallel tasks ---
#SBATCH --array=1-100

# --- MODIFIED: Use array variables in log filenames to keep them separate ---
# %A is the Job ID, %a is the Array Task ID
#SBATCH --output=logs/dino_extraction_%A_%a.out
#SBATCH --error=logs/dino_extraction_%A_%a.err

# --- Environment Setup ---
# Create the logs directory before submitting: mkdir -p logs
# Make sure to activate your project's Python environment
# For example:
# module load anaconda3
# source activate your_conda_env

echo "--- Starting SLURM Job ---"
echo "Job ID: $SLURM_JOB_ID | Array Task ID: $SLURM_ARRAY_TASK_ID / $SLURM_ARRAY_TASK_COUNT"
echo "Running on host: $(hostname)"
echo "Allocated CPUs: $SLURM_CPUS_PER_TASK"
echo "-----------------------------------------------------"

# --- Define Data and Script Paths ---
INPUT_DIR="/scratch/gpfs/ZHUANGL/bz1474/Objaverse/dataset/Trellis500K"
OUTPUT_DIR="/scratch/gpfs/bz1474/sp2526/Dinov2-LFD"
SCRIPT_PATH="process_objaverse.py"

# --- Execute the Python Script ---
# This passes the array task ID to the script so it knows which chunk of data to process.
# The number of workers is dynamically set to match the allocated CPUs.
python -u $SCRIPT_PATH \
    --input_dir $INPUT_DIR \
    --output_dir $OUTPUT_DIR \
    --task_id $SLURM_ARRAY_TASK_ID \
    --num_tasks $SLURM_ARRAY_TASK_COUNT \
    --views 50 \

echo "--- Task $SLURM_ARRAY_TASK_ID Finished ---"