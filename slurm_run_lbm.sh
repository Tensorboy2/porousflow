#!/bin/bash
#SBATCH --job-name=lbm_parallel
#SBATCH --output=slurm_out/lbm_%j.out
#SBATCH --error=slurm_out/lbm_%j.err
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G

# Make sure the output directory exists
mkdir -p slurm_out

# Move to your repo directory
cd ~/porousflow

# Activate your venv
source ~/porousflow/bin/activate

# Set number of parallel jobs to the number of CPUs allocated by SLURM
JOBS=$SLURM_CPUS_PER_TASK

# Define data types
DATA_TYPES=("train" "validation" "test")

for DATA_TYPE in "${DATA_TYPES[@]}"; do
    echo "Processing $DATA_TYPE samples..."

    DATA_PATH="./data/${DATA_TYPE}.zarr/filled_images/filled_images"
    SAVE_PATH="./data/${DATA_TYPE}"

    if [[ -d "./data/${DATA_TYPE}.zarr/lbm_results/K" ]]; then
        echo "Data initialized"
    else
        python data/init_zarr_dataset.py $DATA_TYPE "lbm"
    fi

    # Get indices of samples that don't exist yet
    INDICES=$(python - <<END
import zarr, numpy as np
store_path = "./data/${DATA_TYPE}.zarr"
root = zarr.open(store_path, mode="a")
k_ds = root['lbm_results']['K']
indices = [i for i in range(k_ds.shape[0]) if np.isnan(k_ds[i, 0, 0])]
print(" ".join(map(str, indices)))
END
)

    if [ -z "$INDICES" ]; then
        echo "All $DATA_TYPE samples already done. Skipping."
        continue
    fi

    # Run GNU Parallel
    parallel --jobs $JOBS --line-buffer python -u run_lbm_sample.py {} $DATA_TYPE ::: $INDICES
done