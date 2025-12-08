#!/bin/bash

# Set number of parallel jobs
JOBS=8

# Dataset sizes
TRAIN_N=512
VAL_N=64
TEST_N=64
PE_COUNT=5

# Define data types
DATA_TYPES=("train" "validation" "test")

for DATA_TYPE in "${DATA_TYPES[@]}"; do
    echo "Processing $DATA_TYPE dispersion samples..."
    
    # Determine N for this dataset
    case $DATA_TYPE in
        train) N=$TRAIN_N ;;
        validation) N=$VAL_N ;;
        test) N=$TEST_N ;;
    esac
    
    # Initialize dataset if needed
    if [[ ! -d "./data/${DATA_TYPE}.zarr/dispersion_results" ]]; then
        echo "Initializing dataset ${DATA_TYPE}.zarr"
        python3 data/init_zarr_dataset.py "$DATA_TYPE" "dispersion"
    fi
    
    # Generate list of (sample_idx, pe_idx) pairs that need to be run
    TASK_LIST=$(python3 - <<END
import zarr
import numpy as np

root = zarr.open(f"./data/${DATA_TYPE}.zarr", mode="a")
Dx = root['dispersion_results']['Dx']

tasks = []
for sample_idx in range($N):
    for pe_idx in range($PE_COUNT):
        val = Dx[sample_idx, pe_idx, 0, 0]
        if np.isnan(val):
            tasks.append(f"{sample_idx} {pe_idx}")

print("\n".join(tasks))
END
)
    
    if [ -z "$TASK_LIST" ]; then
        echo "All $DATA_TYPE samples already complete. Skipping."
        continue
    fi
    
    # Count tasks
    TASK_COUNT=$(echo "$TASK_LIST" | wc -l)
    echo "Found $TASK_COUNT missing simulations for $DATA_TYPE"
    
    # Run GNU Parallel with live output
    echo "$TASK_LIST" | parallel --jobs $JOBS --line-buffer --colsep ' ' \
        python3 -u run_dispersion_sample.py {1} $DATA_TYPE {2}
    
    echo "Finished $DATA_TYPE dataset"
done

echo "All datasets complete!"