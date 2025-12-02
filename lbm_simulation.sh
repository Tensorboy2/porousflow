#!/bin/bash

# Set number of parallel jobs (8 physical cores)
JOBS=8

# Define data types
DATA_TYPES=("train" "validation" "test")

for DATA_TYPE in "${DATA_TYPES[@]}"; do
    echo "Processing $DATA_TYPE samples..."

    DATA_PATH="./data/${DATA_TYPE}.zarr/filled_images/filled_images"
    SAVE_PATH="./data/${DATA_TYPE}"

    if [[ -d "./data/${DATA_TYPE}.zarr/lbm_results/K" ]]; then
        echo "Data initet"
    else
        python3 data/init_zarr_dataset.py $DATA_TYPE "lbm" 
    fi


    # Get indices of samples that don't exist yet
    INDICES=$(python3 - <<END
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
    parallel --jobs $JOBS --line-buffer python3 -u run_lbm_sample.py {} $DATA_TYPE ::: $INDICES
done
