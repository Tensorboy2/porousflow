#!/bin/bash

# Set number of parallel jobs (8 physical cores)
JOBS=8

# Define data types
DATA_TYPES=("train" "validation" "test")

for DATA_TYPE in "${DATA_TYPES[@]}"; do
    echo "Processing $DATA_TYPE dispersion samples..."

    DATA_PATH="./data/media_samples_${DATA_TYPE}/media_samples.npz"
    LBM_PATH="./data/lbm_simulation_results_${DATA_TYPE}"
    SAVE_PATH="./data/${DATA_TYPE}"

    if [[ -d "./data/${DATA_TYPE}.zarr/dispersion_results/Dx" ]]; then
        echo "Data initet"
    else
        python3 data/init_zarr_dataset.py $DATA_TYPE "dispersion" 
    fi

    # Get indices of samples that don't exist yet
    INDICES=$(python3 - <<END
import zarr, numpy as np
store_path = "./data/${DATA_TYPE}.zarr"
root = zarr.open(store_path, mode="a")
Dx_ds = root['dispersion_results']['Dx']
indices = [i for i in range(Dx_ds.shape[0]) if np.isnan(Dx_ds[i, 0, 0, 0])]
print(" ".join(map(str, indices)))
END
)

    if [ -z "$INDICES" ]; then
        echo "All $DATA_TYPE samples already done. Skipping."
        continue
    fi

    # Run GNU Parallel with live output
    parallel --jobs $JOBS --line-buffer python3 -u run_dispersion_sample.py {1} $DATA_TYPE {2} ::: $INDICES ::: {0..4}
done
