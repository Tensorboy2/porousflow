#!/bin/bash

# Set number of parallel jobs (8 physical cores)
JOBS=16

# Define data types
DATA_TYPES=("validation" "test")

for DATA_TYPE in "${DATA_TYPES[@]}"; do
    echo "Processing $DATA_TYPE dispersion samples..."

    DATA_PATH="./data/media_samples_${DATA_TYPE}/media_samples.npz"
    LBM_PATH="./data/lbm_simulation_results_${DATA_TYPE}"
    SAVE_PATH="./data/self_diffusivity_results_${DATA_TYPE}"

    # Create save folder if missing
    mkdir -p $SAVE_PATH

    # Get indices of samples that don't exist yet
    INDICES=$(python3 - <<END
import numpy as np, os
data = np.load("$DATA_PATH")
filled_images = data["filled_images"]
save_path = "$SAVE_PATH"
indices = [i for i in range(filled_images.shape[0]) if not os.path.exists(f"{save_path}/simulation_result_{i}.npz")]
print(" ".join(map(str, indices)))
END
)

    if [ -z "$INDICES" ]; then
        echo "All $DATA_TYPE samples already done. Skipping."
        continue
    fi

    # Run GNU Parallel with live output
    parallel --jobs $JOBS --line-buffer python3 -u effective_self_diffusivity.py {} $DATA_TYPE ::: $INDICES
done
