#!/bin/bash
#SBATCH --job-name=dispersion_array
#SBATCH --output=slurm_out/disp_%A_%a.out
#SBATCH --error=slurm_out/disp_%A_%a.err
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --array=0-1000%72     # MaxArraySize limit

# ---------------------------------------------------------
# Dataset sizes
TRAIN_N=16000
VAL_N=4000
TEST_N=4000
TOTAL=$((TRAIN_N + VAL_N + TEST_N))   # 24000
PE_COUNT=5
# ---------------------------------------------------------

# Optional test mode: set TEST=1 when submitting to only parse/simulate
TEST=${TEST:-0}

mkdir -p slurm_out
cd ~/porousflow || { echo "Cannot cd to ~/porousflow"; exit 1; }

# Activate virtualenv
VENV_PATH=~/porousflow_venv
if [[ -f "$VENV_PATH/bin/activate" ]]; then
    source "$VENV_PATH/bin/activate"
else
    echo "Virtualenv not found at $VENV_PATH"
    exit 1
fi

# Show python info for debugging
echo "Python: $(which python) $(python --version)"

# ------------- Chunk calculation -------------
ARRAY_MAX=1001
CHUNK=$(( (TOTAL + ARRAY_MAX - 1) / ARRAY_MAX ))  # ceil(TOTAL / ARRAY_MAX)
START=$(( SLURM_ARRAY_TASK_ID * CHUNK ))
END=$(( START + CHUNK - 1 ))
if [ "$END" -ge "$TOTAL" ]; then
    END=$(( TOTAL - 1 ))
fi

echo "Array $SLURM_ARRAY_TASK_ID handling samples $START → $END"

# ------------- Loop through assigned samples -------------
for SAMPLE_IDX in $(seq $START $END); do

    # Test mode: skip real work
    if [[ "$TEST" == "1" ]]; then
        echo "TEST mode: skipping sample $SAMPLE_IDX"
        continue
    fi

    # Determine dataset type
    if [ "$SAMPLE_IDX" -lt "$TRAIN_N" ]; then
        DATA_TYPE="train"
        SAMPLE_LOCAL=$SAMPLE_IDX
    elif [ "$SAMPLE_IDX" -lt $((TRAIN_N + VAL_N)) ]; then
        DATA_TYPE="validation"
        SAMPLE_LOCAL=$(( SAMPLE_IDX - TRAIN_N ))
    else
        DATA_TYPE="test"
        SAMPLE_LOCAL=$(( SAMPLE_IDX - TRAIN_N - VAL_N ))
    fi

    echo "Processing sample $SAMPLE_LOCAL in dataset $DATA_TYPE"

    # Ensure Zarr dataset exists
    if [[ ! -d "./data/${DATA_TYPE}.zarr" ]]; then
        echo "Initializing dataset ${DATA_TYPE}.zarr"
        python data/init_zarr_dataset.py "$DATA_TYPE" "dispersion" || { echo "Failed to init dataset"; exit 1; }
    fi

    # Loop through Pe values for this sample
    for PE_IDX in $(seq 0 $((PE_COUNT-1))); do

        # Safe Python check
        SHOULD_RUN=$(python - <<END
import zarr, numpy as np, sys
try:
    root = zarr.open(f"./data/${DATA_TYPE}.zarr", mode="a")
    Dx = root['dispersion_results']['Dx']
    val = Dx[$SAMPLE_LOCAL, $PE_IDX, 0, 0]
    print(1 if np.isnan(val) else 0)
except Exception as e:
    sys.stderr.write(f"Error checking sample {SAMPLE_LOCAL} Pe {PE_IDX}: {e}\n")
    print(0)
END
)

        # Default to 0 if Python fails or returns empty
        SHOULD_RUN="${SHOULD_RUN:-0}"

        if [[ "$SHOULD_RUN" -eq 1 ]]; then
            echo "Pe $PE_IDX missing → running"
            python -u run_dispersion_sample.py "$SAMPLE_LOCAL" "$DATA_TYPE" "$PE_IDX"
        else
            echo "Pe $PE_IDX already complete or error → skipping"
        fi

    done

done

echo "Array task $SLURM_ARRAY_TASK_ID complete."
