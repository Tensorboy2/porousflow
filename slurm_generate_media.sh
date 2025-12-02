#!/bin/bash
#SBATCH --job-name=generate_porous_media
#SBATCH --output=slurm_out/make_media_%j.out
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

echo "Fetching venv"
source ~/porousflow/bin/activate
echo "Generating media..."
python3 generate_media.py