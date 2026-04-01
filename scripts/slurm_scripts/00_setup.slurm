#!/bin/bash

# General job information and output files
#SBATCH --job-name=emoset_setup
#SBATCH --output=/home/gmago/Emotions/outputs/slurm_out/emoset_setup_%j.txt
#SBATCH --error=/home/gmago/Emotions/outputs/slurm_error/emoset_setup_%j.txt
#SBATCH --partition=hava
#SBATCH --account=havausers

# Job resource information
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=5-00:00:00
#SBATCH --mem=8G
#SBATCH --nodes=1

# GPU resource information
#SBATCH --gpus=1
#SBATCH --ntasks-per-gpu=1

# Activate environment
source "$PYTHON_ENV"

# Set PYTHONPATH (add both meru and emoset directories)
export PYTHONPATH="/home/gmago/Emotions/code/meru:/home/gmago/Emotions/code/meru/emoset:${PYTHONPATH:-}"

# Configuration
export EMOSET_ROOT="/ivi/zfs/s0/original_homes/gmago/emoset"

# Run setup
bash /home/gmago/Emotions/code/meru/scripts/emotion_experiments/00_setup.sh
