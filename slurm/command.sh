#!/bin/bash
#SBATCH --job-name=crafter_dqn
#SBATCH --time=12:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8gb
#SBATCH --qos=medium
#SBATCH --output=slurm/1716228096-out.txt
#SBATCH --error=slurm/1716228096-err.txt

TOTAL_MEMORY=49140
TARGET_MEMORY=3686
FRACTION=0.07501017501017501
echo "Allocating $FRACTION of GPU ($TARGET_MEMORY MiB out of $TOTAL_MEMORY MiB)"
export XLA_PYTHON_CLIENT_MEM_FRACTION=$FRACTION

srun conda run --no-capture-output -n diayn4 python diayn_fast.py -c paper/diayn_lang.yml
