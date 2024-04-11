#!/bin/bash
#SBATCH --job-name=lunar_1h_sweep
#SBATCH --time=24:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4gb
#SBATCH --qos=default
#SBATCH --output=slurm/1712856705-out.txt
#SBATCH --error=slurm/1712856705-err.txt

TOTAL_MEMORY=49140
TARGET_MEMORY=3686
FRACTION=0.07501017501017501
echo "Allocating $FRACTION of GPU ($TARGET_MEMORY MiB out of $TOTAL_MEMORY MiB)"
export XLA_PYTHON_CLIENT_MEM_FRACTION=$FRACTION

srun conda run --no-capture-output -n diayn python sweep.py -i flg27f9g
