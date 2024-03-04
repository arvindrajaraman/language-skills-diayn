#!/bin/bash
#SBATCH --job-name=lunarlander_s3
#SBATCH --cpus-per-task=1
#SBATCH --mem=1gb
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --qos=default
#SBATCH --output=logs/my-job-%j.txt

cd ..

srun conda run -n diayn python diayn_mlp_sweep_agent.py -s afnz7im6
