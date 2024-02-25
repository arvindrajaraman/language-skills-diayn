#!/bin/bash
#SBATCH --job-name=lunarlander_s2
#SBATCH --cpus-per-task=1
#SBATCH --mem=1gb
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --qos=high

cd ..

srun conda run -n diayn python diayn_mlp.py -c lunarlander_s2.yml
