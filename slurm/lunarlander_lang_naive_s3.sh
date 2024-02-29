#!/bin/bash
#SBATCH --job-name=lunarlander_lang_naive_s3
#SBATCH --cpus-per-task=1
#SBATCH --mem=1gb
#SBATCH --gpus=1
#SBATCH --time=03:00:00
#SBATCH --qos=high
#SBATCH --output=logs/my-job-%j.txt

cd ..

srun conda run -n diayn python diayn_mlp.py -c lunarlander_lang_naive_s3.yml
