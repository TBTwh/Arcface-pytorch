#!/bin/bash
#SBATCH -J lfw  #slurm job name
#SBATCH -t 72:00:00
#SBATCH -A zbwang
#SBATCH -p gpu --gres=gpu:1
#SBATCH -N 1 -c 4

source activate pytorch36

python3 train.py


