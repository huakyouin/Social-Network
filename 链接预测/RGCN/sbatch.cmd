#!/bin/bash

#SBATCH --job-name=RGCN
#SBATCH --output=RGCN_wn18.log
#SBATCH --gpus=0
#SBATCH --cpus-per-task=16
#SBATCH --nodelist=bigMem1

python main.py --evaluate-every 1000 --n-epochs 30000 --n-bases 2 --gpu 0
