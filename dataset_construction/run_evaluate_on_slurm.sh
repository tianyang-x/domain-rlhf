#!/bin/bash
#SBATCH --job-name=evaluate_on_eurus_2
#SBATCH --time=4-00:00:00
#SBATCH --gpus-per-node=2
#SBATCH --mem=128G
#SBATCH --account=lusu-k

conda activate domain_rlhf

python3 evaluate_on_eurus_2.py --tensor_parallel_size 2