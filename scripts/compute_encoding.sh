#!/bin/bash

##############################################################################
# User Request:
#   - request an exclusive node
#   - run a single program
#
# Provided Allocation:
#   - exclusive node
#   - 48 physical cores / 96 logical cores
#   - 96 GB memory
#   - not used: 48 tasks on 1 node
#   - not used: 1 physical cores bound to each task
#
# VSC policy:
#   - shared=0 -> exclusive node access
#   - ntasks=48 (to request all 48 physical cores of the node; 1 physical core per task)
#   
# Accounting:
#   - 48 core hours / hour
##############################################################################

#SBATCH --job-name=lpca-peptides-enc
#SBATCH --nodes=1
#SBATCH --partition=skylake_0096
#SBATCH --qos=skylake_0096

module purge

module load miniconda3
eval "$(conda shell.bash hook)"

conda activate thesis

python lpca_with_sim_runner.py Peptides 4 20 0.5
