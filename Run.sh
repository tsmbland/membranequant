#!/usr/bin/env bash
#SBATCH --array=0-1

module load Python/3.5.2-foss-2016b

srun -n 400 python -m Experiments.Analysis_cluster


