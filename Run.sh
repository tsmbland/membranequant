#!/usr/bin/env bash

#SBATCH --time=0
#SBATCH --cpus-per-task=16

module load Python/3.5.2-foss-2016b

srun python -m Experiments.e1802__par6_rundown_nelio


