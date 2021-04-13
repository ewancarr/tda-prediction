#!/bin/bash
#SBATCH --partition=brc
#SBATCH --tasks=4
#SBATCH --mem=4000
#SBATCH --job-name=array
#SBATCH --array=720-790
#SBATCH --output=/scratch/users/%u/logs/%a.out
#SBATCH --time=0-72:00
module load apps/singularity
cd /scratch/users/k1644956/tda/prediction/v2/

singularity exec containers/python.simg \
	python growth_curves/run_single_gc.py $SLURM_ARRAY_TASK_ID
