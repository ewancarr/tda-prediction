#!/bin/bash
#SBATCH --partition=brc
#SBATCH --tasks=4
#SBATCH --mem=4000
#SBATCH --job-name=array
#SBATCH --array=0-243
#SBATCH --output=/scratch/users/%u/logs/%a.out
#SBATCH --time=0-72:00
module load apps/singularity
cd /scratch/users/k1644956/tda/prediction/v2/
n_rep=100
singularity exec containers/python.simg \
	python prediction/test_single_model.py $SLURM_ARRAY_TASK_ID $n_rep
