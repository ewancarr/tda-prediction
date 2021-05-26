#!/bin/bash
#SBATCH --partition=brc
#SBATCH --tasks=8
#SBATCH --mem=16000
#SBATCH --job-name=array
#SBATCH --array=5,16
#SBATCH --output=/scratch/users/%u/tda/prediction/v2/prediction/logs/%a.out
#SBATCH --time=0-72:00
echo "Starting model: $SLURM_ARRAY_TASK_ID"
module load apps/singularity
cd /scratch/users/k1644956/tda/prediction/v2/
n_rep=100
singularity exec containers/python.simg \
	python prediction/test_single_model.py $SLURM_ARRAY_TASK_ID $n_rep
echo "Finished model: $SLURM_ARRAY_TASK_ID"
