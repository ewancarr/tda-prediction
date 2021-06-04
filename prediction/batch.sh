#!/bin/bash
#SBATCH --partition=brc
#SBATCH --tasks=4
#SBATCH --mem=16000
#SBATCH --job-name=array
#SBATCH --array=0-542
#SBATCH --output=/scratch/users/%u/tda/prediction/v2/prediction/logs/%a.out
#SBATCH --time=0-72:00
module load apps/singularity
cd /scratch/users/k1644956/tda/prediction/v2/
REP=100
ID=$(printf "%04d" $SLURM_ARRAY_TASK_ID)
OUTPUT=prediction/fits/$ID.joblib
while [ ! -f $OUTPUT ]
do
    singularity exec containers/python.simg \
        python prediction/test_single_model.py $ID $REP
done
