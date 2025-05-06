#!/bin/bash 

#SBATCH --account=mcmorran_lab
#SBATCH --job-name=test
#SBATCH --output=test.output
#SBATCH --error=test.err
#SBATCH --partition=gpu
#SBATCH --time=0-01:00:00
#SBATCH --nodes=1             ### Node count required for the job
#SBATCH --ntasks-per-node=1   ### Nuber of tasks to be launched per Node
#SBATCH --gres=gpu:1          ### General REServation of gpu:number of gpus
#SBATCH --mem=256G

date

module load tensorflow
python keplerianTelescope.py

date

