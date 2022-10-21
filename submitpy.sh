#!/bin/bash
#
#SBATCH --job-name=rate_100
#SBATCH --qos=long
#SBATCH --time=23:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --mail-type=END,FAIL # notifications for job done & fail
#SBATCH --mail-user=zhaonanq@stanford.edu
#SBATCH --output=main3_3.out

module load python
python counterfactual_estimation.py
