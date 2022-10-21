#!/bin/bash
#
#SBATCH --job-name=run-1
#
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --mail-type=END,FAIL # notifications for job done & fail
#SBATCH --mail-user=zhaonanq@stanford.edu
#SBATCH --output=main.out

module load matlab

matlab -nodisplay -r "test"
