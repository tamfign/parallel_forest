#!/bin/csh
#SBATCH --time=1:00:00
#SBATCH --nodes=10
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=8
#SBATCH --output="result"

# Run info and srun job launch
srun ./exec Dataset/train/train-first1000.arff Dataset/test/dev-first1000.arff
