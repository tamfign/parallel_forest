#!/bin/csh
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=8
#SBATCH --output="result"

# Run info and srun job launch
srun -n 2 ./exec Dataset/train/train-first10.arff Dataset/test/dev-first10.arff
