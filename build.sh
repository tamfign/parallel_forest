#!/bin/csh
#SBATCH --time=00:10:00
#SBATCH --nodes=10
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=8
#SBATCH --output="result"

# Run info and srun job launch
srun ./exec Dataset/train/train_1000.arff Dataset/test/dev_1000.arff
