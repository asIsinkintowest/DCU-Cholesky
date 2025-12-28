#!/bin/bash
#SBATCH -J chol_bench
#SBATCH -o output/slurm_%j.out
#SBATCH -e output/slurm_%j.err
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --gres=dcu:1
#SBATCH -t 00:30:00

set -euo pipefail

module purge
module load rocm
module load scalapack

export OMP_NUM_THREADS=1

make all

./build/run_bench \
  --n 8192 \
  --block 256 \
  --p 2 \
  --q 2 \
  --iters 3 \
  --runs 1 \
  --peak-tflops 0.0
