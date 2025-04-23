#! /usr/bin/env zsh
#SBATCH --partition=instruction
#SBATCH --job-name=hadamard_cuda
#SBATCH --output="hadamard_%j.out"
#SBATCH --error="hadamard_%j.err"
#SBATCH --gres=gpu:1
#SBATCH --time=0-00:30:00
#SBATCH --mem=32GB

# Load CUDA
module load nvidia/cuda/11.8.0

# Compile
nvcc hadamard.cu ./cnpy/cnpy.cpp ./utils/load_ckpt.cpp \
  -o hadamard_exec \
  -std=c++17 -O3 -Xcompiler "-Wall -Wno-unused-variable -Wno-unused-but-set-variable" \
  -I./utils -I./cnpy \
  -lz


# Execute with correct argument order
# Format: <ct_rows> <ct_cols> <num_tokens> <threads_per_block> <CT> <LOCS> <X>
./hadamard_exec 2304 2048 1 16 \
  ./gemma-2-2b/query/Gemma-2-2b-hadamard-query-CT.npy \
  ./gemma-2-2b/query/Gemma-2-2b-hadamard-query-locs.npy \
  ./gemma-2-2b/inputs/x_1.npy

  ./hadamard_exec 2304 2048 16 16 \
  ./gemma-2-2b/query/Gemma-2-2b-hadamard-query-CT.npy \
  ./gemma-2-2b/query/Gemma-2-2b-hadamard-query-locs.npy \
  ./gemma-2-2b/inputs/x_16.npy

    ./hadamard_exec 2304 2048 128 16 \
  ./gemma-2-2b/query/Gemma-2-2b-hadamard-query-CT.npy \
  ./gemma-2-2b/query/Gemma-2-2b-hadamard-query-locs.npy \
  ./gemma-2-2b/inputs/x_128.npy

   ./hadamard_exec 2304 2048 512 16 \
  ./gemma-2-2b/query/Gemma-2-2b-hadamard-query-CT.npy \
  ./gemma-2-2b/query/Gemma-2-2b-hadamard-query-locs.npy \
  ./gemma-2-2b/inputs/x_512.npy

   ./hadamard_exec 2304 2048 1024 16 \
  ./gemma-2-2b/query/Gemma-2-2b-hadamard-query-CT.npy \
  ./gemma-2-2b/query/Gemma-2-2b-hadamard-query-locs.npy \
  ./gemma-2-2b/inputs/x_1024.npy

  
    ./hadamard_exec 2304 2048 1 256 \
  ./gemma-2-2b/query/Gemma-2-2b-hadamard-query-CT.npy \
  ./gemma-2-2b/query/Gemma-2-2b-hadamard-query-locs.npy \
  ./gemma-2-2b/inputs/x_1.npy

   ./hadamard_exec 2304 2048 1024 256 \
  ./gemma-2-2b/query/Gemma-2-2b-hadamard-query-CT.npy \
  ./gemma-2-2b/query/Gemma-2-2b-hadamard-query-locs.npy \
  ./gemma-2-2b/inputs/x_1024.npy