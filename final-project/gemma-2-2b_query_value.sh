#! /usr/bin/env zsh
#SBATCH --partition=instruction
#SBATCH --job-name=hadamard_cuda
#SBATCH --output="hadamard-gemma-2-2b.out"
#SBATCH --error="hadamard-gemma-2-2b.err"
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

# Common parameters
threads=(32 128 512 1024)
typeset -A inputs
inputs=(
  1     ./gemma-2-2b/inputs/x_1.npy
  16    ./gemma-2-2b/inputs/x_16.npy
  128   ./gemma-2-2b/inputs/x_128.npy
  512   ./gemma-2-2b/inputs/x_512.npy
  1024  ./gemma-2-2b/inputs/x_1024.npy
)

# Modes to run (query + value)
typeset -A modes
modes=(
  query "./gemma-2-2b/query/Gemma-2-2b-hadamard-query-CT.npy ./gemma-2-2b/query/Gemma-2-2b-hadamard-query-locs.npy 2048"
  value "./gemma-2-2b/value/Gemma-2-2b-hadamard-value-CT.npy ./gemma-2-2b/value/Gemma-2-2b-hadamard-value-locs.npy 1024"
)

CT_ROWS=2304

# Loop through all combinations
for mode in "${(k)modes}"; do
  IFS=" " read -r CT LOCS CT_COLS <<< "${modes[$mode]}"
  for tokens in "${(k)inputs}"; do
    for tpb in $threads; do
      echo "Running mode=$mode tokens=$tokens threads_per_block=$tpb"
      ./hadamard_exec $CT_ROWS $CT_COLS $tokens $tpb $CT $LOCS ${inputs[$tokens]}
    done
  done
done
