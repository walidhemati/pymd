#!/usr/bin/env bash
# Build HOOMD-blue in single precision (LongReal = ShortReal = 32 bit) for bench/counterparts.py.
# conda-forge ships only HOOMD_LONGREAL_SIZE=64 builds, whose pair kernels run in float64 --
# 1/64 of the float32 rate on this consumer GPU. Prerequisites (separate env, never base conda):
#   git clone --recursive --depth 1 --branch v7.2.0 https://github.com/glotzerlab/hoomd-blue
#   conda create -p $REF/hoomdbuild -c conda-forge --override-channels python=3.13 numpy \
#     pybind11 eigen cmake ninja gxx_linux-64=14 cuda-nvcc=12.9 cuda-version=12.9 \
#     cuda-cudart-dev libcublas-dev libcufft-dev libcusolver-dev libcusparse-dev cuda-profiler-api
# The module is used from the build tree (sys.path), it is not installed.
set -euo pipefail
REF=${PYHPC_REF:-$HOME/python-hpc/_ref}
SRC=${HOOMD_SRC:-$REF/hoomd-blue}
ENV=${HOOMD_BUILD_ENV:-$REF/hoomdbuild}
JOBS=${JOBS:-4}
export PATH=$ENV/bin:$PATH
B=$SRC/build-single
flags=(-GNinja -DCMAKE_BUILD_TYPE=Release -DENABLE_GPU=on -DCUDA_ARCH_LIST=86
       -DHOOMD_LONGREAL_SIZE=32 -DHOOMD_SHORTREAL_SIZE=32 -DENABLE_MPI=off
       -DBUILD_HPMC=off -DBUILD_METAL=off -DBUILD_MPCD=off -DBUILD_TESTING=off
       -DPython_EXECUTABLE=$ENV/bin/python -DCMAKE_PREFIX_PATH=$ENV
       -DCMAKE_C_COMPILER=$ENV/bin/x86_64-conda-linux-gnu-gcc
       -DCMAKE_CXX_COMPILER=$ENV/bin/x86_64-conda-linux-gnu-g++
       -DCMAKE_CUDA_COMPILER=$ENV/bin/nvcc -DCMAKE_CUDA_HOST_COMPILER=$ENV/bin/x86_64-conda-linux-gnu-g++)
# v7.2.0 does not compile with LongReal = 32: GSD constraint values are std::vector<float>
# but writeFloatDoubleChunk takes std::vector<double>. Convert explicitly (GSD output only).
sed -i 's|writeFloatDoubleChunk("constraints/value", constraint.val, N, 1);|writeFloatDoubleChunk("constraints/value", std::vector<double>(constraint.val.begin(), constraint.val.end()), N, 1);|' "$SRC/hoomd/GSDDumpWriter.cc"
mkdir -p "$B"
echo "cmake -B $B -S $SRC ${flags[*]}" > "$B/cmake_command.txt"
cmake -B "$B" -S "$SRC" "${flags[@]}" > "$B/cmake.log" 2>&1
ninja -C "$B" -j"$JOBS" > "$B/ninja.log" 2>&1
ls -d "$B/hoomd"
