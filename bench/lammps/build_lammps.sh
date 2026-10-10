#!/usr/bin/env bash
# Build one LAMMPS variant for bench/counterparts.py (serial, MPI STUBS, no OpenMP).
#   usage: bench/lammps/build_lammps.sh {gpu-mixed|gpu-single|kokkos-double|kokkos-single}
# Sources: git clone --depth 1 --branch stable_30Sep2026 https://github.com/lammps/lammps
#          into $LAMMPS_SRC (default ~/python-hpc/_ref/lammps); builds go to $LAMMPS_BUILDS.
set -euo pipefail
SRC=${LAMMPS_SRC:-$HOME/python-hpc/_ref/lammps}
OUT=${LAMMPS_BUILDS:-$HOME/python-hpc/_ref/lammps-builds}
JOBS=${JOBS:-4}
name=$1
export PATH=/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda
common=(-D CMAKE_BUILD_TYPE=Release -D BUILD_MPI=no -D BUILD_OMP=no -D LAMMPS_MACHINE="$name")
case $name in
  gpu-mixed|gpu-single)
    flags=("${common[@]}" -C "$SRC/cmake/presets/gpu-cuda.cmake" -D GPU_PREC="${name#gpu-}"
           -D GPU_ARCH=sm_86 -D CUDA_BUILD_MULTIARCH=off) ;;
  kokkos-double|kokkos-single|kokkos-mixed)
    flags=("${common[@]}" -C "$SRC/cmake/presets/kokkos-cuda.cmake" -D Kokkos_ARCH_AMPERE86=yes
           -D KOKKOS_PREC="${name#kokkos-}") ;;
  *) echo "unknown variant $name" >&2; exit 1 ;;
esac
B=$OUT/$name
mkdir -p "$B"
cd "$B"
echo "cmake ${flags[*]} $SRC/cmake" > cmake_command.txt
cmake "${flags[@]}" "$SRC/cmake" > cmake.log 2>&1
make -j"$JOBS" > make.log 2>&1
ls -la "$B/lmp_$name"
