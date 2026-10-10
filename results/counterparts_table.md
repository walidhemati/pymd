Source: results/counterparts.json (provisional: False), N = 4,000,000

| code | variant | precision | Matom-steps/s |
|---|---|---|---:|
| pymd | cupy-cuda-c/full-fused | single | 470.5 |
| pymd | cupy-cuda-c/full | single | 449.4 |
| pymd | triton/full-fused | single | 426.2 |
| pymd | numba/full-fused | single | 394.2 |
| pymd | warp/full | single | 370.0 |
| hoomd | gpu-single | single | 364.3 |
| lammps | kokkos-single-half | single | 103.2 |
| lammps | kokkos-single | single | 95.5 |
| lammps | kokkos-double-half | double | 58.1 |
| lammps | kokkos-double | double | 50.5 |
| lammps | gpu-single | single | 50.4 |
| lammps | gpu-mixed | mixed | 38.6 |
| hoomd | gpu-double | double | 30.4 |
