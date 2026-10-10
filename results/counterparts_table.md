Source: results/counterparts.json (provisional: True), N = 32,000

| code | variant | precision | Matom-steps/s |
|---|---|---|---:|
| pymd | cupy-cuda-c/full-fused | single | 362.7 |
| pymd | cupy-cuda-c/full | single | 350.0 |
| pymd | triton/full-fused | single | 342.6 |
| hoomd | gpu-single | single | 296.5 |
| lammps | kokkos-single | single | 103.6 |
| lammps | kokkos-single-half | single | 85.3 |
| lammps | gpu-single | single | 67.6 |
| lammps | gpu-mixed | mixed | 49.8 |
| lammps | kokkos-double-half | double | 45.8 |
| lammps | kokkos-double | double | 42.9 |
| hoomd | gpu-double | double | 27.5 |
| lammps | cpu-1core | double | 2.6 |
