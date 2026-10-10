Source: results/rtx3070-laptop.json (provisional: False), N = 2,048,000

| backend | variant | Matom-steps/s | % copy BW | ns/day (Ar) |
|---|---|---:|---:|---:|
| cupy-cuda-c | full-fused | 488.0 | 47 | 222 |
| cupy-cuda-c | full | 456.6 | 44 | 208 |
| cupy-cuda-c | full-morton | 453.5 | 44 | 206 |
| triton | full-fused | 437.0 | 42 | 199 |
| cupy-cuda-c | full-tpa2 | 429.5 | 41 | 195 |
| triton | full | 420.5 | 41 | 191 |
| cupy-cuda-c | full-tpa4 | 409.6 | 39 | 186 |
| numba | full-fused | 403.8 | 39 | 184 |
| numba | full | 381.9 | 37 | 174 |
| warp | full | 376.6 | 36 | 171 |
| warp | hashgrid-verlet | 328.9 | 32 | 150 |
| cupy-cuda-c | half-atomic | 212.9 | 12 | 97 |
| triton | half-atomic | 156.1 | 9 | 71 |
| warp | hashgrid | 88.1 | - | 40 |
| cupy-cuda-c | full-mixed | 63.7 | 7 | 29 |
| cupy-cuda-c | full-nosort | 39.0 | 4 | 18 |
| cupy-cuda-c | full-fp64 | 33.4 | 4 | 15 |
