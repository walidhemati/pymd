Source: results/dev.json (provisional: True), N = 500,000

| backend | variant | Matom-steps/s | % copy BW | ns/day (Ar) |
|---|---|---:|---:|---:|
| cupy-cuda-c | full-fused | 212.4 | 20 | 396 |
| triton | full-fused | 206.8 | 20 | 385 |
| numba | full | 168.2 | 16 | 313 |
| triton | full | 164.5 | 16 | 306 |
| warp | full | 161.8 | 16 | 301 |
| numba | full-fused | 154.2 | 15 | 287 |
| cupy-cuda-c | full | 144.4 | 14 | 269 |
| cupy-cuda-c | half-atomic | 80.4 | 4 | 150 |
| warp | hashgrid | 30.2 | - | 56 |
| torch-compile | gather | 4.0 | 0 | 7 |
| torch-eager | gather | 3.4 | 0 | 6 |
