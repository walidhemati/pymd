Source: results/quick.json (provisional: True), N = 32,000

| backend | variant | Matom-steps/s | % copy BW | ns/day (Ar) |
|---|---|---:|---:|---:|
| cupy-cuda-c | full | 308.5 | 30 | 8980 |
| cupy-cuda-c | full-fused | 304.1 | 29 | 8851 |
| warp | full | 266.3 | 26 | 7753 |
| numba | full-fused | 255.8 | 25 | 7446 |
| triton | full-fused | 218.6 | 21 | 6364 |
| triton | full | 207.6 | 20 | 6045 |
| numba | full | 181.9 | 18 | 5296 |
| cupy-cuda-c | half-atomic | 156.8 | 9 | 4563 |
| warp | hashgrid | 61.8 | - | 1800 |
| torch-compile | gather | 26.8 | 3 | 780 |
| torch-eager | gather | 3.7 | 0 | 109 |
