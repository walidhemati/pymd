"""Measure the hardware ceilings used by the roofline on the current machine.

* device-to-device copy bandwidth (``torch`` ``copy_``, read + write counted) -- the
  memory ceiling ("% of STREAM copy");
* fp32 FMA throughput of a CUDA C microbenchmark (8 independent FMA chains per thread,
  1 FMA = 2 flop) -- the compute ceiling.

Run directly: ``python bench/roofline.py [--quick]``.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

SPEC_BANDWIDTH_GBS = 448.0  # RTX 3070 Laptop: 14 Gbps GDDR6 x 256-bit

FMA_SRC = r"""
extern "C" __global__ void fma_bench(float* out, int iters, float a, float b) {
    float x0 = threadIdx.x, x1 = x0 + 1, x2 = x0 + 2, x3 = x0 + 3;
    float x4 = x0 + 4, x5 = x0 + 5, x6 = x0 + 6, x7 = x0 + 7;
    for (int i = 0; i < iters; ++i) {
        #pragma unroll
        for (int k = 0; k < 16; ++k) {
            x0 = fmaf(x0, a, b); x1 = fmaf(x1, a, b); x2 = fmaf(x2, a, b); x3 = fmaf(x3, a, b);
            x4 = fmaf(x4, a, b); x5 = fmaf(x5, a, b); x6 = fmaf(x6, a, b); x7 = fmaf(x7, a, b);
        }
    }
    out[blockIdx.x * blockDim.x + threadIdx.x] = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7;
}
"""


def copy_bandwidth(nbytes: int = 1 << 29, repeats: int = 20) -> float:
    """Median device-to-device copy bandwidth in GB/s (bytes read + bytes written)."""
    import torch

    a = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
    b = torch.empty_like(a)
    for _ in range(3):
        b.copy_(a)
    torch.cuda.synchronize()
    rates = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        b.copy_(a)
        torch.cuda.synchronize()
        rates.append(2 * nbytes / (time.perf_counter() - t0) / 1e9)
    del a, b
    torch.cuda.empty_cache()
    return statistics.median(rates)


def fma_tflops(iters: int = 4096, repeats: int = 7) -> float:
    """Median fp32 FMA throughput in TFLOP/s (1 FMA counted as 2 flop)."""
    from pymd.backends.cupy_backend import cp

    fn = cp.RawModule(code=FMA_SRC).get_function("fma_bench")
    sms = cp.cuda.Device().attributes["MultiProcessorCount"]
    threads, blocks = 256, sms * 16
    out = cp.empty(threads * blocks, dtype=cp.float32)
    args = (out, cp.int32(iters), cp.float32(0.999), cp.float32(0.001))
    fn((blocks,), (threads,), args)
    cp.cuda.Device().synchronize()
    rates = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn((blocks,), (threads,), args)
        cp.cuda.Device().synchronize()
        flops = 2.0 * 8 * 16 * iters * threads * blocks
        rates.append(flops / (time.perf_counter() - t0) / 1e12)
    return statistics.median(rates)


def measure(quick: bool = False) -> dict:
    """All ceilings as a dict (the ``ceiling`` block of the results JSON)."""
    out = {"copy_bandwidth_GBs": None, "spec_bandwidth_GBs": SPEC_BANDWIDTH_GBS,
           "fp32_tflops": None}
    try:
        out["copy_bandwidth_GBs"] = round(copy_bandwidth(1 << (27 if quick else 29)), 1)
    except Exception as exc:  # pragma: no cover
        out["copy_error"] = repr(exc)
    try:
        out["fp32_tflops"] = round(fma_tflops(1024 if quick else 4096), 2)
    except Exception as exc:  # pragma: no cover
        out["fma_error"] = repr(exc)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    print(json.dumps(measure(ap.parse_args().quick), indent=1))
