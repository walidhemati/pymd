"""Make NVRTC loadable in a pip-only CUDA environment (no system CUDA toolkit).

CuPy ``dlopen``s NVRTC by soname (``libnvrtc.so.12``). When NVRTC comes from the
``nvidia-cuda-nvrtc-cu12`` wheel (pulled in by PyTorch) it is not on the loader path;
preloading it with ``RTLD_GLOBAL`` lets CuPy find it.
"""

from __future__ import annotations

import ctypes
import functools
import os


@functools.cache
def preload_nvrtc() -> bool:
    """Preload the pip-installed NVRTC; returns True if a library was loaded."""
    try:
        import nvidia.cuda_nvrtc as pkg  # type: ignore[import-not-found]
    except Exception:
        return False
    for base in list(getattr(pkg, "__path__", [])):
        lib = os.path.join(base, "lib", "libnvrtc.so.12")
        if os.path.exists(lib):
            try:
                ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
                return True
            except OSError:
                return False
    return False
