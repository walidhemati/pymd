"""Backend registry.

Each backend module defines one :class:`pymd.engine.Engine` subclass. Modules whose
dependencies (or a GPU) are missing are skipped cleanly; :func:`available` reports why.
"""

from __future__ import annotations

import importlib

from ..engine import Engine

#: backend name -> (module, class name). Order = the optimization ladder.
_SPECS: dict[str, tuple[str, str]] = {
    "numpy": ("numpy_backend", "NumpyEngine"),
    "torch-eager": ("torch_backend", "TorchEagerEngine"),
    "torch-compile": ("torch_backend", "TorchCompileEngine"),
    "triton": ("triton_backend", "TritonEngine"),
    "warp": ("warp_backend", "WarpEngine"),
    "numba": ("numba_backend", "NumbaEngine"),
    "cupy-cuda-c": ("cupy_backend", "CupyEngine"),
}

_cache: dict[str, type[Engine] | Exception] = {}


def _load(name: str) -> type[Engine] | Exception:
    if name not in _cache:
        mod_name, cls_name = _SPECS[name]
        try:
            mod = importlib.import_module(f"{__name__}.{mod_name}")
            cls = getattr(mod, cls_name)
            ok = getattr(mod, "is_available", lambda: True)()
            _cache[name] = cls if ok else RuntimeError("no CUDA device")
        except Exception as exc:  # missing package, no driver, ...
            _cache[name] = exc
    return _cache[name]


def names() -> list[str]:
    """All registered backend names (available or not)."""
    return list(_SPECS)


def available() -> dict[str, type[Engine]]:
    """Backends that import and find their device."""
    out = {}
    for n in _SPECS:
        cls = _load(n)
        if isinstance(cls, type):
            out[n] = cls
    return out


def unavailable() -> dict[str, str]:
    """Backends that could not be loaded, with the reason."""
    return {n: repr(c) for n in _SPECS if not isinstance(c := _load(n), type)}


def get(name: str) -> type[Engine]:
    """Engine class for ``name``; raises the import error if unavailable."""
    cls = _load(name)
    if isinstance(cls, Exception):
        raise RuntimeError(f"backend {name!r} unavailable: {cls!r}")
    return cls


def all_variants() -> list[tuple[str, str]]:
    """(backend, variant) pairs for every available backend."""
    return [(b, v) for b, cls in available().items() for v in cls.variants]
