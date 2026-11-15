"""Import faiss only through `load_faiss()`; one OpenMP thread on macOS.

The macOS wheels of faiss-cpu (1.16) and torch (2.15) each bundle their own `libomp.dylib`.
With two OpenMP runtimes in one process, multi-threaded OpenMP use crashes it: `OMP: Error #15`
(abort) or a segmentation fault inside torch, whichever library is imported first, and also
with `KMP_DUPLICATE_LIB_OK=TRUE`. With `OMP_NUM_THREADS=1` set before either
library loads, embedding and search are stable in both import orders and give the same
results. `claimcheck.index` sets that default on import (`omp_guard`); an explicit
`OMP_NUM_THREADS` in the environment wins. Linux wheels use GNU OpenMP (not measured here).
"""

from __future__ import annotations

import importlib
import os
import sys
import warnings
from types import ModuleType


def omp_guard() -> None:
    """On macOS, default to one OpenMP thread (must run before torch or faiss is imported)."""
    if sys.platform == "darwin":
        os.environ.setdefault("OMP_NUM_THREADS", "1")


def load_faiss() -> ModuleType:
    omp_guard()
    torch = sys.modules.get("torch")
    if sys.platform == "darwin" and torch is not None and torch.get_num_threads() > 1:
        warnings.warn(
            "torch was imported with several OpenMP threads before faiss; on macOS this can "
            "crash the process. Import claimcheck.index first or set OMP_NUM_THREADS=1.",
            RuntimeWarning,
            stacklevel=2,
        )
    return importlib.import_module("faiss")
