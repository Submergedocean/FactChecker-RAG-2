"""claimcheck.index package: embeddings, BM25 and the FAISS index.

Importing it sets `OMP_NUM_THREADS=1` on macOS unless already set: faiss and torch bundle two
OpenMP runtimes there, which crash together when multi-threaded (see `index/_faiss.py`).
"""

from claimcheck.index._faiss import omp_guard

omp_guard()
