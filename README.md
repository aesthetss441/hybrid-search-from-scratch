# hybrid-search-from-scratch

[![CI](https://github.com/aesthetss441/hybrid-search-from-scratch/actions/workflows/ci.yml/badge.svg)](https://github.com/aesthetss441/hybrid-search-from-scratch/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

The three pieces every RAG retrieval stack is built on, each written from scratch in about 100 lines of NumPy and checked against the production libraries:

- **[`hnsw.py`](src/hsearch/hnsw.py)** — an HNSW approximate nearest-neighbour index (the algorithm inside FAISS, hnswlib, pgvector, Qdrant and Weaviate)
- **[`bm25.py`](src/hsearch/bm25.py)** — Okapi BM25 over an inverted index
- **[`hybrid.py`](src/hsearch/hybrid.py)** — reciprocal rank fusion to combine the two

It exists because "tune `ef` and `M`" and "why did hybrid search rank this first" are hard to reason about when the index is a black box. This one is short enough to read in one sitting, and the benchmark shows it behaves like the real thing.

## Does it actually work?

Same data, same parameters (M=16, ef_construction=100), single thread, against hnswlib and FAISS. Fashion-MNIST, 20,000 vectors of 784 dimensions, 500 queries, recall@10. Produced by [`bench/ann.py`](bench/ann.py) in GitHub Actions and committed by the workflow ([RESULTS.md](RESULTS.md)).

| ef | hsearch recall | hsearch QPS | hnswlib recall | hnswlib QPS | FAISS recall | FAISS QPS |
|---|---|---|---|---|---|---|
| 10 | 0.977 | 2,310 | 0.948 | 19,579 | 0.941 | 19,569 |
| 20 | 0.995 | 1,617 | 0.987 | 13,788 | 0.982 | 13,784 |
| 40 | 0.998 | 1,030 | 0.996 | 8,395 | 0.996 | 8,586 |
| 80 | 0.999 | 632 | 0.998 | 5,583 | 0.998 | 5,578 |
| 160 | 1.000 | 380 | 0.999 | 3,704 | 1.000 | 3,607 |

What this shows:

- **Correctness.** Recall at every `ef` is at or slightly above both C++ libraries, so the graph is being built and searched properly.
- **The cost of Python.** Queries are 8–10× slower and the build is about 38× slower (132 s against 3.5 s). The graph walk is interpreted; only distance computations are vectorised.
- **Why anyone uses an index.** An exact NumPy scan of the same data manages 34 queries per second. Even this pure-Python index is 11–68× faster at 97.7–100% recall.

Timings come from a shared CI runner and will move between runs; recall is deterministic for a fixed seed. One dataset at one size is evidence of correctness, not a general performance claim.

## Use

```bash
pip install git+https://github.com/aesthetss441/hybrid-search-from-scratch
```

```python
import numpy as np
from hsearch import HNSW, BM25, HybridIndex, tokenize

index = HNSW(dim=384, M=16, ef_construction=200, metric="cosine")
index.add_batch(embeddings)                 # (n, 384) array
ids, distances = index.search(query_vector, k=10, ef=50)

bm25 = BM25([tokenize(t) for t in texts])
ids, scores = bm25.search(tokenize("refund policy"), k=10)

hybrid = HybridIndex(texts, embeddings)     # both, fused with RRF
ids = hybrid.search(text="refund policy", vector=query_vector, k=10)
```

## How HNSW works, in the order the code does it

1. **Layers.** Each new vector gets a random top layer drawn from an exponential distribution, so layer 0 holds everything and each layer above holds roughly 1/M of the one below. Upper layers are express lanes.
2. **Search one layer** (`_search_layer`, Algorithm 2). A best-first walk with two heaps: candidates to expand (closest first) and the best `ef` results so far (worst on top). It stops when the closest unexpanded candidate is farther than the worst kept result. `ef` is the recall/speed dial.
3. **Insert** (`add`, Algorithm 1). Descend greedily from the top to the new vector's layer, then on each layer below, find `ef_construction` candidates, link to up to `M` of them, and trim any neighbour that is now over capacity.
4. **Choose neighbours** (`_select_neighbors`, Algorithm 4). Not simply the `M` closest: a candidate is kept only if it is closer to the new point than to every neighbour already kept. That preserves edges between clusters, which is what keeps the graph navigable on clustered data.
5. **Query** (`search`, Algorithm 5). Greedy descent to layer 0, then one `ef`-wide search there.

Reference: Malkov & Yashunin, *Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs*, 2018.

## What this is not

A production index. There is no deletion, persistence, filtering, quantisation or concurrency, and the graph is stored in Python lists. Use hnswlib, FAISS or a vector database for real workloads; use this to understand them.

## Development

```bash
pip install -e ".[dev]" && pytest -q        # recall vs exact search, graph invariants, BM25, fusion
pip install -e ".[bench]" && python bench/ann.py
```

MIT licensed.
