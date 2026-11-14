"""Recall/throughput of hsearch.HNSW against hnswlib and FAISS on Fashion-MNIST (784-d, L2).

Run: python bench/ann.py [--base 20000] [--queries 500] [--out RESULTS.md]
All three use M=16, ef_construction=100 and a single thread.
"""
from __future__ import annotations

import argparse
import platform
import time

import numpy as np

from hsearch import HNSW

M, EFC, K = 16, 100, 10
EFS = [10, 20, 40, 80, 160]


def load(n_base: int, n_query: int) -> tuple[np.ndarray, np.ndarray]:
    from datasets import load_dataset
    ds = load_dataset("zalando-datasets/fashion_mnist", split="train")
    images = ds[: n_base + n_query]["image"]
    data = np.stack([np.asarray(im, dtype=np.float32).reshape(-1) for im in images])
    return data[:n_base], data[n_base:]


def exact(base: np.ndarray, queries: np.ndarray) -> np.ndarray:
    d = (queries ** 2).sum(1)[:, None] - 2 * queries @ base.T + (base ** 2).sum(1)[None, :]
    return np.argsort(d, axis=1)[:, :K]


def measure(search, queries: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    start = time.perf_counter()
    found = [search(q) for q in queries]
    seconds = time.perf_counter() - start
    recall = np.mean([len(set(f) & set(t)) / K for f, t in zip(found, truth.tolist())])
    return float(recall), len(queries) / seconds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=int, default=20000)
    ap.add_argument("--queries", type=int, default=500)
    ap.add_argument("--out", default="RESULTS.md")
    args = ap.parse_args()

    import faiss
    import hnswlib
    faiss.omp_set_num_threads(1)

    base, queries = load(args.base, args.queries)
    truth = exact(base, queries)
    rows, builds = [], {}

    t = time.perf_counter()
    ours = HNSW(base.shape[1], M=M, ef_construction=EFC)
    ours.add_batch(base)
    builds["hsearch (this repo, pure Python + NumPy)"] = time.perf_counter() - t

    t = time.perf_counter()
    lib = hnswlib.Index(space="l2", dim=base.shape[1])
    lib.init_index(max_elements=len(base), M=M, ef_construction=EFC, random_seed=0)
    lib.set_num_threads(1)
    lib.add_items(base)
    builds["hnswlib (C++)"] = time.perf_counter() - t

    t = time.perf_counter()
    fa = faiss.IndexHNSWFlat(base.shape[1], M)
    fa.hnsw.efConstruction = EFC
    fa.add(base)
    builds["FAISS IndexHNSWFlat (C++)"] = time.perf_counter() - t

    for ef in EFS:
        lib.set_ef(ef)
        fa.hnsw.efSearch = ef
        rows.append((ef,
                     measure(lambda q: ours.search(q, K, ef)[0], queries, truth),
                     measure(lambda q: lib.knn_query(q, K)[0][0].tolist(), queries, truth),
                     measure(lambda q: fa.search(q[None, :], K)[1][0].tolist(), queries, truth)))

    brute_start = time.perf_counter()
    for q in queries:
        np.argpartition(((base - q) ** 2).sum(1), K)[:K]
    brute_qps = len(queries) / (time.perf_counter() - brute_start)

    lines = [
        "# Benchmark results",
        "",
        f"Fashion-MNIST, {len(base):,} base vectors, {len(queries)} queries, 784 dimensions, squared L2, recall@{K}.",
        f"M={M}, ef_construction={EFC}, single thread. {platform.processor() or platform.machine()}, Python {platform.python_version()}.",
        "Produced by `bench/ann.py` in GitHub Actions; timings vary between runs, recall does not.",
        "",
        "| ef | hsearch recall | hsearch QPS | hnswlib recall | hnswlib QPS | FAISS recall | FAISS QPS |",
        "|---|---|---|---|---|---|---|",
    ]
    for ef, o, h, f in rows:
        lines.append(f"| {ef} | {o[0]:.3f} | {o[1]:,.0f} | {h[0]:.3f} | {h[1]:,.0f} | {f[0]:.3f} | {f[1]:,.0f} |")
    lines += ["", f"Exact NumPy scan: recall 1.000 at {brute_qps:,.0f} QPS.", "", "| Index | Build seconds |", "|---|---|"]
    lines += [f"| {name} | {seconds:.1f} |" for name, seconds in builds.items()]
    report = "\n".join(lines) + "\n"
    print(report)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(report)


if __name__ == "__main__":
    main()
