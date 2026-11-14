# Benchmark results

Fashion-MNIST, 20,000 base vectors, 500 queries, 784 dimensions, squared L2, recall@10.
M=16, ef_construction=100, single thread. x86_64, Python 3.12.16.
Produced by `bench/ann.py` in GitHub Actions; timings vary between runs, recall does not.

| ef | hsearch recall | hsearch QPS | hnswlib recall | hnswlib QPS | FAISS recall | FAISS QPS |
|---|---|---|---|---|---|---|
| 10 | 0.977 | 2,310 | 0.948 | 19,579 | 0.941 | 19,569 |
| 20 | 0.995 | 1,617 | 0.987 | 13,788 | 0.982 | 13,784 |
| 40 | 0.998 | 1,030 | 0.996 | 8,395 | 0.996 | 8,586 |
| 80 | 0.999 | 632 | 0.998 | 5,583 | 0.998 | 5,578 |
| 160 | 1.000 | 380 | 0.999 | 3,704 | 1.000 | 3,607 |

Exact NumPy scan: recall 1.000 at 34 QPS.

| Index | Build seconds |
|---|---|
| hsearch (this repo, pure Python + NumPy) | 132.2 |
| hnswlib (C++) | 3.5 |
| FAISS IndexHNSWFlat (C++) | 3.4 |
