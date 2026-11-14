import numpy as np
import pytest

from hsearch import BM25, HNSW, HybridIndex, rrf, tokenize


def brute(base, q, k, metric):
    if metric == "cosine":
        b = base / np.linalg.norm(base, axis=1, keepdims=True)
        d = 1 - b @ (q / np.linalg.norm(q))
    else:
        d = ((base - q) ** 2).sum(1)
    return set(np.argsort(d)[:k].tolist())


@pytest.mark.parametrize("metric", ["l2", "cosine"])
def test_hnsw_recall_against_exact_search(metric):
    rng = np.random.default_rng(0)
    centers = rng.normal(size=(20, 32)) * 4  # clustered data is the hard case for graph indexes
    base = (centers[rng.integers(0, 20, 3000)] + rng.normal(size=(3000, 32))).astype(np.float32)
    queries = (centers[rng.integers(0, 20, 100)] + rng.normal(size=(100, 32))).astype(np.float32)
    index = HNSW(32, M=12, ef_construction=80, metric=metric)
    index.add_batch(base)
    assert len(index) == 3000

    def recall(ef):
        hits = sum(len(set(index.search(q, 10, ef)[0]) & brute(base, q, 10, metric)) for q in queries)
        return hits / 1000

    low, high = recall(10), recall(100)
    assert high >= 0.95, high
    assert high >= low
    ids, dists = index.search(queries[0], 10, 100)
    assert dists == sorted(dists) and len(set(ids)) == 10


def test_hnsw_graph_invariants():
    rng = np.random.default_rng(1)
    index = HNSW(8, M=6, ef_construction=40)
    index.add_batch(rng.normal(size=(500, 8)))
    for node, layers in enumerate(index.links):
        for layer, adjacency in enumerate(layers):
            assert len(adjacency) <= (index.M0 if layer == 0 else index.M)
            assert node not in adjacency and len(set(adjacency)) == len(adjacency)
            assert all(len(index.links[n]) > layer for n in adjacency)
    assert HNSW(4).search(np.zeros(4)) == ([], [])


def test_bm25_prefers_rare_terms_and_shorter_documents():
    docs = ["the cat sat on the mat", "the dog chased the cat around the large garden all day", "quantum computing uses qubits"]
    bm25 = BM25([tokenize(d) for d in docs])
    assert bm25.search(tokenize("qubits"), 3)[0] == [2]
    assert bm25.search(tokenize("cat"), 3)[0] == [0, 1]
    assert bm25.search(tokenize("zebra"), 3)[0] == []


def test_rrf_and_hybrid_index():
    assert rrf([[1, 2, 3], [3, 1, 4]], top=2) == [1, 3]
    texts = ["refund policy for damaged items", "shipping times to Dhaka", "how to reset a password"]
    vectors = np.eye(3, dtype=np.float32)
    index = HybridIndex(texts, vectors)
    assert index.search(text="refund")[0] == 0
    assert index.search(vector=vectors[2])[0] == 2
    assert index.search(text="shipping", vector=vectors[1])[0] == 1
