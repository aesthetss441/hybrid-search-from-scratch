"""Lexical + vector search fused with reciprocal rank fusion."""
from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np

from .bm25 import BM25, tokenize
from .hnsw import HNSW


def rrf(rankings: Sequence[Sequence[int]], k: int = 60, top: int = 10) -> list[int]:
    """Reciprocal rank fusion (Cormack et al., 2009): score(d) = sum over lists of 1 / (k + rank).

    It uses ranks only, so BM25 scores and vector distances never need to be put
    on a common scale.
    """
    scores: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            scores[doc] += 1.0 / (k + rank)
    return sorted(scores, key=lambda d: (-scores[d], d))[:top]


class HybridIndex:
    """Documents searchable by keywords, by embedding, or both."""

    def __init__(self, texts: list[str], vectors: np.ndarray, metric: str = "cosine", M: int = 16, ef_construction: int = 200):
        if len(texts) != len(vectors):
            raise ValueError("texts and vectors must have the same length")
        self.texts = texts
        self.bm25 = BM25([tokenize(t) for t in texts])
        self.hnsw = HNSW(vectors.shape[1], M=M, ef_construction=ef_construction, metric=metric)
        self.hnsw.add_batch(vectors)

    def search(self, text: str | None = None, vector: np.ndarray | None = None, k: int = 10, depth: int = 50) -> list[int]:
        rankings = []
        if text:
            rankings.append(self.bm25.search(tokenize(text), depth)[0])
        if vector is not None:
            rankings.append(self.hnsw.search(vector, depth, ef=max(depth, 50))[0])
        return rrf(rankings, top=k)
