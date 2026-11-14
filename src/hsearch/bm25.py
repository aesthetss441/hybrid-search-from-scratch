"""Okapi BM25 over an inverted index.

Each posting stores the document's finished score contribution for that term,
so answering a query is one vectorised add per query term.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

import numpy as np

_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25:
    def __init__(self, documents: list[list[str]], k1: float = 1.2, b: float = 0.75):
        self.n = len(documents)
        lengths = np.array([len(d) for d in documents], dtype=np.float32)
        norm = k1 * (1.0 - b + b * lengths / max(float(lengths.mean()) if self.n else 0.0, 1e-9))
        postings: dict[str, tuple[list[int], list[int]]] = defaultdict(lambda: ([], []))
        for doc_id, tokens in enumerate(documents):
            for term, tf in Counter(tokens).items():
                ids, tfs = postings[term]
                ids.append(doc_id)
                tfs.append(tf)
        self.postings: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for term, (ids, tfs) in postings.items():
            idx = np.asarray(ids, dtype=np.int64)
            tf = np.asarray(tfs, dtype=np.float32)
            idf = math.log(1.0 + (self.n - len(ids) + 0.5) / (len(ids) + 0.5))
            self.postings[term] = (idx, idf * tf * (k1 + 1.0) / (tf + norm[idx]))

    def scores(self, query: list[str]) -> np.ndarray:
        out = np.zeros(self.n, dtype=np.float32)
        for term in set(query):
            hit = self.postings.get(term)
            if hit:
                out[hit[0]] += hit[1]
        return out

    def search(self, query: list[str], k: int = 10) -> tuple[list[int], list[float]]:
        scores = self.scores(query)
        k = min(k, self.n)
        top = np.argpartition(-scores, k - 1)[:k] if k < self.n else np.arange(self.n)
        top = top[np.argsort(-scores[top], kind="stable")]
        top = top[scores[top] > 0]
        return top.tolist(), scores[top].tolist()
