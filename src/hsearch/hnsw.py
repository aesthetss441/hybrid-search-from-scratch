"""Hierarchical Navigable Small World graphs (Malkov & Yashunin, 2018).

The index is a stack of proximity graphs. Layer 0 holds every vector; each
higher layer holds an exponentially thinner random sample. A query starts at
the top, greedily walks towards the query, drops a layer, and repeats, so the
upper layers act as express lanes and the search is roughly logarithmic.

This implementation follows the paper's Algorithms 1-5 and favours clarity over
speed: the graph lives in Python lists and only distance computations are
vectorised.
"""
from __future__ import annotations

import heapq
import math
import random

import numpy as np


class HNSW:
    def __init__(self, dim: int, M: int = 16, ef_construction: int = 200, metric: str = "l2", seed: int = 0):
        if metric not in ("l2", "cosine"):
            raise ValueError("metric must be 'l2' or 'cosine'")
        self.dim, self.M, self.M0 = dim, M, 2 * M  # layer 0 is denser, as in the paper
        self.ef_construction = ef_construction
        self.metric = metric
        self.level_mult = 1.0 / math.log(M)
        self.vectors = np.empty((1024, dim), dtype=np.float32)
        self.size = 0
        # links[node][layer] is the adjacency list of `node` on `layer`
        self.links: list[list[list[int]]] = []
        self.entry = -1
        self.max_level = -1
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return self.size

    # ------------------------------------------------------------ distances
    def _prepare(self, v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=np.float32).reshape(self.dim)
        if self.metric == "cosine":
            v = v / (np.linalg.norm(v) or 1.0)  # unit vectors: cosine distance = 1 - dot
        return v

    def _dist(self, q: np.ndarray, ids: list[int]) -> np.ndarray:
        block = self.vectors[ids]
        if self.metric == "cosine":
            return 1.0 - block @ q
        diff = block - q
        return np.einsum("ij,ij->i", diff, diff)  # squared L2 keeps the ordering and skips the sqrt

    # ------------------------------------------------------------ Algorithm 2
    def _search_layer(self, q: np.ndarray, entry_points: list[int], ef: int, layer: int) -> list[tuple[float, int]]:
        """Best-first search on one layer. Returns up to `ef` (distance, id) pairs, nearest first."""
        visited = set(entry_points)
        dists = self._dist(q, entry_points).tolist()
        candidates = list(zip(dists, entry_points))  # min-heap: closest unexplored node first
        heapq.heapify(candidates)
        best = [(-d, e) for d, e in candidates]  # max-heap of the ef best seen: worst on top
        heapq.heapify(best)
        while len(best) > ef:
            heapq.heappop(best)
        while candidates:
            d, node = heapq.heappop(candidates)
            if d > -best[0][0] and len(best) >= ef:
                break  # every remaining candidate is farther than our current worst result
            fresh = [n for n in self.links[node][layer] if n not in visited]
            if not fresh:
                continue
            visited.update(fresh)
            for dn, n in zip(self._dist(q, fresh).tolist(), fresh):
                if len(best) < ef or dn < -best[0][0]:
                    heapq.heappush(candidates, (dn, n))
                    heapq.heappush(best, (-dn, n))
                    if len(best) > ef:
                        heapq.heappop(best)
        return sorted((-d, n) for d, n in best)

    # ------------------------------------------------------------ Algorithm 4
    def _select_neighbors(self, candidates: list[tuple[float, int]], m: int) -> list[int]:
        """Pick up to `m` neighbours that are close *and* spread out.

        A candidate is kept only if it is closer to the base point than to every
        neighbour already kept. Plain "take the m closest" would link a point only
        into its own cluster; this heuristic keeps edges that bridge clusters,
        which is what makes the graph navigable. Skipped candidates backfill any
        free slots.
        """
        kept: list[int] = []
        skipped: list[int] = []
        for d, c in sorted(candidates):
            if len(kept) >= m:
                break
            if not kept or bool((self._dist(self.vectors[c], kept) > d).all()):
                kept.append(c)
            else:
                skipped.append(c)
        return kept + skipped[: m - len(kept)]

    # ------------------------------------------------------------ Algorithm 1
    def add(self, vector: np.ndarray) -> int:
        q = self._prepare(vector)
        if self.size == len(self.vectors):
            self.vectors = np.concatenate([self.vectors, np.empty_like(self.vectors)])
        node = self.size
        self.vectors[node] = q
        self.size += 1
        level = int(-math.log(1.0 - self._rng.random()) * self.level_mult)
        self.links.append([[] for _ in range(level + 1)])

        if self.entry < 0:
            self.entry, self.max_level = node, level
            return node

        entry_points = [self.entry]
        for layer in range(self.max_level, level, -1):  # express lanes: greedy descent
            entry_points = [self._search_layer(q, entry_points, 1, layer)[0][1]]
        for layer in range(min(level, self.max_level), -1, -1):
            found = self._search_layer(q, entry_points, self.ef_construction, layer)
            neighbors = self._select_neighbors(found, self.M)
            self.links[node][layer] = neighbors
            cap = self.M0 if layer == 0 else self.M
            for n in neighbors:  # edges are bidirectional; prune a neighbour that is now over capacity
                adjacency = self.links[n][layer]
                adjacency.append(node)
                if len(adjacency) > cap:
                    scored = list(zip(self._dist(self.vectors[n], adjacency).tolist(), adjacency))
                    self.links[n][layer] = self._select_neighbors(scored, cap)
            entry_points = [n for _, n in found]
        if level > self.max_level:
            self.entry, self.max_level = node, level
        return node

    def add_batch(self, vectors: np.ndarray) -> None:
        for v in vectors:
            self.add(v)

    # ------------------------------------------------------------ Algorithm 5
    def search(self, query: np.ndarray, k: int = 10, ef: int = 50) -> tuple[list[int], list[float]]:
        """Approximate k nearest neighbours. Larger `ef` trades speed for recall."""
        if self.entry < 0:
            return [], []
        q = self._prepare(query)
        entry_points = [self.entry]
        for layer in range(self.max_level, 0, -1):
            entry_points = [self._search_layer(q, entry_points, 1, layer)[0][1]]
        found = self._search_layer(q, entry_points, max(ef, k), 0)[:k]
        return [n for _, n in found], [d for d, _ in found]
