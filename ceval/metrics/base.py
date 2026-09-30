"""Shared helpers for the metrics.

Graph contract: points (N,3) float64 in mm, edges (E,2) int64 undirected.
"""
from collections import defaultdict

import numpy as np


def f1(precision, recall):
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def adjacency(n_nodes, edges):
    adj = defaultdict(set)
    for u, v in edges:
        adj[int(u)].add(int(v))
        adj[int(v)].add(int(u))
    return adj


def components(n_nodes, edges):
    """Node indices per connected component."""
    parent = list(range(n_nodes))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for u, v in edges:
        a, b = find(int(u)), find(int(v))
        if a != b:
            parent[a] = b
    roots = np.array([find(i) for i in range(n_nodes)])
    return [np.flatnonzero(roots == r) for r in np.unique(roots)]


def branch_paths(points, edges):
    """Node paths between degree != 2 nodes.

    The decomposition is a hidden parameter of every branch-level metric:
    ATM'22 states its 80% rule precisely but never how a skeleton is split.
    """
    if not len(edges):
        return []
    adj = adjacency(len(points), edges)
    key = {i for i in range(len(points)) if len(adj[i]) != 2}
    if not key:                       # pure cycle: break it somewhere
        key = {int(edges[0][0])}

    seen, paths = set(), []
    for start in sorted(key):
        for nb in sorted(adj[start]):
            if (min(start, nb), max(start, nb)) in seen:
                continue
            seen.add((min(start, nb), max(start, nb)))
            path, prev, cur = [start, nb], start, nb
            while cur not in key:
                nxt = adj[cur] - {prev}
                if not nxt:
                    break
                nx = min(nxt)
                seen.add((min(cur, nx), max(cur, nx)))
                path.append(nx)
                prev, cur = cur, nx
            paths.append(path)
    return paths


def resample_path(points, path, step):
    """Sample a node path uniformly by arc length.

    Used to approximate a polyline for distance queries, not to rewrite the
    graph: cEval checks that two graphs are comparably sampled and refuses if
    they are not, rather than resampling them itself. Resampling a graph is a
    decision about the data — keeping a cycle formed by two parallel branches,
    for one, needs a rule only its owner can choose — and making it silently
    would hide the confound this tool exists to expose.
    """
    q = points[np.asarray(path)]
    seg = np.linalg.norm(np.diff(q, axis=0), axis=1)
    total = seg.sum()
    if total < 1e-9:
        return q[:1]
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    want = np.linspace(0.0, total, max(2, int(np.ceil(total / step)) + 1))
    i = np.clip(np.searchsorted(cum, want, "right") - 1, 0, len(seg) - 1)
    t = ((want - cum[i]) / np.maximum(seg[i], 1e-12))[:, None]
    return q[i] + t * (q[i + 1] - q[i])


def sample_graph(points, edges, step):
    """Point cloud along every edge, sampled by arc length.

    Sampling per unit length rather than per edge matters: a fixed count per
    edge makes the cloud density depend on how the graph was discretised, which
    is the confound these metrics exist to avoid.
    """
    if not len(edges):
        return points.copy()
    out = []
    for p0, p1 in zip(points[edges[:, 0]], points[edges[:, 1]]):
        k = max(2, int(np.ceil(np.linalg.norm(p1 - p0) / step)) + 1)
        t = np.linspace(0, 1, k)[:, None]
        out.append(p0 * (1 - t) + p1 * t)
    return np.unique(np.round(np.concatenate(out), 6), axis=0)


def spacing(points, edges):
    """Median distance between adjacent nodes."""
    if not len(edges):
        return float("nan")
    return float(np.median(np.linalg.norm(
        points[edges[:, 0]] - points[edges[:, 1]], axis=1)))


def densify(points, edges, max_edge_mm):
    """Insert nodes along edges so that no edge is longer than max_edge_mm.

    Every original node keeps its exact position and every original edge is
    replaced by a chain of collinear pieces, so the traced curve, the branch
    decomposition, the degree of every node and both Betti numbers are
    unchanged: this adds sampling, it does not resample. Its purpose is to put
    two graphs on comparable node densities before metrics that pair discrete
    elements are computed, without moving a node either method actually
    reported.

    A metric that matches within a tolerance tau can only be satisfied
    reliably when both graphs carry a node at least every tau; below that the
    score reports the discretisation rather than the reconstruction.
    """
    pts = [np.asarray(p, dtype=np.float64) for p in points]
    out = []
    for u, v in edges:
        u, v = int(u), int(v)
        a, b = pts[u], pts[v]
        n_piece = max(1, int(np.ceil(np.linalg.norm(b - a) / max_edge_mm)))
        prev = u
        for t in range(1, n_piece):
            pts.append(a + (b - a) * (t / n_piece))
            out.append((prev, len(pts) - 1))
            prev = len(pts) - 1
        out.append((prev, v))
    return (np.asarray(pts, dtype=np.float64),
            np.asarray(out, dtype=np.int64).reshape(-1, 2))
