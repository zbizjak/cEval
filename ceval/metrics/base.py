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


def spacing(points, edges):
    """Median distance between adjacent nodes (a description of the graph;
    no metric depends on it)."""
    if not len(edges):
        return float("nan")
    return float(np.median(np.linalg.norm(
        points[edges[:, 0]] - points[edges[:, 1]], axis=1)))


def greedy_match(a_pos, b_pos, tol):
    """One-to-one matching of points a to points b within tol, closest pairs first.

    Returns {index in a: index in b}. Used by the NetMets connectivity score on
    junctions and endpoints, and by false merges and false splits on all nodes.
    """
    from scipy.spatial import cKDTree
    if not len(a_pos) or not len(b_pos):
        return {}
    pairs = cKDTree(a_pos).sparse_distance_matrix(
        cKDTree(b_pos), tol, output_type="ndarray")
    out, used_b = {}, set()
    for i, j, _ in sorted(pairs.tolist(), key=lambda r: r[2]):
        i, j = int(i), int(j)
        if i in out or j in used_b:
            continue
        out[i] = j
        used_b.add(j)
    return out
