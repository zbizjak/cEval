"""Morphometric descriptors.

Properties of one graph, reported for both sides so the reader compares them.
They answer whether the prediction is plausible, which no matching metric asks —
and they are distributional, so a graph can match the reference on average with
no node in the right place. Never sufficient alone.

Definitions and criticism: see metrics.md.
"""
import numpy as np

from .base import adjacency, branch_paths, components


def describe(pos, edges):
    """All descriptors of a single graph."""
    if not len(edges):
        return dict(N_cc=0.0, N_bif=0.0, Len_tot=0.0, Len_br=0.0,
                    Ang_bif=float("nan"), tortuosity=float("nan"))

    adj = adjacency(len(pos), edges)
    seg = np.linalg.norm(pos[edges[:, 0]] - pos[edges[:, 1]], axis=1)

    lengths, ratios = [], []
    for path in branch_paths(pos, edges):
        q = pos[np.asarray(path)]
        length = float(np.linalg.norm(np.diff(q, axis=0), axis=1).sum())
        chord = float(np.linalg.norm(q[-1] - q[0]))
        lengths.append(length)
        if chord > 1e-6:
            ratios.append(length / chord)

    angles = []
    for i in range(len(pos)):
        if len(adj[i]) < 3:
            continue
        dirs = []
        for nb in adj[i]:
            v = pos[nb] - pos[i]
            n = np.linalg.norm(v)
            if n > 1e-9:
                dirs.append(v / n)
        for a in range(len(dirs)):
            for b in range(a + 1, len(dirs)):
                angles.append(np.degrees(np.arccos(np.clip(dirs[a] @ dirs[b], -1, 1))))

    return dict(
        N_cc=float(len(components(len(pos), edges))),
        N_bif=float(sum(1 for i in range(len(pos)) if len(adj[i]) >= 3)),
        Len_tot=float(seg.sum()),
        Len_br=float(np.mean(lengths)) if lengths else 0.0,
        Ang_bif=float(np.mean(angles)) if angles else float("nan"),
        tortuosity=float(np.mean(ratios)) if ratios else float("nan"))


def morphology(pred_pos, pred_edges, gt_pos, gt_edges, tol):
    """Every descriptor, for prediction and reference, with the difference."""
    p, g = describe(pred_pos, pred_edges), describe(gt_pos, gt_edges)
    return {name: {"value": p[name], "gt_value": g[name],
                   "abs_error": abs(p[name] - g[name])}
            for name in p}
