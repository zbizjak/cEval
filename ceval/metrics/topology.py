"""Graph topology counts.

Betti numbers here are computed on the graph (b1 = E - N + b0), not on voxels,
so no 6/18/26-connectivity convention is involved and the values are not
comparable with mask-level Betti errors.

Definitions and criticism: see metrics.md.
"""
import numpy as np
from scipy.spatial import cKDTree

from .base import adjacency, components, f1


def _betti(n_nodes, edges):
    if not n_nodes:
        return 0, 0
    b0 = len(components(n_nodes, edges))
    return b0, int(len(edges) - n_nodes + b0)


def beta0_err(pred_pos, pred_edges, gt_pos, gt_edges, tol):
    """|components(pred) - components(gt)|."""
    p = _betti(len(pred_pos), pred_edges)[0]
    g = _betti(len(gt_pos), gt_edges)[0]
    return {"value": float(abs(p - g)), "pred": p, "gt": g}


def beta1_err(pred_pos, pred_edges, gt_pos, gt_edges, tol):
    """|cycles(pred) - cycles(gt)|. A spurious and a missing loop cancel."""
    p = _betti(len(pred_pos), pred_edges)[1]
    g = _betti(len(gt_pos), gt_edges)[1]
    return {"value": float(abs(p - g)), "pred": p, "gt": g}


def _nodes_of_degree(pos, edges, keep):
    adj = adjacency(len(pos), edges)
    idx = [i for i in range(len(pos)) if keep(len(adj[i]))]
    return pos[idx] if idx else np.zeros((0, 3))


def _point_f1(p, g, tol):
    if not len(p) or not len(g):
        return {"value": 1.0 if not len(p) and not len(g) else 0.0,
                "n_pred": len(p), "n_gt": len(g)}
    prec = float((cKDTree(g).query(p)[0] <= tol).mean())
    rec = float((cKDTree(p).query(g)[0] <= tol).mean())
    return {"value": f1(prec, rec), "precision": prec, "recall": rec,
            "n_pred": len(p), "n_gt": len(g)}


def junction_f1(pred_pos, pred_edges, gt_pos, gt_edges, tol):
    """Bifurcations (degree >= 3) matched within tol."""
    return _point_f1(_nodes_of_degree(pred_pos, pred_edges, lambda d: d >= 3),
                     _nodes_of_degree(gt_pos, gt_edges, lambda d: d >= 3), tol)


def endpoint_f1(pred_pos, pred_edges, gt_pos, gt_edges, tol):
    """Free ends (degree 1) matched within tol — where pruning shows up."""
    return _point_f1(_nodes_of_degree(pred_pos, pred_edges, lambda d: d == 1),
                     _nodes_of_degree(gt_pos, gt_edges, lambda d: d == 1), tol)
