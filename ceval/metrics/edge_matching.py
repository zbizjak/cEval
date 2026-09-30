"""Edge and branch matching rules.

Definitions and criticism: see metrics.md.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from .base import branch_paths, f1, resample_path


def _candidate_pairs(pred_pos, pred_edges, gt_pos, gt_edges, tol):
    """Edge pairs close enough to possibly match, as (i, j) index arrays.

    The full cost matrix is |E_pred| x |E_gt|, which is a million entries for a
    thousand-node graph and dominates the runtime. A pair can only satisfy a
    summed-endpoint threshold of 2*tol if the midpoints are within tol plus half
    the two edge lengths, so a radius query on midpoints discards almost
    everything without changing any result.
    """
    pm = 0.5 * (pred_pos[pred_edges[:, 0]] + pred_pos[pred_edges[:, 1]])
    gm = 0.5 * (gt_pos[gt_edges[:, 0]] + gt_pos[gt_edges[:, 1]])
    plen = np.linalg.norm(pred_pos[pred_edges[:, 0]] - pred_pos[pred_edges[:, 1]], axis=1)
    glen = np.linalg.norm(gt_pos[gt_edges[:, 0]] - gt_pos[gt_edges[:, 1]], axis=1)
    radius = tol + 0.5 * (plen.max() + glen.max())

    neighbours = cKDTree(gm).query_ball_point(pm, r=radius)
    i = np.repeat(np.arange(len(pm)), [len(n) for n in neighbours])
    j = np.fromiter((x for n in neighbours for x in n), dtype=np.int64,
                    count=len(i))
    return i, j


def _pair_cost(pred_pos, pred_edges, gt_pos, gt_edges, i, j):
    """Summed endpoint distance for the given pairs, over both orientations."""
    ps, pd = pred_pos[pred_edges[i, 0]], pred_pos[pred_edges[i, 1]]
    gs, gd = gt_pos[gt_edges[j, 0]], gt_pos[gt_edges[j, 1]]
    n = np.linalg.norm
    return np.minimum(n(ps - gs, axis=1) + n(pd - gd, axis=1),
                      n(ps - gd, axis=1) + n(pd - gs, axis=1))


def edge_m2m_f1(pred_pos, pred_edges, gt_pos, gt_edges, tol):
    """Many-to-many matching on summed endpoint distance <= 2*tol."""
    if not len(pred_edges) or not len(gt_edges):
        return {"value": 0.0}
    i, j = _candidate_pairs(pred_pos, pred_edges, gt_pos, gt_edges, tol)
    if not len(i):
        return {"value": 0.0, "precision": 0.0, "recall": 0.0}
    ok = _pair_cost(pred_pos, pred_edges, gt_pos, gt_edges, i, j) <= 2 * tol
    prec = len(np.unique(i[ok])) / len(pred_edges)
    rec = len(np.unique(j[ok])) / len(gt_edges)
    return {"value": f1(prec, rec), "precision": prec, "recall": rec}


def edge_hungarian_f1(pred_pos, pred_edges, gt_pos, gt_edges, tol):
    """One-to-one assignment on the same cost.

    Capped at min(|E_pred|,|E_gt|)/|E_gt| by construction, so a coarser
    prediction cannot score above its edge-count ratio however good it is.
    """
    if not len(pred_edges) or not len(gt_edges):
        return {"value": 0.0}
    i, j = _candidate_pairs(pred_pos, pred_edges, gt_pos, gt_edges, tol)
    cost = (_pair_cost(pred_pos, pred_edges, gt_pos, gt_edges, i, j)
            if len(i) else np.zeros(0))
    keep = cost <= 2 * tol
    i, j, cost = i[keep], j[keep], cost[keep]

    n_matched = 0
    if len(i):
        # Solve the assignment only over edges that have at least one candidate.
        pi, i_local = np.unique(i, return_inverse=True)
        gj, j_local = np.unique(j, return_inverse=True)
        big = 1e6
        c = np.full((len(pi), len(gj)), big)
        c[i_local, j_local] = cost
        ri, ci = linear_sum_assignment(c)
        n_matched = int((c[ri, ci] < big).sum())

    prec, rec = n_matched / len(pred_edges), n_matched / len(gt_edges)
    return {"value": f1(prec, rec), "precision": prec, "recall": rec,
            "structural_cap": min(len(pred_edges), len(gt_edges)) / len(gt_edges)}


def edge_midpoint_f1(pred_pos, pred_edges, gt_pos, gt_edges, tol):
    """Edge midpoints matched within tol."""
    if not len(pred_edges) or not len(gt_edges):
        return {"value": 0.0}
    pm = 0.5 * (pred_pos[pred_edges[:, 0]] + pred_pos[pred_edges[:, 1]])
    gm = 0.5 * (gt_pos[gt_edges[:, 0]] + gt_pos[gt_edges[:, 1]])
    prec = float((cKDTree(gm).query(pm)[0] <= tol).mean())
    rec = float((cKDTree(pm).query(gm)[0] <= tol).mean())
    return {"value": f1(prec, rec), "precision": prec, "recall": rec}


def edge_box_iou_f1(pred_pos, pred_edges, gt_pos, gt_edges, tol, thresh=0.5):
    """Relationformer-style: edges as boxes, matched at IoU >= thresh."""
    if not len(pred_edges) or not len(gt_edges):
        return {"value": 0.0}

    def boxes(pos, edges):
        s, d = pos[edges[:, 0]], pos[edges[:, 1]]
        lo, hi = np.minimum(s, d), np.maximum(s, d)
        pad = np.maximum(0.0, tol / 2 - (hi - lo)) / 2
        return lo - pad, hi + pad

    plo, phi = boxes(pred_pos, pred_edges)
    glo, ghi = boxes(gt_pos, gt_edges)

    # Boxes that do not even come close cannot overlap, so only nearby pairs
    # are scored — the full matrix is the dominant cost otherwise.
    i, j = _candidate_pairs(pred_pos, pred_edges, gt_pos, gt_edges, tol)
    if not len(i):
        return {"value": 0.0, "precision": 0.0, "recall": 0.0}

    inter = np.prod(np.clip(np.minimum(phi[i], ghi[j])
                            - np.maximum(plo[i], glo[j]), 0, None), axis=1)
    vp, vg = np.prod(phi - plo, 1), np.prod(ghi - glo, 1)
    iou = inter / np.maximum(vp[i] + vg[j] - inter, 1e-12)

    ok = iou >= thresh
    prec = len(np.unique(i[ok])) / len(pred_edges)
    rec = len(np.unique(j[ok])) / len(gt_edges)
    return {"value": f1(prec, rec), "precision": prec, "recall": rec}


def branch_coverage_f1(pred_pos, pred_edges, gt_pos, gt_edges, tol, coverage=0.8):
    """A branch is detected when >= coverage of its points lie within tol."""
    if not len(pred_edges) or not len(gt_edges):
        return {"value": 0.0}
    pb = [resample_path(pred_pos, p, tol / 2)
          for p in branch_paths(pred_pos, pred_edges)]
    gb = [resample_path(gt_pos, p, tol / 2)
          for p in branch_paths(gt_pos, gt_edges)]
    if not pb or not gb:
        return {"value": 0.0}
    pt, gt_tree = cKDTree(np.vstack(pb)), cKDTree(np.vstack(gb))
    rec = float(np.mean([(pt.query(g)[0] <= tol).mean() >= coverage for g in gb]))
    prec = float(np.mean([(gt_tree.query(p)[0] <= tol).mean() >= coverage for p in pb]))
    return {"value": f1(prec, rec), "precision": prec, "recall": rec,
            "n_pred_branches": len(pb), "n_gt_branches": len(gb)}
