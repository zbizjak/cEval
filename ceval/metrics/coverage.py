"""Coverage and point-cloud distances.

Definitions and criticism: see metrics.md.
"""
import numpy as np
from scipy.spatial import cKDTree

from .base import branch_paths, f1, resample_path, sample_graph


def cl_coverage(pred_pos, pred_edges, gt_pos, gt_edges, tol):
    """Fraction of sampled points within tol of the other graph.

    NOT clDice: true clDice measures skeleton against mask, so its tolerance is
    the local vessel calibre. With a global tolerance this is stricter on thick
    vessels and looser on thin ones.
    """
    if not len(pred_pos) or not len(gt_pos):
        return {"value": 0.0}
    p = sample_graph(pred_pos, pred_edges, tol / 4)
    g = sample_graph(gt_pos, gt_edges, tol / 4)
    prec = float((cKDTree(g).query(p)[0] < tol).mean())
    rec = float((cKDTree(p).query(g)[0] < tol).mean())
    return {"value": f1(prec, rec), "precision": prec, "recall": rec}


def cov_len(pred_pos, pred_edges, gt_pos, gt_edges, tol):
    """Fraction of reference length lying within tol of the prediction."""
    if not len(gt_edges) or not len(pred_edges):
        return {"value": 0.0}
    tree = cKDTree(sample_graph(pred_pos, pred_edges, tol / 4))
    covered = total = 0.0
    for path in branch_paths(gt_pos, gt_edges):
        q = gt_pos[np.asarray(path)]
        seg = np.linalg.norm(np.diff(q, axis=0), axis=1)
        if not len(seg):
            continue
        hit = tree.query(0.5 * (q[:-1] + q[1:]))[0] < tol
        covered += float(seg[hit].sum())
        total += float(seg.sum())
    return {"value": covered / total if total else 0.0, "gt_length_mm": total}


def smd(pred_pos, pred_edges, gt_pos, gt_edges, tol):
    """Mean symmetric nearest-point distance between sampled clouds, in mm."""
    if not len(pred_pos) or not len(gt_pos):
        return {"value": float("inf")}
    p = sample_graph(pred_pos, pred_edges, tol / 4)
    g = sample_graph(gt_pos, gt_edges, tol / 4)
    p2g = cKDTree(g).query(p)[0]
    g2p = cKDTree(p).query(g)[0]
    return {"value": float(p2g.mean() + g2p.mean()) / 2,
            "p2g_mm": float(p2g.mean()), "g2p_mm": float(g2p.mean())}


def overlap_until_first_error(pred_pos, pred_edges, gt_pos, gt_edges, tol):
    """CAT08 'OF': fraction of each branch traced before the first miss.

    CAT08 uses the annotated vessel radius as its tolerance and orders branches
    from the ostium; without a radius channel or a root, tol and an arbitrary
    branch start are substituted.
    """
    if not len(gt_edges) or not len(pred_edges):
        return {"value": 0.0}
    tree = cKDTree(sample_graph(pred_pos, pred_edges, tol / 4))
    fracs, weights = [], []
    for path in branch_paths(gt_pos, gt_edges):
        pts = resample_path(gt_pos, path, tol / 2)
        if len(pts) < 2:
            continue
        ok = tree.query(pts)[0] < tol
        first_bad = len(ok) if ok.all() else int(np.argmin(ok))
        q = gt_pos[np.asarray(path)]
        fracs.append(first_bad / len(ok))
        weights.append(float(np.linalg.norm(np.diff(q, axis=0), axis=1).sum()))
    if not fracs:
        return {"value": 0.0}
    return {"value": float(np.average(fracs, weights=weights)),
            "n_branches": len(fracs)}
