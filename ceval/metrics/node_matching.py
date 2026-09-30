"""Node matching: are predicted nodes where the reference nodes are?

Definitions and criticism: see metrics.md.
"""
import numpy as np
from scipy.spatial import cKDTree

from .base import f1, sample_graph


def node_f1(pred_pos, pred_edges, gt_pos, gt_edges, tol):
    """Point-to-point: a node is matched if another node lies within tol."""
    if not len(pred_pos) or not len(gt_pos):
        return {"value": 0.0, "precision": 0.0, "recall": 0.0}
    prec = float((cKDTree(gt_pos).query(pred_pos)[0] < tol).mean())
    rec = float((cKDTree(pred_pos).query(gt_pos)[0] < tol).mean())
    return {"value": f1(prec, rec), "precision": prec, "recall": rec}


def node_precision(pred_pos, pred_edges, gt_pos, gt_edges, tol):
    """Point-to-point precision alone, as its own top-level metric entry —
    node_f1 already computes this internally, this just exposes it so a
    caller doesn't have to reach into node_F1's nested result to get it."""
    return {"value": node_f1(pred_pos, pred_edges, gt_pos, gt_edges, tol)["precision"]}


def node_recall(pred_pos, pred_edges, gt_pos, gt_edges, tol):
    """Point-to-point recall alone; see node_precision."""
    return {"value": node_f1(pred_pos, pred_edges, gt_pos, gt_edges, tol)["recall"]}


def node_p2c_f1(pred_pos, pred_edges, gt_pos, gt_edges, tol):
    """Point-to-curve: distance to the other graph's polyline, not its nodes."""
    if not len(pred_pos) or not len(gt_pos):
        return {"value": 0.0, "precision": 0.0, "recall": 0.0}
    gt_curve = sample_graph(gt_pos, gt_edges, tol / 4)
    pred_curve = sample_graph(pred_pos, pred_edges, tol / 4)
    prec = float((cKDTree(gt_curve).query(pred_pos)[0] < tol).mean())
    rec = float((cKDTree(pred_curve).query(gt_pos)[0] < tol).mean())
    return {"value": f1(prec, rec), "precision": prec, "recall": rec}


def node_hd95(pred_pos, pred_edges, gt_pos, gt_edges, tol):
    """95th percentile of the pooled node-to-nearest-node distances, in mm."""
    if not len(pred_pos) or not len(gt_pos):
        return {"value": float("inf")}
    d = np.concatenate([cKDTree(gt_pos).query(pred_pos)[0],
                        cKDTree(pred_pos).query(gt_pos)[0]])
    return {"value": float(np.percentile(d, 95))}


def node_chamfer(pred_pos, pred_edges, gt_pos, gt_edges, tol):
    """Mean of the two directed mean nearest-node distances, in mm."""
    if not len(pred_pos) or not len(gt_pos):
        return {"value": float("inf")}
    p2g = cKDTree(gt_pos).query(pred_pos)[0]
    g2p = cKDTree(pred_pos).query(gt_pos)[0]
    return {"value": float(p2g.mean() + g2p.mean()) / 2}
