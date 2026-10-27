"""Metrics defined on the nodes of the two graphs.

These compare node positions, not curves, so their value depends on where each
graph places its nodes along the same curve: two graphs that trace one curve
with different node spacing score below perfect. They are reported as a
separate group for that reason.
"""
import numpy as np
from scipy.spatial import cKDTree

from .base import f1


def node_f1(pred_pos, pred_edges, gt_pos, gt_edges, tol):
    """Point-to-point: a node is matched if another node lies within tol."""
    if not len(pred_pos) or not len(gt_pos):
        return {"value": 0.0, "precision": 0.0, "recall": 0.0}
    prec = float((cKDTree(gt_pos).query(pred_pos)[0] < tol).mean())
    rec = float((cKDTree(pred_pos).query(gt_pos)[0] < tol).mean())
    return {"value": f1(prec, rec), "precision": prec, "recall": rec}


def node_chamfer(pred_pos, pred_edges, gt_pos, gt_edges, tol):
    """Mean of the two directed mean nearest-node distances, in mm."""
    if not len(pred_pos) or not len(gt_pos):
        return {"value": float("inf")}
    p2g = cKDTree(gt_pos).query(pred_pos)[0]
    g2p = cKDTree(pred_pos).query(gt_pos)[0]
    return {"value": float(p2g.mean() + g2p.mean()) / 2}
