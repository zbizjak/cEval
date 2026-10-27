"""Run every metric on two centerline graphs and build the JSON report.

Every metric is computed on the graphs exactly as given. The curve metrics
measure the curves themselves and do not depend on where either graph places
its nodes; the structural and shape metrics depend only on the junctions,
endpoints and branches. The node metrics (UNSUITABLE) compare node positions
and are reported separately, because their value changes when only the node
spacing changes.
"""
from datetime import datetime, timezone

import numpy as np

from .metrics import connectivity, curve, morphology, nodes, topology

VERSION = "0.2"

# name -> (function, family, lower_is_better)
METRICS = {
    # curve coverage within tau
    "edge_F1":            (curve.edge_f1, "coverage", False),
    "edge_R":             (curve.edge_r, "coverage", False),
    "overlap_OV":         (curve.overlap, "coverage", False),
    "branch_coverage_F1_80": (curve.branch_coverage_f1_80, "coverage", False),
    "branch_coverage_F1": (curve.branch_coverage_f1, "coverage", False),
    # distances along the curves
    "ASCD":               (curve.ascd, "distance", True),
    "HD95":               (curve.hd95, "distance", True),
    "netmets_geom_F1":    (curve.netmets_geometry, "distance", False),
    # paths
    "APLS":               (curve.apls, "path", False),
    # junctions, endpoints and connections between them
    "junction_F1":        (topology.junction_f1, "topology", False),
    "endpoint_F1":        (topology.endpoint_f1, "topology", False),
    "netmets_conn_F1":    (connectivity.netmets_connectivity, "topology", False),
    # counts
    "beta0_err":          (topology.beta0_err, "count", True),
    "beta1_err":          (topology.beta1_err, "count", True),
    "branch_count_ratio": (topology.branch_count_ratio, "count", True),
}

# Defined on nodes: their value depends on node spacing, so they are reported
# apart and kept out of any selection of metrics.
UNSUITABLE = {
    "node_F1":            (nodes.node_f1, "nodes", False),
    "chamfer":            (nodes.node_chamfer, "nodes", True),
    "false_merges":       (connectivity.false_merges, "nodes", True),
    "false_splits":       (connectivity.false_splits, "nodes", True),
}

# Shape descriptors, computed for each graph; reported as |pred - reference|.
DESCRIPTORS = ["Len_tot", "Len_br", "tortuosity", "Ang_bif"]

# Metrics that need information a centerline graph does not carry, listed so
# the omission is visible.
EXCLUDED = {
    "Dice": "needs a segmentation mask",
    "clDice": "needs a mask; edge_F1 is its graph-to-graph counterpart",
    "cbDice": "needs a mask and its distance transform",
    "ccDice": "needs a mask",
    "NSD": "needs surfaces",
    "surface_HD95": "needs surfaces; HD95 here is curve to curve",
    "Betti_matching": "defined on cubical complexes of masks",
    "CAT08 OT / AI": "need the vessel radius at every point",
    "CAT08 OF": "needs single ordered vessels from a designated root",
    "DIADEM": "needs a rooted, directed tree",
    "TOPO / OPT-G": "too slow at this scale",
    "NRI": "needs synapse annotations",
}


def _clean(value):
    """JSON has no NaN or infinity."""
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (int, str, bool)) or value is None:
        return value
    v = float(value)
    return v if np.isfinite(v) else None


def evaluate(pred_pos, pred_edges, gt_pos, gt_edges, tau):
    """All metrics at tolerance tau (mm). Returns {name: {value, ...}}."""
    out = {}
    for group, table in (("main", METRICS), ("unsuitable", UNSUITABLE)):
        for name, (fn, family, lower) in table.items():
            try:
                res = fn(pred_pos, pred_edges, gt_pos, gt_edges, tau)
            except Exception as exc:
                res = {"value": float("nan"), "error": str(exc)[:200]}
            res = _clean(res)
            res.update(family=family, lower_is_better=lower, group=group)
            out[name] = res
    for name, res in morphology.morphology(
            pred_pos, pred_edges, gt_pos, gt_edges, tau).items():
        res = _clean(res)
        res.update(family="descriptor",
                   group="main" if name in DESCRIPTORS else "similar")
        out[name] = res
    return out


def build(pred_pos, pred_edges, gt_pos, gt_edges, tau, pred_name=None, gt_name=None):
    """The full JSON-serialisable report."""
    return {
        "ceval_version": VERSION,
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "inputs": {
            "prediction": pred_name,
            "reference": gt_name,
            "units": "mm",
            "n_nodes_pred": int(len(pred_pos)),
            "n_edges_pred": int(len(pred_edges)),
            "n_nodes_gt": int(len(gt_pos)),
            "n_edges_gt": int(len(gt_edges)),
        },
        "tolerance": {
            "tau_mm": float(tau),
            "definition": "dataset mean voxel diagonal",
        },
        "metrics": evaluate(pred_pos, pred_edges, gt_pos, gt_edges, tau),
        "excluded": EXCLUDED,
    }


def summary(report, digits=4):
    """One number per metric: the main metrics grouped by family, then the
    node metrics. Descriptors are given as |pred - reference|."""
    def one(res):
        v = res["abs_error"] if "abs_error" in res else res.get("value")
        return None if v is None else round(float(v), digits)

    groups = {}
    for name, res in report["metrics"].items():
        if res["group"] == "main":
            groups.setdefault(res["family"], {})[name] = one(res)
        elif res["group"] == "unsuitable":
            groups.setdefault("nodes", {})[name] = one(res)
    return {
        "ceval_version": report["ceval_version"],
        "prediction": report["inputs"]["prediction"],
        "reference": report["inputs"]["reference"],
        "tolerance_mm": report["tolerance"]["tau_mm"],
        "metrics": groups,
    }
