"""Run every metric on two centerline graphs and build the JSON report."""
from datetime import datetime, timezone

import numpy as np

from .metrics import (continuity, coverage, edge_matching, morphology,
                      node_matching, path_metrics, topology)
from .metrics.base import densify, spacing

VERSION = "0.1"

# name -> (function, family, lower_is_better)
METRICS = {
    "node_F1":            (node_matching.node_f1, "matching", False),
    "node_precision":     (node_matching.node_precision, "matching", False),
    "node_recall":        (node_matching.node_recall, "matching", False),
    "node_p2c_F1":        (node_matching.node_p2c_f1, "matching", False),
    "node_HD95":          (node_matching.node_hd95, "matching", True),
    "node_chamfer":       (node_matching.node_chamfer, "matching", True),
    "edge_m2m_F1":        (edge_matching.edge_m2m_f1, "matching", False),
    "edge_hung_F1":       (edge_matching.edge_hungarian_f1, "matching", False),
    "edge_midpoint_F1":   (edge_matching.edge_midpoint_f1, "matching", False),
    "edge_box_iou_F1":    (edge_matching.edge_box_iou_f1, "matching", False),
    "branch_coverage_F1": (edge_matching.branch_coverage_f1, "coverage", False),
    "branch_continuity":  (continuity.branch_continuity, "coverage", False),
    "branch_count_ratio": (continuity.branch_count_ratio, "coverage", True),
    "cl_coverage":        (coverage.cl_coverage, "coverage", False),
    "cov_len":            (coverage.cov_len, "coverage", False),
    "SMD":                (coverage.smd, "coverage", True),
    "overlap_first_error": (coverage.overlap_until_first_error, "coverage", False),
    "beta0_err":          (topology.beta0_err, "topology", True),
    "beta1_err":          (topology.beta1_err, "topology", True),
    "junction_F1":        (topology.junction_f1, "topology", False),
    "endpoint_F1":        (topology.endpoint_f1, "topology", False),
    "APLS":               (path_metrics.apls, "path", False),
}

# TOPO is implemented (path_metrics.topo, correct on the identity/missing-
# branch sanity checks) but its per-seed hole/marble matching does not scale
# to real-size graphs (still running after minutes on a 3000-node case) --
# excluded from the default set until that is fixed, rather than blocking the
# ablation sweep.
TOPO_METRICS = {"TOPO_F1": (path_metrics.topo, "path", False)}

# Metrics that need a segmentation volume or a radius channel, listed so the
# omission is visible rather than silent.
EXCLUDED = {
    "Dice": "needs a segmentation mask",
    "clDice": "needs a mask (Tprec = |S_P n V_L| / |S_P|); see cl_coverage",
    "cbDice": "needs a mask and its distance transform",
    "ccDice": "needs a mask",
    "NSD": "needs surfaces",
    "surface_HD95": "needs surfaces; node_HD95 is the graph analogue",
    "Betti_matching": "defined on cubical complexes of masks",
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


SPACING_TOLERANCE = 0.2   # prediction may differ from the reference by +/-20%


def spacing_check(pred_pos, pred_edges, gt_pos, gt_edges):
    """Are the two graphs sampled closely enough to be comparable?

    Node- and edge-matching metrics measure the difference between two sampling
    conventions as readily as they measure reconstruction error: a graph
    compared against a coarser copy of itself — no error whatsoever — scores
    0.05 on Hungarian edge F1 and 0.01 on box-IoU edge F1 in our measurements.

    cEval checks rather than repairs. Resampling a graph is a decision about the
    data: preserving a cycle formed by two parallel branches, for instance,
    needs a rule that only the owner of the data can sensibly choose, and
    applying it silently would hide exactly the confound this tool exists to
    expose. Whether to resample the prediction, the reference, or both is the
    caller's call.
    """
    sp_pred = spacing(pred_pos, pred_edges)
    sp_gt = spacing(gt_pos, gt_edges)
    ok = True
    ratio = float("nan")
    if np.isfinite(sp_pred) and np.isfinite(sp_gt) and sp_gt > 0:
        ratio = sp_pred / sp_gt
        ok = abs(ratio - 1.0) <= SPACING_TOLERANCE
    return {
        "passed": bool(ok),
        "spacing_pred_mm": _clean(sp_pred),
        "spacing_gt_mm": _clean(sp_gt),
        "ratio": _clean(ratio),
        "tolerance": SPACING_TOLERANCE,
        "requirement": "median centerline spacing of the prediction must be "
                       "within +/-20% of the reference",
    }


def evaluate(pred_pos, pred_edges, gt_pos, gt_edges, d_mm, densify_to=1.0):
    """All metrics at tolerance d_mm. Returns {name: {value, ...}}.

    Both graphs are first densified so that no edge is longer than
    ``densify_to * d_mm``, the matching tolerance itself by default.

    This inserts nodes along existing edges and moves nothing: every original
    node keeps its position, and the traced curve, the branch decomposition,
    every node degree and both Betti numbers are unchanged. What it does change
    is that a point on one curve is then guaranteed a node within d_mm on the
    other wherever the two agree, so a metric that pairs discrete elements
    reports the reconstruction rather than the discretisation.

    Measured on TopCoW MR with no error present and the prediction decimated
    against a common dense base, every tolerance-based metric returns exactly
    1.000 for prediction spacings from 0.09 to 3 times d_mm, and begins to fall
    only near 6 times d_mm. Finer targets (d_mm/2, d_mm/4) give identical
    results, so the tolerance itself is used. Metrics that require a one-to-one
    assignment remain bounded by the ratio of edge counts and are improved but
    not repaired by this step. Pass ``densify_to=None`` to compare the graphs
    exactly as given.
    """
    if densify_to:
        step = densify_to * d_mm
        pred_pos, pred_edges = densify(pred_pos, pred_edges, step)
        gt_pos, gt_edges = densify(gt_pos, gt_edges, step)
    out = {}
    for name, (fn, family, lower) in METRICS.items():
        try:
            res = fn(pred_pos, pred_edges, gt_pos, gt_edges, d_mm)
        except Exception as exc:
            res = {"value": float("nan"), "error": str(exc)[:200]}
        res = _clean(res)
        res["family"] = family
        res["lower_is_better"] = lower
        out[name] = res

    for name, res in morphology.morphology(
            pred_pos, pred_edges, gt_pos, gt_edges, d_mm).items():
        res = _clean(res)
        res["family"] = "descriptor"
        out[name] = res
    return out


def warnings_for(pred_pos, pred_edges, gt_pos, gt_edges, d_mm):
    sp_pred = spacing(pred_pos, pred_edges)
    out = []
    if np.isfinite(sp_pred) and sp_pred > d_mm:
        out.append(
            f"prediction spacing ({sp_pred:.3f} mm) exceeds the tolerance D "
            f"({d_mm} mm): reference nodes can fall between predicted nodes "
            f"with no correspondence, depressing node_F1 for reasons unrelated "
            f"to reconstruction quality.")
    return out


class SpacingMismatch(Exception):
    """The two graphs are not sampled closely enough to be comparable."""

    def __init__(self, check):
        self.check = check
        super().__init__(
            f"centerline spacing differs by {check['ratio']:.2f}x "
            f"({check['spacing_pred_mm']:.3f} mm vs "
            f"{check['spacing_gt_mm']:.3f} mm); the limit is "
            f"+/-{int(100 * check['tolerance'])}%.\n"
            f"Resample the prediction, the reference, or both to a common step "
            f"before comparing, or pass --force to evaluate anyway. Forced "
            f"results carry a failed spacing check and the matching metrics "
            f"are not interpretable.")


def build(pred_pos, pred_edges, gt_pos, gt_edges, d_mm,
          pred_name=None, gt_name=None, force=False):
    """The full JSON-serialisable report.

    Raises ``SpacingMismatch`` unless ``force`` when the two graphs are sampled
    too differently for the matching metrics to mean anything.
    """
    check = spacing_check(pred_pos, pred_edges, gt_pos, gt_edges)
    if not check["passed"] and not force:
        raise SpacingMismatch(check)

    sp_pred = spacing(pred_pos, pred_edges)
    sp_gt = spacing(gt_pos, gt_edges)
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
            "spacing_pred_mm": _clean(sp_pred),
            "spacing_gt_mm": _clean(sp_gt),
        },
        "tolerance": {
            "D_mm": float(d_mm),
            "definition": "dataset mean voxel diagonal",
            "caveat": "widely used in practice; no paper found that justifies "
                      "it. CAT08 ties its tolerance to the local vessel radius "
                      "instead, which is better justified but needs a radius "
                      "channel.",
        },
        "spacing_check": check,
        "metrics": evaluate(pred_pos, pred_edges, gt_pos, gt_edges, d_mm),
        "excluded": EXCLUDED,
        "warnings": (warnings_for(pred_pos, pred_edges, gt_pos, gt_edges, d_mm)
                     + ([] if check["passed"] else [
                         "SPACING CHECK FAILED and evaluation was forced. The "
                         "node- and edge-matching metrics measure the sampling "
                         "difference as much as the reconstruction and should "
                         "not be reported."])),
    }
