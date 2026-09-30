"""Branch continuity: is a vessel traced as one branch, or as several?

Every metric in this study asks whether the predicted structure lies where the
reference does. None asks whether it is *segmented into the same branches*.
A prediction can place every node and edge correctly and still split one
20 mm vessel into four pieces by inventing two 1 mm stubs along it: the nodes
are right, the edges are right, the Betti numbers are right, and the 80%
branch-coverage rule is satisfied because the reference branch is fully covered
— by three predicted branches instead of one.

That matters. A branch is the unit in which vessel trees are read: branch
length, bifurcation count and the generation a segment belongs to are all
defined on it. A tree fragmented into twice as many branches gives the wrong
answer to every one of those questions while scoring perfectly on overlap.

The metric here asks, for each reference branch, what fraction of its length is
covered by the single best-matching predicted branch — not by any combination
of them. Spurious mid-branch bifurcations cap that fraction at the largest
piece they leave behind.

Definitions and criticism: see metrics.md.
"""
import numpy as np
from scipy.spatial import cKDTree

from .base import branch_paths, f1, resample_path


def _covered_runs(pts, tree, tol):
    """Lengths of the maximal runs of consecutive points within tol."""
    hit = tree.query(pts)[0] <= tol
    step = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    runs, cur = [], 0.0
    for k, d in enumerate(step):
        if hit[k] and hit[k + 1]:
            cur += float(d)
        elif cur > 0:
            runs.append(cur)
            cur = 0.0
    if cur > 0:
        runs.append(cur)
    return runs


def branch_continuity(pred_pos, pred_edges, gt_pos, gt_edges, tol):
    """Fraction of each reference branch covered by one predicted branch.

    Returns the length-weighted mean over reference branches, so a fragmented
    long vessel counts for more than a fragmented twig.

    ``recall`` is that quantity; ``precision`` is its mirror, penalising a
    prediction that merges several reference branches into one. ``value`` is
    their harmonic mean.
    """
    if not len(pred_edges) or not len(gt_edges):
        return {"value": 0.0}
    step = tol / 2

    pred_br = [resample_path(pred_pos, b, step)
               for b in branch_paths(pred_pos, pred_edges)]
    gt_br = [resample_path(gt_pos, b, step)
             for b in branch_paths(gt_pos, gt_edges)]
    if not pred_br or not gt_br:
        return {"value": 0.0}

    pred_trees = [cKDTree(b) for b in pred_br]
    gt_trees = [cKDTree(b) for b in gt_br]

    def directed(sources, trees):
        fracs, weights = [], []
        for pts in sources:
            total = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
            if total < 1e-9:
                continue
            # The best single counterpart, not the union of all of them: the
            # union is what hides fragmentation.
            best = 0.0
            for tree in trees:
                runs = _covered_runs(pts, tree, tol)
                if runs:
                    best = max(best, max(runs))
            fracs.append(min(1.0, best / total))
            weights.append(total)
        if not fracs:
            return 0.0, 0.0
        fracs, weights = np.asarray(fracs), np.asarray(weights)
        mean = float(np.average(fracs, weights=weights))
        # Fragmentation is usually local: one main vessel split into pieces
        # among fifty intact twigs. A length-weighted mean dilutes that to
        # nothing, so the worst branch is reported alongside it — clinically it
        # is the shattered vessel that matters, not the average.
        worst = float(fracs[np.argsort(fracs)[:max(1, len(fracs) // 10)]].mean())
        return mean, worst

    rec, rec_worst = directed(gt_br, pred_trees)
    prec, prec_worst = directed(pred_br, gt_trees)
    return {"value": f1(prec, rec), "precision": prec, "recall": rec,
            "worst_decile": min(rec_worst, prec_worst),
            "n_pred_branches": len(pred_br), "n_gt_branches": len(gt_br)}


def branch_count_ratio(pred_pos, pred_edges, gt_pos, gt_edges, tol):
    """|branches(pred) / branches(gt) - 1|, a blunt companion measure.

    Fragmentation raises the branch count and merging lowers it. Unlike
    ``branch_continuity`` this says nothing about *which* branches are affected,
    but it is trivial to compute and is what a reader would check first.
    """
    n_pred = len(branch_paths(pred_pos, pred_edges))
    n_gt = len(branch_paths(gt_pos, gt_edges))
    if n_gt == 0:
        return {"value": float("inf"), "n_pred": n_pred, "n_gt": n_gt}
    return {"value": abs(n_pred / n_gt - 1.0),
            "n_pred": n_pred, "n_gt": n_gt}
