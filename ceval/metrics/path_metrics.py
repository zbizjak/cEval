"""Path-based metrics: TOPO (holes & marbles) and APLS.

Both operate on the graph's shortest-path structure rather than on isolated
points or edges. Ported here in mm space on cEval's node/edge contract
(points mm, edges undirected) using scipy.sparse.csgraph for Dijkstra —
no extra dependency beyond what's already in the Docker image.

Definitions and criticism: see metrics.md (Family 4 "TOPO / holes and
marbles" and Family 5 "APLS").
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree


def _adjacency(pos, edges):
    """Symmetric sparse adjacency (mm edge weights) for scipy's csgraph."""
    n = len(pos)
    if not len(edges):
        return csr_matrix((n, n))
    u, v = edges[:, 0].astype(np.int64), edges[:, 1].astype(np.int64)
    w = np.linalg.norm(pos[u] - pos[v], axis=1)
    keep = w > 0
    u, v, w = u[keep], v[keep], w[keep]
    return csr_matrix((np.concatenate([w, w]),
                       (np.concatenate([u, v]), np.concatenate([v, u]))),
                      shape=(n, n))


def _edge_orientation(pos, edges):
    """Unsigned edge direction in [0, pi), 3D projected onto its dominant plane.

    Orientation only needs to distinguish "same road" from "crossing road", so
    the direction vector itself (not an angle in a fixed 2D plane) is what
    gets compared: two edges "agree in orientation" when their unit directions
    are parallel within ``angle_tol``, which also covers non-planar (vessel)
    graphs that the original 2D road definition did not anticipate.
    """
    d = pos[edges[:, 1]] - pos[edges[:, 0]]
    return d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)


def _seed_points(pos, edges, step):
    """(coords, owning edge index) for points dropped every ``step`` mm.

    Vectorised over edges: every edge gets the same oversized parameter grid
    (sized to its own point count), and the unused tail is masked off, which
    is cheaper than a per-edge Python loop for the thousands of edges these
    metrics run against.
    """
    if not len(edges):
        return np.zeros((0, pos.shape[1])), np.zeros(0, dtype=np.int64)
    a, b = pos[edges[:, 0]], pos[edges[:, 1]]
    length = np.linalg.norm(b - a, axis=1)
    n = np.maximum(1, np.ceil(length / step).astype(np.int64))
    n_max = int(n.max())
    t = np.linspace(0.0, 1.0, n_max + 1)[None, :, None]          # (1, n_max+1, 1)
    coords = a[:, None, :] * (1 - t) + b[:, None, :] * t          # (E, n_max+1, 3)
    keep = np.arange(n_max + 1)[None, :] <= n[:, None]            # (E, n_max+1)
    owner = np.repeat(np.arange(len(edges)), n_max + 1)
    return coords.reshape(-1, pos.shape[1])[keep.ravel()], owner[keep.ravel()]


def _nearest_endpoint(pos, edge_nodes, point):
    """Which of an edge's two endpoints is closer to ``point``."""
    a, b = pos[edge_nodes[0]], pos[edge_nodes[1]]
    return int(edge_nodes[0] if np.linalg.norm(a - point) <= np.linalg.norm(b - point)
              else edge_nodes[1])


def topo(pred_pos, pred_edges, gt_pos, gt_edges, tol,
         seed_step=50.0, hole_step=5.0, match_dist=15.0, angle_tol_deg=45.0,
         radius=300.0):
    """TOPO: precision/recall from bottleneck-matched holes and marbles.

    Follows He et al. (RoadTracer, ACM MM'18): seeds every ``seed_step`` mm on
    the reference; a seed is valid if a corresponding point (similar location
    and orientation) exists on the prediction. From each valid seed, "holes"
    are dropped every ``hole_step`` mm on reference edges reachable within
    ``radius`` mm, "marbles" likewise on the prediction from the corresponding
    point, and the two clouds are matched one-to-one within ``match_dist`` mm
    and ``angle_tol_deg`` degrees. An invalid seed scores recall 0 and is
    excluded from precision, per the paper, so predictions with different
    numbers of valid seeds stay comparable.

    Criticism (Citraro et al., see metrics.md): sampling starts from the
    reference only, so predicted structure far from any reference edge is
    never covered by a control point and never penalised by this metric.
    """
    empty = {"value": 0.0, "precision": 0.0, "recall": 0.0, "n_seeds": 0,
             "n_valid_seeds": 0}
    if not len(pred_edges) or not len(gt_edges):
        return empty

    gt_adj = _adjacency(gt_pos, gt_edges)
    pred_adj = _adjacency(pred_pos, pred_edges)
    pred_orient = _edge_orientation(pred_pos, pred_edges)
    gt_orient = _edge_orientation(gt_pos, gt_edges)
    cos_tol = np.cos(np.radians(angle_tol_deg))

    seeds, seed_owner = _seed_points(gt_pos, gt_edges, seed_step)
    pred_cloud, pred_owner = _seed_points(pred_pos, pred_edges, seed_step)
    if not len(seeds) or not len(pred_cloud):
        return empty
    pred_tree = cKDTree(pred_cloud)

    # Hole/marble points are dropped once per edge up front (they don't depend
    # on which seed is being processed, only on which edges end up reachable
    # from it), so each seed below just selects rows instead of resampling.
    gt_holes, gt_holes_owner = _seed_points(gt_pos, gt_edges, hole_step)
    pred_marbles, pred_marbles_owner = _seed_points(pred_pos, pred_edges, hole_step)

    # For every seed, find its best corresponding point on the prediction
    # (closest within match_dist among candidates that agree in orientation).
    # Vectorised via query_ball_point's ragged output rather than a per-seed
    # KD-tree query loop.
    best = np.full(len(seeds), -1, dtype=np.int64)
    best_d = np.full(len(seeds), np.inf)
    neighbours = pred_tree.query_ball_point(seeds, r=match_dist)
    for si, cand in enumerate(neighbours):
        if not cand:
            continue
        cand = np.asarray(cand)
        cos = np.abs(pred_orient[pred_owner[cand]] @ gt_orient[seed_owner[si]])
        cand = cand[cos >= cos_tol]
        if not len(cand):
            continue
        d = np.linalg.norm(pred_cloud[cand] - seeds[si], axis=1)
        j = np.argmin(d)
        best[si], best_d[si] = cand[j], d[j]

    valid = best >= 0
    n_valid = int(valid.sum())
    recalls = np.zeros(len(seeds))
    precisions = []
    if not n_valid:
        return {"value": 0.0, "precision": 0.0, "recall": 0.0,
                "n_seeds": len(seeds), "n_valid_seeds": 0}

    # One Dijkstra call per graph, multi-source over every distinct anchor
    # node used by a valid seed, instead of one call per seed.
    valid_idx = np.flatnonzero(valid)
    gt_edge_of = seed_owner[valid_idx]
    pred_edge_of = pred_owner[best[valid_idx]]
    gt_anchor = np.array([_nearest_endpoint(gt_pos, gt_edges[e], seeds[si])
                          for e, si in zip(gt_edge_of, valid_idx)])
    pred_anchor = np.array([_nearest_endpoint(pred_pos, pred_edges[e], pred_cloud[m])
                            for e, m in zip(pred_edge_of, best[valid_idx])])

    gt_uniq, gt_inv = np.unique(gt_anchor, return_inverse=True)
    pred_uniq, pred_inv = np.unique(pred_anchor, return_inverse=True)
    gt_dist_all = dijkstra(gt_adj, directed=False, indices=gt_uniq, limit=radius)
    pred_dist_all = dijkstra(pred_adj, directed=False, indices=pred_uniq, limit=radius)

    for k, si in enumerate(valid_idx):
        gt_reach = np.isfinite(gt_dist_all[gt_inv[k]])
        pred_reach = np.isfinite(pred_dist_all[pred_inv[k]])

        gt_keep = gt_reach[gt_edges[:, 0]] & gt_reach[gt_edges[:, 1]]
        pred_keep = pred_reach[pred_edges[:, 0]] & pred_reach[pred_edges[:, 1]]
        holes = gt_holes[gt_keep[gt_holes_owner]]
        marbles = pred_marbles[pred_keep[pred_marbles_owner]]

        if not len(holes):
            continue  # recalls[si] stays 0
        if not len(marbles):
            precisions.append(0.0)
            continue

        # One-to-one bottleneck matching within match_dist. Both clouds are
        # bounded by radius/hole_step (a few hundred points at most, since
        # they only cover the 300 mm neighbourhood of one seed), so a dense
        # LSA over the padded square matrix stays cheap — the cost that
        # mattered was the per-seed Dijkstra call, now batched above.
        d_mh = np.linalg.norm(holes[:, None, :] - marbles[None, :, :], axis=2)
        big = match_dist * 10 + 1
        cost = np.where(d_mh <= match_dist, d_mh, big)
        n = max(cost.shape)
        pad = np.full((n, n), big)
        pad[:cost.shape[0], :cost.shape[1]] = cost
        ri, ci = linear_sum_assignment(pad)
        matched = int((pad[ri, ci] < big).sum())

        precisions.append(matched / len(marbles))
        recalls[si] = matched / len(holes)

    prec = float(np.mean(precisions)) if precisions else 0.0
    rec = float(np.mean(recalls))
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"value": f1, "precision": prec, "recall": rec,
            "n_seeds": len(seeds), "n_valid_seeds": n_valid}


def apls(pred_pos, pred_edges, gt_pos, gt_edges, tol,
         n_pairs=200, snap_dist=None, seed=0):
    """APLS: agreement of shortest-path lengths between corresponding nodes.

    ``APLS = mean_pairs 2 / (1/D_pred_given_gt + 1/D_gt_given_pred)`` with
    ``D = 1 - mean min(1, |L_pred - L_gt| / L_gt)`` over sampled node pairs,
    following Van Etten et al. (SpaceNet/CRESI, arXiv:1904.09901). A pair with
    no path in one direction, or whose corresponding node cannot be snapped
    within ``snap_dist`` (default: ``tol``), scores 0 for that pair.

    Ported to mm space on cEval's node/edge contract: a node is snapped to its
    nearest point in the other graph directly, rather than by injecting a
    proposal node onto the nearest edge as the original CRS-space
    implementation does — the two agree wherever ``snap_dist`` <= half the
    local edge length, and differ only in how finely a near-miss is credited.
    """
    empty = {"value": 0.0, "n_pairs": 0}
    if not len(pred_pos) or not len(gt_pos) or not len(pred_edges) or not len(gt_edges):
        return empty
    snap = tol if snap_dist is None else snap_dist

    gt_adj = _adjacency(gt_pos, gt_edges)
    pred_adj = _adjacency(pred_pos, pred_edges)
    pred_tree = cKDTree(pred_pos)
    gt_tree = cKDTree(gt_pos)

    def snap_all(pos_tree, points):
        d, j = pos_tree.query(points)
        j = j.astype(np.int64)
        j[d > snap] = -1
        return j

    def directed_scores(src_adj, src_pos, dst_adj, dst_tree):
        """Sample node pairs on the source graph, score by path-length agreement."""
        n = src_adj.shape[0]
        if n < 2:
            return []
        rng = np.random.default_rng(seed)
        pairs = rng.choice(n, size=(min(n_pairs, n * (n - 1) // 2), 2))
        pairs = pairs[pairs[:, 0] != pairs[:, 1]]
        if not len(pairs):
            return []
        snapped = snap_all(dst_tree, src_pos)  # src node -> nearest dst node, or -1

        # One multi-source Dijkstra call per graph (covering every distinct
        # source/snapped-source actually needed), instead of one call per
        # unique source in a loop.
        src_uniq, src_inv = np.unique(pairs[:, 0], return_inverse=True)
        src_dist = dijkstra(src_adj, directed=False, indices=src_uniq)
        l_src = src_dist[src_inv, pairs[:, 1]]

        dst_s = snapped[pairs[:, 0]]
        dst_uniq, dst_inv = np.unique(dst_s[dst_s >= 0], return_inverse=True)
        dst_dist = (dijkstra(dst_adj, directed=False, indices=dst_uniq)
                    if len(dst_uniq) else np.zeros((0, dst_adj.shape[0])))
        # dst_row[k] indexes into dst_dist for pair k, or -1 if unsnapped.
        dst_row = np.full(len(pairs), -1, dtype=np.int64)
        dst_row[dst_s >= 0] = dst_inv

        scores = []
        for k in range(len(pairs)):
            l = l_src[k]
            if not np.isfinite(l):
                continue  # both graphs must route the pair; unroutable source-side is skipped, not scored 0
            dst_t = snapped[pairs[k, 1]]
            if dst_row[k] < 0 or dst_t < 0:
                scores.append(0.0)
                continue
            l_dst = dst_dist[dst_row[k], dst_t]
            if not np.isfinite(l_dst):
                scores.append(0.0)
            elif l <= 0:
                scores.append(1.0)
            else:
                scores.append(1.0 - min(1.0, abs(l_dst - l) / l))
        return scores

    scores_gp = directed_scores(gt_adj, gt_pos, pred_adj, pred_tree)    # D(pred | gt)
    scores_pg = directed_scores(pred_adj, pred_pos, gt_adj, gt_tree)    # D(gt | pred)

    if not scores_gp and not scores_pg:
        return empty
    d_gp = float(np.mean(scores_gp)) if scores_gp else 0.0
    d_pg = float(np.mean(scores_pg)) if scores_pg else 0.0
    value = (2 / (1 / d_gp + 1 / d_pg)) if d_gp > 0 and d_pg > 0 else 0.0
    return {"value": value, "D_pred_given_gt": d_gp, "D_gt_given_pred": d_pg,
            "n_pairs": len(scores_gp) + len(scores_pg)}
