"""Connectivity between matched nodes: false merges, false splits, NetMets.

These metrics first put the nodes of the two graphs into correspondence and
then ask whether corresponding nodes are connected the same way. They differ
from the topology counts (beta0, beta1), which compare totals and never look
at which nodes are joined.

Definitions and criticism: see metrics.md.
"""
import heapq

import numpy as np
from scipy.spatial import cKDTree

from .base import adjacency, components, f1, greedy_match


def _component_label(n, edges):
    label = np.empty(n, dtype=np.int64)
    for k, comp in enumerate(components(n, edges)):
        label[comp] = k
    return label


def _within(adj, pos, src, dst, limit):
    """Is dst reachable from src along the graph within `limit` mm?"""
    if src == dst:
        return True
    best = {src: 0.0}
    heap = [(0.0, src)]
    while heap:
        d, u = heapq.heappop(heap)
        if u == dst:
            return True
        if d > best.get(u, np.inf):
            continue
        for v in adj[u]:
            nd = d + float(np.linalg.norm(pos[u] - pos[v]))
            if nd <= limit and nd < best.get(v, np.inf):
                best[v] = nd
                heapq.heappush(heap, (nd, v))
    return False


def _pred_connections(pred_pos, pred_edges, matched):
    """Connections between matched predicted nodes.

    A connection is either an edge between two matched nodes or a path whose
    interior runs only through unmatched nodes. Unmatched stretches arise
    wherever the prediction leaves the reference by more than tau (a bridge,
    a chord, a displaced junction), so contracting them is what lets a long
    spurious bridge count as one connection rather than as nothing.
    Returns [(u, v, length_mm, edge_list)].
    """
    adj = adjacency(len(pred_pos), pred_edges)
    out = []
    for u, v in pred_edges:
        u, v = int(u), int(v)
        if u in matched and v in matched:
            out.append((u, v, float(np.linalg.norm(pred_pos[u] - pred_pos[v])),
                        [(u, v)]))
    seen = set()
    for start in range(len(pred_pos)):
        if start in matched or start in seen or not adj[start]:
            continue
        # one unmatched region and the matched nodes on its boundary
        region, stack, boundary = {start}, [start], set()
        while stack:
            x = stack.pop()
            for y in adj[x]:
                if y in matched:
                    boundary.add(y)
                elif y not in region:
                    region.add(y)
                    stack.append(y)
        seen |= region
        boundary = sorted(boundary)
        if len(boundary) < 2:
            continue
        allowed = region | set(boundary)
        for i, a in enumerate(boundary):
            # shortest path from a through the region to every other boundary node
            dist, prev = {a: 0.0}, {}
            heap = [(0.0, a)]
            while heap:
                d, x = heapq.heappop(heap)
                if d > dist[x] or (x != a and x in matched):
                    continue
                for y in adj[x]:
                    if y not in allowed or (x == a and y in matched):
                        continue
                    nd = d + float(np.linalg.norm(pred_pos[x] - pred_pos[y]))
                    if nd < dist.get(y, np.inf):
                        dist[y], prev[y] = nd, x
                        heapq.heappush(heap, (nd, y))
            for b in boundary[i + 1:]:
                if b not in dist:
                    continue
                path, x = [], b
                while x != a:
                    path.append((prev[x], x))
                    x = prev[x]
                out.append((a, b, dist[b], path))
    return out


def _false_merges(pred_pos, pred_edges, gt_pos, gt_edges, tol):
    m = greedy_match(gt_pos, pred_pos, tol)            # gt -> pred
    inv = {p: g for g, p in m.items()}                  # pred -> gt
    gt_label = _component_label(len(gt_pos), gt_edges)
    gt_adj = adjacency(len(gt_pos), gt_edges)
    fm, inter, intra = [], 0, 0
    for u, v, length, path in _pred_connections(pred_pos, pred_edges, inv):
        x, y = inv[u], inv[v]
        if gt_label[x] != gt_label[y]:
            inter += 1
            fm.append(path)
        elif not _within(gt_adj, gt_pos, x, y, length + 2 * tol):
            intra += 1
            fm.append(path)
    return fm, inter, intra


def false_merges(pred_pos, pred_edges, gt_pos, gt_edges, tol):
    """Vesselpose false merges: predicted connections the reference does not have.

    Palaniappan et al. (2026) count a predicted edge as a false merge when its
    two matched reference nodes have no path between them in the reference,
    which they take on a directed (rooted) tree, so that joining two sibling
    branches also counts. Without a root, the same two cases are detected here
    as: the matched reference nodes lie in different components (inter), or
    they lie in one component but no reference path between them is shorter
    than the predicted connection plus 2 tau (intra; a shortcut or chord).
    Nodes are matched one-to-one within tau; connections through unmatched
    predicted nodes are contracted (see _pred_connections).
    """
    if not len(pred_edges) or not len(gt_edges):
        return {"value": 0.0}
    fm, inter, intra = _false_merges(pred_pos, pred_edges, gt_pos, gt_edges, tol)
    return {"value": float(len(fm)), "inter_component": inter,
            "intra_component": intra,
            "relative": len(fm) / max(1, len(pred_edges))}


def false_splits(pred_pos, pred_edges, gt_pos, gt_edges, tol):
    """Vesselpose false splits: component-count difference after removing merges.

    |b0(reference) - b0(prediction without its false-merge edges)|. Removing
    the merges first stops a merge and a split from cancelling, which is the
    one way this differs from beta0_err.
    """
    if not len(pred_edges) or not len(gt_edges):
        return {"value": float("nan")}
    fm, _, _ = _false_merges(pred_pos, pred_edges, gt_pos, gt_edges, tol)
    # Vesselpose removes the merging edge itself. A contracted connection is a
    # chain of edges; cutting its first edge separates what it joined, while
    # cutting all of them would leave its interior nodes as spurious
    # one-node components.
    drop = {(min(path[0]), max(path[0])) for path in fm}
    kept = np.array([e for e in pred_edges.tolist()
                     if (min(e), max(e)) not in drop], dtype=np.int64).reshape(-1, 2)
    # components over the nodes each graph uses, ignoring isolated points
    n_pred = len(components(len(pred_pos), kept)) - \
        (len(pred_pos) - len(np.unique(pred_edges)))
    n_gt = len(components(len(gt_pos), gt_edges)) - \
        (len(gt_pos) - len(np.unique(gt_edges)))
    return {"value": float(abs(n_gt - n_pred)), "n_components_gt": n_gt,
            "n_components_pred_after_fm": n_pred}


def _core_edges(pos, edges, colour):
    """NetMets core connectivity graph over the coloured key nodes.

    Key nodes are junctions and endpoints (degree != 2). Two coloured key nodes
    are joined when a path between them passes only through uncoloured key
    nodes. Returns a set of frozensets of colours.
    """
    adj = adjacency(len(pos), edges)
    key = {i for i in range(len(pos)) if adj[i] and len(adj[i]) != 2}
    # key-node adjacency: walk each branch to the next key node
    kadj = {k: set() for k in key}
    for k in key:
        for nb in adj[k]:
            prev, cur = k, nb
            while cur not in key:
                nxt = adj[cur] - {prev}
                if not nxt:
                    break
                prev, cur = cur, next(iter(nxt))
            if cur in key and cur != k:
                kadj[k].add(cur)
                kadj[cur].add(k)
    out = set()
    for k in key:
        if k not in colour:
            continue
        stack, seen = [k], {k}
        while stack:
            x = stack.pop()
            for y in kadj[x]:
                if y in seen:
                    continue
                seen.add(y)
                if y in colour:
                    if colour[y] != colour[k]:
                        out.add(frozenset((colour[k], colour[y])))
                else:
                    stack.append(y)
    return out


def netmets_connectivity(pred_pos, pred_edges, gt_pos, gt_edges, tol):
    """NetMets connectivity score (CFNR / CFPR), reported as F1.

    Mayerich et al. (2012): junctions and endpoints of the two networks are
    matched within sigma ("coloured"), each network is reduced to a core graph
    joining coloured nodes through uncoloured ones, and the core edges are
    compared: CFNR = FN / (FN + TP), CFPR = FP / (FP + TP). Sigma is tau here
    and the matching is one-to-one. The value is the F1 of 1 - CFPR and
    1 - CFNR.
    """
    if not len(pred_edges) or not len(gt_edges):
        return {"value": 0.0}
    deg_p = np.bincount(pred_edges.ravel(), minlength=len(pred_pos))
    deg_g = np.bincount(gt_edges.ravel(), minlength=len(gt_pos))
    kp = np.flatnonzero((deg_p > 0) & (deg_p != 2))
    kg = np.flatnonzero((deg_g > 0) & (deg_g != 2))
    m = greedy_match(gt_pos[kg], pred_pos[kp], tol)
    colour_g = {int(kg[i]): int(kg[i]) for i in m}
    colour_p = {int(kp[j]): int(kg[i]) for i, j in m.items()}
    eg = _core_edges(gt_pos, gt_edges, colour_g)
    ep = _core_edges(pred_pos, pred_edges, colour_p)
    tp, fn, fp = len(eg & ep), len(eg - ep), len(ep - eg)
    if tp + fn + fp == 0:
        # no core edges on either side (e.g. no matched key nodes)
        return {"value": 1.0 if not len(kg) and not len(kp) else 0.0,
                "TP": 0, "FN": 0, "FP": 0}
    cfnr = fn / (fn + tp) if fn + tp else 0.0
    cfpr = fp / (fp + tp) if fp + tp else 0.0
    return {"value": f1(1 - cfpr, 1 - cfnr), "CFNR": cfnr, "CFPR": cfpr,
            "TP": tp, "FN": fn, "FP": fp}
