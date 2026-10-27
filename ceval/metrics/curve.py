"""Curve metrics: everything is computed on the centerline curves themselves.

A centerline graph is a polyline in 3D: its edges are straight segments. The
part of one segment that lies within tau of another segment is the part
inside the capsule around the other (a cylinder of radius tau capped by two
balls). A capsule is convex, so that part is a single interval whose ends are
roots of quadratics; the union of these intervals over every nearby segment of
the other graph gives the exact length of a segment within tau of that graph,
and summing over segments gives it for a whole graph. Distances along a curve
(for ASCD, HD95 and NetMets geometry) are integrated adaptively to a
resolution of tau/10, and path metrics (APLS) start and end on exact points
of the curve. Nothing depends on where either graph places its nodes.

Notation used below: |G| is the length of a graph, C_p the length of the
prediction within tau of the reference and C_g the length of the reference
within tau of the prediction.
"""
import hashlib

import numpy as np
from scipy.spatial import cKDTree

from .base import branch_paths, f1

_EPS = 1e-12


def _quad_le0(a, b, c):
    """Per row, {t : a t^2 + b t + c <= 0} with a >= 0, as (lo, hi); NaN if empty.

    Rows with a ~ 0 are linear or constant in t; for the constant case the set
    is everything or nothing.
    """
    lo = np.full(a.shape, np.nan)
    hi = np.full(a.shape, np.nan)
    quad = a > _EPS
    disc = b * b - 4 * a * c
    ok = quad & (disc >= 0)
    s = np.sqrt(np.where(ok, disc, 0.0))
    lo[ok] = (-b[ok] - s[ok]) / (2 * a[ok])
    hi[ok] = (-b[ok] + s[ok]) / (2 * a[ok])
    lin = ~quad & (np.abs(b) > _EPS)
    r = np.where(lin, -c / np.where(lin, b, 1.0), 0.0)
    up = lin & (b > 0)                      # b t + c <= 0  ->  t <= r
    dn = lin & (b < 0)                      #               ->  t >= r
    lo[up], hi[up] = -np.inf, r[up]
    lo[dn], hi[dn] = r[dn], np.inf
    const = ~quad & ~lin & (c <= 0)
    lo[const], hi[const] = -np.inf, np.inf
    return lo, hi


def capsule_intervals(a0, a1, b0, b1, tau):
    """Exact [lo, hi] (in t, a0 + t (a1 - a0), t in [0, 1]) of each segment a0-a1
    inside the capsule of radius tau around the paired segment b0-b1.

    All inputs are (n, 3) arrays of paired segments. Returns two (n,) arrays,
    NaN where the segment does not enter the capsule.
    """
    v = a1 - a0
    vv = np.einsum("ij,ij->i", v, v)
    parts_lo, parts_hi = [], []
    for c in (b0, b1):                                  # the two balls
        w = a0 - c
        lo, hi = _quad_le0(vv, 2 * np.einsum("ij,ij->i", w, v),
                           np.einsum("ij,ij->i", w, w) - tau * tau)
        parts_lo.append(lo); parts_hi.append(hi)
    d = b1 - b0
    L = np.linalg.norm(d, axis=1)
    has_axis = L > _EPS
    e = np.where(has_axis[:, None], d / np.where(has_axis, L, 1.0)[:, None], 0.0)
    w = a0 - b0
    ve = np.einsum("ij,ij->i", v, e)
    we = np.einsum("ij,ij->i", w, e)
    vp = v - ve[:, None] * e                            # components across the axis
    wp = w - we[:, None] * e
    clo, chi = _quad_le0(np.einsum("ij,ij->i", vp, vp),
                         2 * np.einsum("ij,ij->i", wp, vp),
                         np.einsum("ij,ij->i", wp, wp) - tau * tau)
    # axial coordinate we + t ve must lie in [0, L]
    moving = np.abs(ve) > _EPS
    t_a = np.where(moving, (0.0 - we) / np.where(moving, ve, 1.0), -np.inf)
    t_b = np.where(moving, (L - we) / np.where(moving, ve, 1.0), np.inf)
    alo, ahi = np.minimum(t_a, t_b), np.maximum(t_a, t_b)
    inside_slab = (we >= 0) & (we <= L)
    alo = np.where(moving, alo, np.where(inside_slab, -np.inf, np.nan))
    ahi = np.where(moving, ahi, np.where(inside_slab, np.inf, np.nan))
    cyl_lo, cyl_hi = np.maximum(clo, alo), np.minimum(chi, ahi)
    bad = ~has_axis | ~(cyl_lo <= cyl_hi)
    cyl_lo[bad], cyl_hi[bad] = np.nan, np.nan
    parts_lo.append(cyl_lo); parts_hi.append(cyl_hi)
    # A capsule is convex, so its intersection with the line is one interval:
    # the union of the three part intervals, i.e. their min and max.
    # fmin/fmax skip NaN (a part the line misses) without a warning
    lo = np.fmin.reduce(np.vstack(parts_lo), axis=0)
    hi = np.fmax.reduce(np.vstack(parts_hi), axis=0)
    lo, hi = np.maximum(lo, 0.0), np.minimum(hi, 1.0)
    empty = ~(lo < hi)
    lo[empty], hi[empty] = np.nan, np.nan
    return lo, hi


def pair_intervals(a_pos, a_edges, b_pos, b_edges, tau):
    """Covered interval of each A edge against each nearby B edge.

    Returns (a_idx, b_idx, lo, hi), one row per (A edge, B edge) pair whose
    capsule the A edge enters, with lo < hi in [0, 1].
    """
    empty = (np.zeros(0, int), np.zeros(0, int), np.zeros(0), np.zeros(0))
    if not len(a_edges) or not len(b_edges):
        return empty
    a0, a1 = a_pos[a_edges[:, 0]], a_pos[a_edges[:, 1]]
    b0, b1 = b_pos[b_edges[:, 0]], b_pos[b_edges[:, 1]]
    am, ah = 0.5 * (a0 + a1), 0.5 * np.linalg.norm(a1 - a0, axis=1)
    bm, bh = 0.5 * (b0 + b1), 0.5 * np.linalg.norm(b1 - b0, axis=1)
    # Two segments can come within tau only if their midpoints lie within
    # half-length + half-length + tau of each other.
    near = cKDTree(bm).query_ball_point(am, ah + bh.max() + tau)
    ai = np.repeat(np.arange(len(a_edges)), [len(n) for n in near])
    bi = np.fromiter((j for n in near for j in n), dtype=np.int64, count=len(ai))
    if not len(ai):
        return empty
    keep = np.linalg.norm(am[ai] - bm[bi], axis=1) <= ah[ai] + bh[bi] + tau
    ai, bi = ai[keep], bi[keep]
    lo, hi = capsule_intervals(a0[ai], a1[ai], b0[bi], b1[bi], tau)
    ok = ~np.isnan(lo)
    return ai[ok], bi[ok], lo[ok], hi[ok]


def _merge(group, lo, hi):
    """Merge intervals within each group. Inputs need not be sorted.

    Returns (group, lo, hi) of disjoint runs, sorted by group then lo.
    """
    if not len(group):
        return group, lo, hi
    order = np.lexsort((lo, group))
    g, lo, hi = group[order], lo[order], hi[order]
    out_g, out_lo, out_hi = [g[0]], [lo[0]], [hi[0]]
    for k in range(1, len(g)):
        if g[k] == out_g[-1] and lo[k] <= out_hi[-1] + 1e-9:
            if hi[k] > out_hi[-1]:
                out_hi[-1] = hi[k]
        else:
            out_g.append(g[k]); out_lo.append(lo[k]); out_hi.append(hi[k])
    return np.asarray(out_g), np.asarray(out_lo), np.asarray(out_hi)


_CACHE = {}


def _key(*arrays):
    h = hashlib.blake2b(digest_size=16)
    for x in arrays:
        x = np.ascontiguousarray(x)
        h.update(str(x.shape).encode()); h.update(x.tobytes())
    return h.digest()


def covered_runs(a_pos, a_edges, b_pos, b_edges, tau):
    """Disjoint covered runs (edge, lo, hi) of graph A within tau of graph B.

    Memoised on the array contents, because several metrics below ask for the
    same two directions of the same pair of graphs.
    """
    k = _key(a_pos, a_edges, b_pos, b_edges, np.array([tau]))
    if k not in _CACHE:
        if len(_CACHE) > 8:
            _CACHE.clear()
        ai, _, lo, hi = pair_intervals(a_pos, a_edges, b_pos, b_edges, tau)
        _CACHE[k] = _merge(ai, lo, hi)
    return _CACHE[k]


def _edge_lengths(pos, edges):
    return np.linalg.norm(pos[edges[:, 1]] - pos[edges[:, 0]], axis=1)


def covered_length_per_edge(a_pos, a_edges, b_pos, b_edges, tau):
    """Length of each A edge within tau of graph B, in mm."""
    g, lo, hi = covered_runs(a_pos, a_edges, b_pos, b_edges, tau)
    frac = np.bincount(g, weights=hi - lo, minlength=len(a_edges)) if len(g) \
        else np.zeros(len(a_edges))
    return frac * _edge_lengths(a_pos, a_edges)


def closest_on_curve(points, b_pos, b_edges, radius=None):
    """Exact closest point of graph B's curve to each point.

    Returns (dist, edge, t): distance in mm, index of the B edge carrying the
    closest point and its parameter t in [0, 1]. With ``radius``, points
    farther than that get dist = inf and edge = -1. Without it the search is
    exhaustive: the nearest midpoint bounds the answer, and every segment that
    could beat it has its midpoint within that bound plus the longest half
    length.
    """
    n = len(points)
    dist, edge, tt = np.full(n, np.inf), np.full(n, -1, np.int64), np.zeros(n)
    if not n or not len(b_edges):
        return dist, edge, tt
    b0, b1 = b_pos[b_edges[:, 0]], b_pos[b_edges[:, 1]]
    bm, bh = 0.5 * (b0 + b1), 0.5 * np.linalg.norm(b1 - b0, axis=1)
    tree = cKDTree(bm)
    if radius is None:
        dm = tree.query(points)[0]
        r = dm + 2 * bh.max() + 1e-9
    else:
        r = np.full(n, radius + bh.max())
    # Candidates are the segments whose midpoint lies within r. The K nearest
    # midpoints hold all of them unless the K-th is itself within r; only
    # those points get a ball query (same candidates, without a Python loop
    # over every pair for the rest).
    K = min(32, len(bm))
    dk_, ik_ = tree.query(points, k=K)
    dk_, ik_ = dk_.reshape(n, K), ik_.reshape(n, K)
    full = dk_[:, -1] <= r if K < len(bm) else np.zeros(n, bool)
    inside = (dk_ <= r[:, None]) & ~full[:, None]
    pi, bi = np.nonzero(inside)
    bi = ik_[pi, bi]
    if full.any():
        idx = np.flatnonzero(full)
        near = tree.query_ball_point(points[idx], r[idx])
        pi = np.r_[pi, np.repeat(idx, [len(x) for x in near])]
        bi = np.r_[bi, np.fromiter((j for x in near for j in x), dtype=np.int64,
                                   count=sum(len(x) for x in near))]
    if not len(pi):
        return dist, edge, tt
    d = b1[bi] - b0[bi]
    dd = np.einsum("ij,ij->i", d, d)
    t = np.clip(np.einsum("ij,ij->i", points[pi] - b0[bi], d)
                / np.where(dd > _EPS, dd, 1.0), 0.0, 1.0)
    dk = np.linalg.norm(points[pi] - (b0[bi] + t[:, None] * d), axis=1)
    order = np.lexsort((dk, pi))                  # per point, closest first
    first = np.r_[True, pi[order][1:] != pi[order][:-1]]
    sel = order[first]
    p = pi[sel]
    dist[p], edge[p], tt[p] = dk[sel], bi[sel], t[sel]
    if radius is not None:
        far = dist > radius
        dist[far], edge[far] = np.inf, -1
    return dist, edge, tt


def _profile(a_pos, a_edges, b_pos, b_edges, delta, rel_tol=1e-6, max_iter=40):
    """Distance from graph A's curve to graph B's curve, resolved adaptively.

    Each A edge is split into pieces until, on every piece, the distance to B
    varies by less than ``delta`` (mm) and Simpson's rule and the two-half
    Simpson agree to ``rel_tol`` of the piece length. Returns, per accepted
    piece, its length (mm) and the distance at its two ends and its midpoint,
    from which the integral (Simpson), the length-weighted distribution and
    any function of the distance follow. Nothing depends on where A or B
    place their nodes: pieces are cut by the distance function, not by nodes.
    """
    if not len(a_edges) or not len(b_edges):
        return np.zeros(0), np.zeros((0, 3))
    a0, a1 = a_pos[a_edges[:, 0]], a_pos[a_edges[:, 1]]
    L = np.linalg.norm(a1 - a0, axis=1)
    keep = L > _EPS
    e = np.flatnonzero(keep)
    lo, hi = np.zeros(len(e)), np.ones(len(e))
    out_len, out_f = [], []

    def f_at(edge_idx, t):
        pts = a0[edge_idx] + t[:, None] * (a1[edge_idx] - a0[edge_idx])
        return closest_on_curve(pts, b_pos, b_edges)[0]

    fa, fb = f_at(e, lo), f_at(e, hi)
    fm = f_at(e, 0.5 * (lo + hi))
    for _ in range(max_iter):
        if not len(e):
            break
        mid = 0.5 * (lo + hi)
        q1, q3 = 0.5 * (lo + mid), 0.5 * (mid + hi)
        f1_, f3 = f_at(e, q1), f_at(e, q3)    # the midpoint is known from the parent
        w = (hi - lo) * L[e]
        s_whole = w / 6 * (fa + 4 * fm + fb)
        s_halves = w / 12 * (fa + 4 * f1_ + 2 * fm + 4 * f3 + fb)
        vals = np.vstack([fa, f1_, fm, f3, fb])
        spread = vals.max(0) - vals.min(0)
        ok = (np.abs(s_halves - s_whole) <= rel_tol * np.maximum(w, 1e-12)) & \
             (spread < delta)
        # accept as two halves, keeping the finer Simpson evaluation
        if ok.any():
            out_len += [w[ok] / 2, w[ok] / 2]
            out_f += [np.c_[fa[ok], f1_[ok], fm[ok]], np.c_[fm[ok], f3[ok], fb[ok]]]
        r = ~ok
        e = np.r_[e[r], e[r]]
        lo, hi = np.r_[lo[r], mid[r]], np.r_[mid[r], hi[r]]
        fa, fb, fm = np.r_[fa[r], fm[r]], np.r_[fm[r], fb[r]], np.r_[f1_[r], f3[r]]
    if len(e):                                     # depth cap reached: accept as is
        out_len.append((hi - lo) * L[e])
        out_f.append(np.c_[fa, fm, fb])
    return np.concatenate(out_len), np.concatenate(out_f)


_PROFILES = {}


# Resolution of the distance profile. tau/100 changed ASCD, HD95 and NetMets
# geometry by at most 7e-4 against tau/10 on ImageCAS and took 8 times longer.
PROFILE_DIVISOR = 10
PROFILE_REL_TOL = 1e-2


def distance_profile(a_pos, a_edges, b_pos, b_edges, tau):
    """Memoised _profile, resolved to delta = tau / PROFILE_DIVISOR."""
    k = _key(a_pos, a_edges, b_pos, b_edges, np.array([tau]))
    if k not in _PROFILES:
        if len(_PROFILES) > 8:
            _PROFILES.clear()
        _PROFILES[k] = _profile(a_pos, a_edges, b_pos, b_edges, tau / PROFILE_DIVISOR,
                                rel_tol=PROFILE_REL_TOL)
    return _PROFILES[k]


def _mean_of(fn, lens, f):
    """Length-weighted mean of fn(distance) along a profile (Simpson)."""
    tot = lens.sum()
    if tot <= 0:
        return float("nan")
    g = fn(f)
    return float((lens / 6 * (g[:, 0] + 4 * g[:, 1] + g[:, 2])).sum() / tot)


def _branch_edges(pos, edges):
    """For each branch: list of (edge index, forward?) along its node path."""
    index = {(min(int(u), int(v)), max(int(u), int(v))): k
             for k, (u, v) in enumerate(edges)}
    out = []
    for path in branch_paths(pos, edges):
        seq = []
        for u, v in zip(path[:-1], path[1:]):
            k = index.get((min(u, v), max(u, v)))
            if k is not None:
                seq.append((k, int(edges[k, 0]) == u))
        if seq:
            out.append(seq)
    return out


def _branch_runs(seq, lens, runs_by_edge):
    """Covered runs of one branch in arc length from its start, merged."""
    out, s0 = [], 0.0
    for k, fwd in seq:
        L = float(lens[k])
        for lo, hi in runs_by_edge.get(k, ()):
            a, b = (lo, hi) if fwd else (1.0 - hi, 1.0 - lo)
            out.append((s0 + a * L, s0 + b * L))
        s0 += L
    out.sort()
    merged = []
    for a, b in out:
        if merged and a <= merged[-1][1] + 1e-9:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return merged, s0


def _by_edge(g, lo, hi):
    d = {}
    for k, a, b in zip(g.tolist(), lo.tolist(), hi.tolist()):
        d.setdefault(k, []).append((a, b))
    return d


def _csr(pos, edges):
    from scipy.sparse import csr_matrix
    n = len(pos)
    w = np.linalg.norm(pos[edges[:, 0]] - pos[edges[:, 1]], axis=1)
    w = np.maximum(w, 1e-12)                       # keep zero-length edges as edges
    u, v = edges[:, 0], edges[:, 1]
    return csr_matrix((np.r_[w, w], (np.r_[u, v], np.r_[v, u])), shape=(n, n))


def _path_lengths(pos, edges, A, B):
    """Shortest path length along the graph between points A[k] and B[k].

    A and B are (edge, t) arrays (two columns). A point on edge (u, v) at t
    reaches u after t L and v after (1 - t) L.
    """
    from scipy.sparse.csgraph import dijkstra
    L = np.linalg.norm(pos[edges[:, 1]] - pos[edges[:, 0]], axis=1)
    ea, ta = A[:, 0].astype(int), A[:, 1]
    eb, tb = B[:, 0].astype(int), B[:, 1]
    src = np.unique(np.r_[edges[ea, 0], edges[ea, 1]])
    D = dijkstra(_csr(pos, edges), directed=False, indices=src)
    row = {int(s): k for k, s in enumerate(src)}
    out = np.full(len(A), np.inf)
    for k in range(len(A)):
        if ea[k] == eb[k]:
            out[k] = abs(ta[k] - tb[k]) * L[ea[k]]
        best = out[k]
        for a, oa in ((edges[ea[k], 0], ta[k] * L[ea[k]]),
                      (edges[ea[k], 1], (1 - ta[k]) * L[ea[k]])):
            r = D[row[int(a)]]
            for b, ob in ((edges[eb[k], 0], tb[k] * L[eb[k]]),
                          (edges[eb[k], 1], (1 - tb[k]) * L[eb[k]])):
                best = min(best, oa + r[b] + ob)
        out[k] = best
    return out


def _arc_control_points(pos, edges, spacing):
    """Junctions, endpoints and points every `spacing` mm along each branch,
    as (edge, t) rows. Placed by arc length, so independent of node spacing.

    Each branch is walked from its end with the smaller coordinates (x, then
    y, then z), and the points are returned sorted by position, so neither the
    points nor their order depend on how the graph numbers its nodes."""
    L = np.linalg.norm(pos[edges[:, 1]] - pos[edges[:, 0]], axis=1)
    pts = []
    for seq in _branch_edges(pos, edges):
        start = pos[edges[seq[0][0], 0 if seq[0][1] else 1]]
        end = pos[edges[seq[-1][0], 1 if seq[-1][1] else 0]]
        if tuple(np.round(end, 6)) < tuple(np.round(start, 6)):
            seq = [(k, not fwd) for k, fwd in reversed(seq)]
        k0, f0 = seq[0]
        pts.append((k0, 0.0 if f0 else 1.0))            # branch start
        s_next, s0 = spacing, 0.0
        for k, fwd in seq:
            while s_next < s0 + L[k] - 1e-9:
                t = (s_next - s0) / L[k]
                pts.append((k, t if fwd else 1.0 - t))
                s_next += spacing
            s0 += L[k]
        k1, f1_ = seq[-1]
        pts.append((k1, 1.0 if f1_ else 0.0))          # branch end
    pts = np.array(pts, float).reshape(-1, 2)
    if len(pts):
        xyz = np.round(_point_xyz(pos, edges, pts), 6)
        pts = pts[np.lexsort((xyz[:, 2], xyz[:, 1], xyz[:, 0]))]
    return pts


def _point_xyz(pos, edges, P):
    e, t = P[:, 0].astype(int), P[:, 1]
    return pos[edges[e, 0]] + t[:, None] * (pos[edges[e, 1]] - pos[edges[e, 0]])


def _two_way(pred_pos, pred_edges, gt_pos, gt_edges, tol):
    cp = covered_length_per_edge(pred_pos, pred_edges, gt_pos, gt_edges, tol)
    cg = covered_length_per_edge(gt_pos, gt_edges, pred_pos, pred_edges, tol)
    return (float(cp.sum()), float(_edge_lengths(pred_pos, pred_edges).sum()),
            float(cg.sum()), float(_edge_lengths(gt_pos, gt_edges).sum()))


def edge_f1(pred_pos, pred_edges, gt_pos, gt_edges, tol):
    """Edge F1: how much of each curve lies within tol of the other.

    Precision P = C_p / |G_p|, recall R = C_g / |G_g|, value = 2PR / (P + R).
    The same quantity is published as correctness (P) and completeness (R)
    by Wiedemann et al. (1998).
    """
    if not len(pred_edges) or not len(gt_edges):
        return {"value": 0.0}
    cp, lp, cg, lg = _two_way(pred_pos, pred_edges, gt_pos, gt_edges, tol)
    prec = cp / lp if lp else 0.0
    rec = cg / lg if lg else 0.0
    return {"value": f1(prec, rec), "precision": prec, "recall": rec,
            "covered_pred_mm": cp, "pred_length_mm": lp,
            "covered_gt_mm": cg, "gt_length_mm": lg}


def edge_r(pred_pos, pred_edges, gt_pos, gt_edges, tol):
    """Edge R: C_g / |G_g|, the fraction of reference length within tol of the
    prediction (the recall of Edge F1; completeness in Wiedemann et al. 1998,
    tree length detected in EXACT'09 and ATM'22)."""
    if not len(pred_edges) or not len(gt_edges):
        return {"value": 0.0}
    cg = covered_length_per_edge(gt_pos, gt_edges, pred_pos, pred_edges, tol)
    lg = float(_edge_lengths(gt_pos, gt_edges).sum())
    return {"value": float(cg.sum()) / lg if lg else 0.0, "gt_length_mm": lg}


def overlap(pred_pos, pred_edges, gt_pos, gt_edges, tol):
    """Overlap (OV): (C_p + C_g) / (|G_p| + |G_g|).

    The overlap of Schaap et al. (2009, CAT08) with covered lengths in place of
    point counts and tau in place of the local vessel radius. Without CAT08's
    ordered point correspondence it needs no root. It equals Edge F1 whenever
    C_p = C_g and differs where one graph's matched part is longer than the
    other's (a zigzag, or two parallel arcs near one).
    """
    if not len(pred_edges) or not len(gt_edges):
        return {"value": 0.0}
    cp, lp, cg, lg = _two_way(pred_pos, pred_edges, gt_pos, gt_edges, tol)
    return {"value": (cp + cg) / (lp + lg) if lp + lg else 0.0}


def branch_coverage_f1_80(pred_pos, pred_edges, gt_pos, gt_edges, tol,
                          coverage=0.8):
    """Branch coverage F1 (80%): a branch counts as detected when at least
    80% of its length lies within tol of the other graph (the union of all its
    branches); precision over predicted branches, recall over reference
    branches, value their F1. The detection rule of EXACT'09 and ATM'22."""
    if not len(pred_edges) or not len(gt_edges):
        return {"value": 0.0}

    def detected(a_pos, a_edges, b_pos, b_edges):
        cov = covered_length_per_edge(a_pos, a_edges, b_pos, b_edges, tol)
        lens = _edge_lengths(a_pos, a_edges)
        hits = []
        for seq in _branch_edges(a_pos, a_edges):
            ks = [k for k, _ in seq]
            total = float(lens[ks].sum())
            if total > _EPS:
                hits.append(float(cov[ks].sum()) / total >= coverage)
        return hits

    hp = detected(pred_pos, pred_edges, gt_pos, gt_edges)
    hg = detected(gt_pos, gt_edges, pred_pos, pred_edges)
    if not hp or not hg:
        return {"value": 0.0}
    prec, rec = float(np.mean(hp)), float(np.mean(hg))
    return {"value": f1(prec, rec), "precision": prec, "recall": rec,
            "n_pred_branches": len(hp), "n_gt_branches": len(hg)}


def branch_coverage_f1(pred_pos, pred_edges, gt_pos, gt_edges, tol):
    """Branch coverage F1: the fraction of each branch covered by a single
    branch of the other graph.

    For every branch, the longest stretch lying within tol of one branch of the
    other graph, as a fraction of its length; length-weighted mean over the
    reference branches (recall) and over the predicted branches (precision),
    value their F1. Requiring one counterpart branch, rather than the union of
    all of them, is what makes a vessel traced in pieces, or two vessels merged
    into one branch, score below 1.
    """
    if not len(pred_edges) or not len(gt_edges):
        return {"value": 0.0}

    def directed(a_pos, a_edges, b_pos, b_edges):
        ai, bi, lo, hi = pair_intervals(a_pos, a_edges, b_pos, b_edges, tol)
        b_branch = np.full(len(b_edges), -1)
        for n, seq in enumerate(_branch_edges(b_pos, b_edges)):
            for k, _ in seq:
                b_branch[k] = n
        lens = _edge_lengths(a_pos, a_edges)
        # runs per (A edge, B branch): merge intervals within that group
        grp = ai * (len(b_edges) + 1) + (b_branch[bi] + 1)
        g, mlo, mhi = _merge(grp, lo, hi)
        per_branch = {}
        for gg, a, b in zip(g.tolist(), mlo.tolist(), mhi.tolist()):
            e, br = divmod(gg, len(b_edges) + 1)
            per_branch.setdefault(br, {}).setdefault(e, []).append((a, b))
        fracs, weights = [], []
        for seq in _branch_edges(a_pos, a_edges):
            total = float(lens[[k for k, _ in seq]].sum())
            if total < _EPS:
                continue
            ks = {k for k, _ in seq}
            best = 0.0
            for runs in per_branch.values():
                if not ks & runs.keys():
                    continue
                merged, _ = _branch_runs(seq, lens, runs)
                if merged:
                    best = max(best, max(b - a for a, b in merged))
            fracs.append(min(1.0, best / total))
            weights.append(total)
        if not fracs:
            return 0.0
        return float(np.average(fracs, weights=weights))

    rec = directed(gt_pos, gt_edges, pred_pos, pred_edges)
    prec = directed(pred_pos, pred_edges, gt_pos, gt_edges)
    return {"value": f1(prec, rec), "precision": prec, "recall": rec}


def ascd(pred_pos, pred_edges, gt_pos, gt_edges, tol):
    """ASCD, the average symmetric centerline distance (Liu et al. 2025), in mm:
    (1/2) (mean over G_p of d(x, G_g) + mean over G_g of d(y, G_p)), each mean
    an integral along the curve, resolved to tol/100."""
    if not len(pred_edges) or not len(gt_edges):
        return {"value": float("inf")}
    lp, fp = distance_profile(pred_pos, pred_edges, gt_pos, gt_edges, tol)
    lg, fg = distance_profile(gt_pos, gt_edges, pred_pos, pred_edges, tol)
    p2g, g2p = _mean_of(lambda d: d, lp, fp), _mean_of(lambda d: d, lg, fg)
    return {"value": (p2g + g2p) / 2, "p2g_mm": p2g, "g2p_mm": g2p}


def hd95(pred_pos, pred_edges, gt_pos, gt_edges, tol):
    """HD95, in mm: the 95th percentile of the distance from each curve to the
    other, pooled over both directions and weighted by length; accurate to
    tol/100."""
    if not len(pred_edges) or not len(gt_edges):
        return {"value": float("inf")}
    lp, fp = distance_profile(pred_pos, pred_edges, gt_pos, gt_edges, tol)
    lg, fg = distance_profile(gt_pos, gt_edges, pred_pos, pred_edges, tol)
    w = np.r_[lp, lg]
    v = np.r_[fp[:, 1], fg[:, 1]]                # piece midpoints; spread < tol/100
    order = np.argsort(v)
    c = np.cumsum(w[order]) / w.sum()
    return {"value": float(v[order][np.searchsorted(c, 0.95)])}


def netmets_geometry(pred_pos, pred_edges, gt_pos, gt_edges, tol):
    """NetMets geometry (Mayerich et al. 2012): GF = mean along a curve of
    1 - exp(-d^2 / 2 sigma^2), sigma = tol, in each direction (GFPR on the
    prediction, GFNR on the reference); value = F1(1 - GFPR, 1 - GFNR)."""
    if not len(pred_edges) or not len(gt_edges):
        return {"value": 0.0}
    w = lambda d: 1.0 - np.exp(-d ** 2 / (2 * tol ** 2))
    lp, fp = distance_profile(pred_pos, pred_edges, gt_pos, gt_edges, tol)
    lg, fg = distance_profile(gt_pos, gt_edges, pred_pos, pred_edges, tol)
    gfpr, gfnr = _mean_of(w, lp, fp), _mean_of(w, lg, fg)
    return {"value": f1(1 - gfpr, 1 - gfnr), "GFPR": gfpr, "GFNR": gfnr}


def apls(pred_pos, pred_edges, gt_pos, gt_edges, tol, n_pairs=200,
         spacing_tau=5.0, seed=0):
    """APLS (Van Etten et al. 2019): agreement of shortest-path lengths.

    Control points are the junctions, the endpoints and a point every 5 tau
    along each branch (the original places them every 50 m along roads); each
    is carried to the closest point of the other curve within tol, as the
    original inserts it on the nearest edge. For a pair of control points with
    path length L on one graph and L' between their counterparts on the other,
    the pair scores 1 - min(1, |L' - L| / L), and 0 when a point has no
    counterpart or no path exists. APLS is the harmonic mean of the two
    directions.
    """
    if not len(pred_edges) or not len(gt_edges):
        return {"value": 0.0}

    def directed(s_pos, s_edges, d_pos, d_edges):
        cp = _arc_control_points(s_pos, s_edges, spacing_tau * tol)
        if len(cp) < 2:
            return []
        rng = np.random.default_rng(seed)
        m = min(n_pairs, len(cp) * (len(cp) - 1) // 2)
        pr = rng.choice(len(cp), size=(m, 2))
        pr = pr[pr[:, 0] != pr[:, 1]]
        if not len(pr):
            return []
        l_src = _path_lengths(s_pos, s_edges, cp[pr[:, 0]], cp[pr[:, 1]])
        dist, de, dt = closest_on_curve(_point_xyz(s_pos, s_edges, cp),
                                        d_pos, d_edges, radius=tol)
        snapped = np.c_[de, dt]
        ok = (de[pr[:, 0]] >= 0) & (de[pr[:, 1]] >= 0) & np.isfinite(l_src)
        l_dst = np.full(len(pr), np.inf)
        if ok.any():
            l_dst[ok] = _path_lengths(d_pos, d_edges, snapped[pr[ok, 0]],
                                      snapped[pr[ok, 1]])
        scores = []
        for k in range(len(pr)):
            l = l_src[k]
            if not np.isfinite(l):
                continue
            if not np.isfinite(l_dst[k]):
                scores.append(0.0)
            elif l <= 1e-12:
                scores.append(1.0)
            else:
                scores.append(1.0 - min(1.0, abs(l_dst[k] - l) / l))
        return scores

    s_gp = directed(gt_pos, gt_edges, pred_pos, pred_edges)
    s_pg = directed(pred_pos, pred_edges, gt_pos, gt_edges)
    d_gp = float(np.mean(s_gp)) if s_gp else 0.0
    d_pg = float(np.mean(s_pg)) if s_pg else 0.0
    value = 2 / (1 / d_gp + 1 / d_pg) if d_gp > 0 and d_pg > 0 else 0.0
    return {"value": value, "D_pred_given_gt": d_gp, "D_gt_given_pred": d_pg}
