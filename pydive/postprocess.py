"""
Post-processing utilities for DIVE void catalogs.

All heavy loops are Numba-JIT compiled; the spatial search uses a cell-list
neighbor search, so no catalog-size-squared Python loops are ever executed.

Main entry point: :func:`filter_non_overlapping`

The filter keeps only the largest non-overlapping spheres in a greedy
"populated-first" pass (the classic void-merging criterion of Einasto et al.
2016 / Nada et al. 2018):

1. Voids are processed in decreasing radius order.
2. A void that is *not* contained inside any already-kept (larger) void is
   kept ("central"); otherwise it is discarded as a "satellite".
3. Optionally, a second pass merges satellites into their host central when
   the overlap fraction ``overlap / r_sat <= merge_threshold`` (i.e. the
   satellite pokes out of the central by at most that fraction of its own
   radius). The merged void takes the volume-weighted centroid and the radius
   of the smallest sphere enclosing both.

Relationship to the central/satellite split
-------------------------------------------
This *is* the same geometric decomposition you were planning as the
central-satellite split: the greedy containment pass labels every void either
as a central (kept root) or as a satellite attached to exactly one central
(the first larger sphere containing it). The difference is only what you do
with the labels afterwards:

* ``return_labels=True`` gives you the raw split: ``labels == -1`` marks
  centrals, ``labels == c`` marks the satellite belonging to central ``c``.
* ``merge_threshold > 0`` turns the split into an actual merge (one combined
  void per family), which is *not* equivalent to simply dropping the
  overlapping spheres.
* ``merge_threshold = 0`` (default) simply discards satellites, keeping the
  pure non-overlapping maximal-sphere catalog.

Periodic boundaries
-------------------
If ``boxsize`` is given, distances use the minimum-image convention. Note
that a sphere whose radius exceeds half the box size is not well defined
under periodic boundaries; such spheres are never considered to contain
another void's center.
"""

import numpy as np
from numba import njit


# ---------------------------------------------------------------------------
# Numba kernels
# ---------------------------------------------------------------------------

@njit(cache=True, fastmath=True)
def _cell_coord(x, lo_d, cell_size, nd, boxsize):
    """Cell index along one axis, clamped (open) or wrapped (periodic)."""
    i = int(np.floor((x - lo_d) / cell_size))
    if boxsize > 0.0:
        while i < 0:
            i += nd
        return i % nd
    if i < 0:
        i = 0
    if i >= nd:
        i = nd - 1
    return i


@njit(cache=True, fastmath=True)
def _greedy_filter(centers, radii, order, boxsize, merge_threshold):
    """Greedy populated-first pass over voids sorted by decreasing radius.

    A uniform cell list with fixed cell size ``cell = 2*r_max`` (the largest
    void radius, which bounds every possible containment distance:
    ``d <= r_j`` and all kept radii ``r_j <= r_max``) indexes the spheres
    kept so far. Containment of sphere *i* in a kept sphere *j* is tested as
    ``d + r_i <= r_j`` using minimum-image distances when ``boxsize > 0``.

    Returns
    -------
    keep : bool array (n,)              -- True for retained voids
    new_centers, new_radii : (n,3), (n,)-- updated geometry (only changed
                                           for hosts that absorbed satellites)
    labels : int array (n,)             -- -1: central root,
                                           j: satellite of original row j,
                                           -2: no host (should not occur)
    """
    n = centers.shape[0]
    keep = np.zeros(n, np.bool_)
    labels = np.full(n, -2, dtype=np.int64)
    new_centers = centers.copy()
    new_radii = radii.copy()

    periodic = boxsize > 0.0
    half_box = 0.5 * boxsize

    # --- grid setup ------------------------------------------------------
    # Query radius is bounded by the LARGEST void radius (a sphere can only
    # contain another one if its own radius is at least as large), so a fixed
    # cell size of 2*r_max keeps every possible containment pair within the
    # 3x3x3 neighbourhood of the query cell.
    r_max = radii[order[0]]
    if r_max <= 0.0:
        r_max = 1.0
    if periodic:
        # A sphere with r >= boxsize/2 is not well defined under PBC and can
        # never act as a container here (see skip below), so the containment
        # search radius only needs to cover spheres with r < boxsize/2. Using
        # cell size L/2 lets us wrap the grid toroidally: every possible host
        # center lies within half a cell of its own wrapped position, so a
        # single-cell (non-neighbour) lookup is already exact.
        cell_size = 0.5 * boxsize
        ns = np.ones(3, dtype=np.int64)
        lo = np.zeros(3)
    else:
        cell_size = 2.0 * r_max
        lo = np.empty(3)
        ns = np.empty(3, dtype=np.int64)
        for d in range(3):
            mn = centers[0, d]
            mx = centers[0, d]
            for i in range(1, n):
                v = centers[i, d]
                if v < mn:
                    mn = v
                if v > mx:
                    mx = v
            lo[d] = mn
            nc = int(np.floor((mx - mn) / cell_size)) + 1
            ns[d] = nc
    total = ns[0] * ns[1] * ns[2]

    # linked-list grid over kept spheres: head[c] -> slot, next_[slot]
    head = np.full(total, -1, dtype=np.int64)
    nxt = np.full(n, -1, dtype=np.int64)
    slot_of = np.full(n, -1, dtype=np.int64)   # original row -> slot
    orig_of = np.arange(n, dtype=np.int64)     # slot == row index here

    n_kept = 0
    for k in range(n):
        i = order[k]
        ri = new_radii[i]
        cx = new_centers[i, 0]
        cy = new_centers[i, 1]
        cz = new_centers[i, 2]

        host = -1
        best_overlap = -1.0
        if n_kept > 0:
            if periodic:
                dx0 = _cell_coord(cx, lo[0], cell_size, ns[0], boxsize)
                dx1 = dx0
                dy0 = _cell_coord(cy, lo[1], cell_size, ns[1], boxsize)
                dy1 = dy0
                dz0 = _cell_coord(cz, lo[2], cell_size, ns[2], boxsize)
                dz1 = dz0
            else:
                ix = _cell_coord(cx, lo[0], cell_size, ns[0], 0.0)
                iy = _cell_coord(cy, lo[1], cell_size, ns[1], 0.0)
                iz = _cell_coord(cz, lo[2], cell_size, ns[2], 0.0)
                dx0 = max(ix - 1, 0)
                dx1 = min(ix + 1, ns[0] - 1)
                dy0 = max(iy - 1, 0)
                dy1 = min(iy + 1, ns[1] - 1)
                dz0 = max(iz - 1, 0)
                dz1 = min(iz + 1, ns[2] - 1)

            for zz in range(dz0, dz1 + 1):
                for yy in range(dy0, dy1 + 1):
                    base = (zz * ns[1] + yy) * ns[0]
                    for xx in range(dx0, dx1 + 1):
                        slot = head[base + xx]
                        while slot >= 0:
                            j = orig_of[slot]
                            rj = new_radii[j]
                            if periodic and rj >= half_box:
                                # sphere bigger than half the box is not
                                # well defined under PBC: cannot contain
                                slot = nxt[slot]
                                continue
                            rx = new_centers[j, 0] - cx
                            ry = new_centers[j, 1] - cy
                            rz = new_centers[j, 2] - cz
                            if periodic:
                                if rx > half_box:
                                    rx -= boxsize
                                elif rx < -half_box:
                                    rx += boxsize
                                if ry > half_box:
                                    ry -= boxsize
                                elif ry < -half_box:
                                    ry += boxsize
                                if rz > half_box:
                                    rz -= boxsize
                                elif rz < -half_box:
                                    rz += boxsize
                            d2 = rx * rx + ry * ry + rz * rz
                            if d2 <= rj * rj:  # cheap reject: need d <= rj
                                d = np.sqrt(d2)
                                if d + ri <= rj:
                                    ov = ri + rj - d  # overlap depth
                                    if ov > best_overlap:
                                        best_overlap = ov
                                        host = j
                            slot = nxt[slot]

        if host < 0:
            # new central: insert into grid
            keep[i] = True
            labels[i] = -1
            if periodic:
                gx = _cell_coord(cx, lo[0], cell_size, ns[0], boxsize)
                gy = _cell_coord(cy, lo[1], cell_size, ns[1], boxsize)
                gz = _cell_coord(cz, lo[2], cell_size, ns[2], boxsize)
            else:
                gx = _cell_coord(cx, lo[0], cell_size, ns[0], 0.0)
                gy = _cell_coord(cy, lo[1], cell_size, ns[1], 0.0)
                gz = _cell_coord(cz, lo[2], cell_size, ns[2], 0.0)
            c = (gz * ns[1] + gy) * ns[0] + gx
            nxt[i] = head[c]
            head[c] = i
            slot_of[i] = c
            n_kept += 1
        elif merge_threshold > 0.0 and best_overlap <= merge_threshold * ri:
            # shallow overlap -> merge satellite into its host central
            hj = host
            rh = new_radii[hj]
            dh = np.sqrt((new_centers[hj, 0] - cx) ** 2 +
                         (new_centers[hj, 1] - cy) ** 2 +
                         (new_centers[hj, 2] - cz) ** 2)
            vi = rh * rh * rh + ri * ri * ri
            ncx = (new_centers[hj, 0] * rh * rh * rh + cx * ri * ri * ri) / vi
            ncy = (new_centers[hj, 1] * rh * rh * rh + cy * ri * ri * ri) / vi
            ncz = (new_centers[hj, 2] * rh * rh * rh + cz * ri * ri * ri) / vi
            nr = max(rh, ri, 0.5 * (dh + ri + rh))
            new_centers[hj, 0] = ncx
            new_centers[hj, 1] = ncy
            new_centers[hj, 2] = ncz
            new_radii[hj] = nr
            keep[i] = True          # merged family stays in the catalog
            labels[i] = hj
        else:
            # deep overlap (or merging disabled) -> satellite dropped
            keep[i] = False
            labels[i] = host

    return keep, new_centers, new_radii, labels, n_kept


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def filter_non_overlapping(catalog, radius_col=3, boxsize=None,
                           merge_threshold=0.0, return_labels=False):
    """Keep only the largest non-overlapping spheres of a void catalog.

    Parameters
    ----------
    catalog : ndarray (n, m)
        Void catalog. Columns 0-2 must be the circumcenter coordinates.
    radius_col : int, optional
        Index of the radius column (default 3, matching both backends'
        output layout ``[x, y, z, r, ...]``).
    boxsize : float, optional
        If given, periodic (minimum-image) distances are used.
    merge_threshold : float, optional
        ``0`` (default): satellites that overlap a larger void are dropped.
        ``f > 0``: satellites whose overlap depth is at most ``f * r_sat``
        (i.e. poking out by less than a fraction ``f`` of their radius) are
        merged into their host central instead of being dropped. The merged
        family stays in the catalog: its *central* row carries the updated
        geometry (volume-weighted centroid and enclosing-sphere radius of
        the union), while satellite rows are kept as members with their
        original geometry. Use the returned counts to identify families.
    return_labels : bool, optional
        Also return per-kept-row information (see below).

    Returns
    -------
    filtered : ndarray
        Catalog rows for the retained voids, sorted by decreasing (possibly
        updated) radius. With merging enabled the central row of each family
        has columns [x, y, z, r] updated to the merged sphere; extra columns
        (volume, dtfe, area) are carried over unchanged from the source row.
    n_members : ndarray (only if return_labels=True)
        Integer array aligned with ``filtered`` giving, for each retained
        row, the number of satellite voids absorbed into it (0 if none).
        With ``merge_threshold == 0`` this is always 0 (satellites are
        dropped, not absorbed); use :func:`central_satellite_split` if you
        want the full assignment table.

    Notes
    -----
    This is the greedy "populated-first" emptiest-sphere selection. It is
    identical to the first stage of a central/satellite void decomposition:
    centrals are the kept roots, satellites are the voids contained in them.
    See :func:`central_satellite_split` for the version that returns the
    explicit assignment instead of filtering.

    Examples
    --------
    >>> from pydive import get_void_catalog_full
    >>> from pydive.postprocess import filter_non_overlapping
    >>> cat, dtfe = get_void_catalog_full(points, backend='scipy')
    >>> big = filter_non_overlapping(cat)                 # drop overlaps
    >>> fam = filter_non_overlapping(cat, merge_threshold=0.2)  # merge them
    """
    catalog = np.ascontiguousarray(catalog, dtype=np.double)
    if catalog.ndim != 2 or catalog.shape[1] <= radius_col:
        raise ValueError(
            f"catalog must be (n, m) with m > {radius_col}, got {catalog.shape}")
    n = catalog.shape[0]
    if n == 0:
        empty = catalog.copy()
        return (empty, np.zeros(0, dtype=np.int64)) if return_labels else empty

    centers = np.ascontiguousarray(catalog[:, :3])
    radii = np.ascontiguousarray(catalog[:, radius_col])
    bs = 0.0 if boxsize is None else float(boxsize)

    order = np.argsort(-radii, kind='stable').astype(np.int64)
    keep, new_centers, new_radii, labels, _ = _greedy_filter(
        centers, radii, order, bs, float(merge_threshold))

    kept_idx = np.where(keep)[0]
    sat_counts = np.zeros(n, dtype=np.int64)
    for lab in labels:
        if lab >= 0:
            sat_counts[lab] += 1

    out = catalog[kept_idx].copy()
    out[:, :3] = new_centers[kept_idx]
    out[:, radius_col] = new_radii[kept_idx]
    # re-sort by decreasing radius (insertion order already follows it, but
    # merged radii may have grown)
    final_order = np.argsort(-out[:, radius_col], kind='stable')
    out = out[final_order]
    counts = sat_counts[kept_idx][final_order]

    if return_labels:
        return out, counts
    return out


def central_satellite_split(catalog, radius_col=3, boxsize=None):
    """Full central/satellite decomposition of a void catalog.

    Equivalent to the containment pass used by
    :func:`filter_non_overlapping`, but instead of dropping satellites it
    returns the explicit assignment.

    Parameters
    ----------
    catalog, radius_col, boxsize : see :func:`filter_non_overlapping`.

    Returns
    -------
    centrals : ndarray
        Rows of the catalog that are void roots (largest non-overlapping
        spheres), sorted by decreasing radius.
    satellites : ndarray
        Rows of every remaining void, grouped by host.
    host_of : ndarray (len(catalog),)
        ``host_of[j]`` is the row index in ``centrals`` of the void that
        contains satellite ``j``, or ``-1`` if ``j`` is itself a central.
    """
    catalog = np.ascontiguousarray(catalog, dtype=np.double)
    n = catalog.shape[0]
    centers = np.ascontiguousarray(catalog[:, :3])
    radii = np.ascontiguousarray(catalog[:, radius_col])
    bs = 0.0 if boxsize is None else float(boxsize)

    order = np.argsort(-radii, kind='stable').astype(np.int64)
    keep, _, _, labels, _ = _greedy_filter(centers, radii, order, bs, 0.0)

    kept_idx = np.where(keep)[0]
    pos_of = np.full(n, -1, dtype=np.int64)
    for p, idx in enumerate(kept_idx):
        pos_of[idx] = p

    host_of = np.full(n, -1, dtype=np.int64)
    for j in range(n):
        if keep[j]:
            host_of[j] = -1
        elif labels[j] >= 0:
            host_of[j] = pos_of[labels[j]]
        else:
            host_of[j] = -2  # no host found (should not happen)

    centrals = catalog[kept_idx]
    sat_mask = ~keep
    satellites = catalog[sat_mask]
    host_of = host_of[sat_mask]
    return centrals, satellites, host_of


__all__ = ['filter_non_overlapping', 'central_satellite_split']
