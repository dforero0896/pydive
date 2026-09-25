"""
SciPy + Numba backend for pydive.

This module provides a pure Python implementation using SciPy's Delaunay
triangulation with Numba JIT compilation for performance-critical loops.

Key design principles:
1. All heavy computation (simplex iteration, volume/area calculations) is done in Numba
2. No Python loops over simplices - all vectorized or JIT-compiled
3. Same API as CGAL backend for easy switching

Performance note: While not as fast as CGAL, this backend provides good
performance and is much easier to install across different environments.
"""

import numpy as np
from scipy.spatial import Delaunay
from numba import njit, prange
from typing import Tuple


@njit(parallel=True, fastmath=True)
def compute_circumcenters_and_radii(vertices: np.ndarray, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute circumcenters and radii for all tetrahedra.
    
    Uses the formula from 'Computing the Circumcenter of a Tetrahedron' by Christer Ericson:
    C = p0 + (|a|^2(b×c) + |b|^2(c×a) + |c|^2(a×b)) / (2*a·(b×c))
    where a = p1-p0, b = p2-p0, c = p3-p0
    
    This formula is symmetric and numerically stable.
    
    Parameters
    ----------
    vertices : ndarray (n_simplices, 4)
        Indices of vertices for each tetrahedron
    points : ndarray (N, 3)
        Point coordinates
        
    Returns
    -------
    centers : ndarray (n_simplices, 3)
        Circumcenter coordinates
    radii : ndarray (n_simplices,)
        Circumradius for each tetrahedron
    """
    n_simplices = vertices.shape[0]
    centers = np.zeros((n_simplices, 3), dtype=np.double)
    radii = np.zeros(n_simplices, dtype=np.double)
    
    for k in prange(n_simplices):
        # Get the four vertices of this tetrahedron
        p0 = points[vertices[k, 0]]
        p1 = points[vertices[k, 1]]
        p2 = points[vertices[k, 2]]
        p3 = points[vertices[k, 3]]
        
        # Edge vectors from p0
        ax = p1[0] - p0[0]
        ay = p1[1] - p0[1]
        az = p1[2] - p0[2]
        
        bx = p2[0] - p0[0]
        by = p2[1] - p0[1]
        bz = p2[2] - p0[2]
        
        cx = p3[0] - p0[0]
        cy = p3[1] - p0[1]
        cz = p3[2] - p0[2]
        
        # Cross products: b×c, c×a, a×b
        bc_x = by*cz - bz*cy
        bc_y = bz*cx - bx*cz
        bc_z = bx*cy - by*cx
        
        ca_x = cy*az - cz*ay
        ca_y = cz*ax - cx*az
        ca_z = cx*ay - cy*ax
        
        ab_x = ay*bz - az*by
        ab_y = az*bx - ax*bz
        ab_z = ax*by - ay*bx
        
        # Squared lengths of edge vectors
        a2 = ax*ax + ay*ay + az*az
        b2 = bx*bx + by*by + bz*bz
        c2 = cx*cx + cy*cy + cz*cz
        
        # Denominator: 2 * a·(b×c)
        denom = 2.0 * (ax*bc_x + ay*bc_y + az*bc_z)
        
        if abs(denom) < 1e-15:
            # Degenerate tetrahedron
            continue
        
        inv_denom = 1.0 / denom
        
        # Numerator: |a|^2(b×c) + |b|^2(c×a) + |c|^2(a×b)
        num_x = a2*bc_x + b2*ca_x + c2*ab_x
        num_y = a2*bc_y + b2*ca_y + c2*ab_y
        num_z = a2*bc_z + b2*ca_z + c2*ab_z
        
        # Circumcenter relative to p0
        ux = num_x * inv_denom
        uy = num_y * inv_denom
        uz = num_z * inv_denom
        
        # Absolute circumcenter
        cx_abs = p0[0] + ux
        cy_abs = p0[1] + uy
        cz_abs = p0[2] + uz
        
        centers[k, 0] = cx_abs
        centers[k, 1] = cy_abs
        centers[k, 2] = cz_abs
        
        # Radius = distance from center to any vertex (use p0)
        dx = cx_abs - p0[0]
        dy = cy_abs - p0[1]
        dz = cz_abs - p0[2]
        radii[k] = np.sqrt(dx*dx + dy*dy + dz*dz)
    
    return centers, radii


@njit(parallel=True, fastmath=True)
def compute_tetrahedron_volumes(vertices: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Compute volumes for all tetrahedra.
    
    Volume = |det(p1-p0, p2-p0, p3-p0)| / 6
    
    Parameters
    ----------
    vertices : ndarray (n_simplices, 4)
        Indices of vertices for each tetrahedron
    points : ndarray (N, 3)
        Point coordinates
        
    Returns
    -------
    volumes : ndarray (n_simplices,)
        Volume for each tetrahedron
    """
    n_simplices = vertices.shape[0]
    volumes = np.zeros(n_simplices, dtype=np.double)
    
    for k in prange(n_simplices):
        p0 = points[vertices[k, 0]]
        p1 = points[vertices[k, 1]]
        p2 = points[vertices[k, 2]]
        p3 = points[vertices[k, 3]]
        
        # Compute vectors from p0
        v1x, v1y, v1z = p1 - p0
        v2x, v2y, v2z = p2 - p0
        v3x, v3y, v3z = p3 - p0
        
        # Compute scalar triple product (determinant)
        det = (v1x * (v2y * v3z - v2z * v3y) 
             - v1y * (v2x * v3z - v2z * v3x) 
             + v1z * (v2x * v3y - v2y * v3x))
        
        volumes[k] = abs(det) / 6.0
    
    return volumes


@njit(parallel=True, fastmath=True)
def compute_tetrahedron_areas(vertices: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Compute surface areas for all tetrahedra.
    
    Sum of areas of 4 triangular faces.
    
    Parameters
    ----------
    vertices : ndarray (n_simplices, 4)
        Indices of vertices for each tetrahedron
    points : ndarray (N, 3)
        Point coordinates
        
    Returns
    -------
    areas : ndarray (n_simplices,)
        Surface area for each tetrahedron
    """
    n_simplices = vertices.shape[0]
    areas = np.zeros(n_simplices, dtype=np.double)
    
    for k in prange(n_simplices):
        p0 = points[vertices[k, 0]]
        p1 = points[vertices[k, 1]]
        p2 = points[vertices[k, 2]]
        p3 = points[vertices[k, 3]]
        
        total_area = 0.0
        
        # Face 0-1-2
        v1 = p1 - p0
        v2 = p2 - p0
        cross = np.array([v1[1]*v2[2] - v1[2]*v2[1],
                         v1[2]*v2[0] - v1[0]*v2[2],
                         v1[0]*v2[1] - v1[1]*v2[0]])
        total_area += 0.5 * np.sqrt(cross[0]**2 + cross[1]**2 + cross[2]**2)
        
        # Face 0-1-3
        v2 = p3 - p0
        cross = np.array([v1[1]*v2[2] - v1[2]*v2[1],
                         v1[2]*v2[0] - v1[0]*v2[2],
                         v1[0]*v2[1] - v1[1]*v2[0]])
        total_area += 0.5 * np.sqrt(cross[0]**2 + cross[1]**2 + cross[2]**2)
        
        # Face 0-2-3
        v1 = p2 - p0
        cross = np.array([v1[1]*v2[2] - v1[2]*v2[1],
                         v1[2]*v2[0] - v1[0]*v2[2],
                         v1[0]*v2[1] - v1[1]*v2[0]])
        total_area += 0.5 * np.sqrt(cross[0]**2 + cross[1]**2 + cross[2]**2)
        
        # Face 1-2-3
        v1 = p2 - p1
        v2 = p3 - p1
        cross = np.array([v1[1]*v2[2] - v1[2]*v2[1],
                         v1[2]*v2[0] - v1[0]*v2[2],
                         v1[0]*v2[1] - v1[1]*v2[0]])
        total_area += 0.5 * np.sqrt(cross[0]**2 + cross[1]**2 + cross[2]**2)
        
        areas[k] = total_area
    
    return areas


@njit(fastmath=True)
def compute_dtfe_weights(vertices: np.ndarray, volumes: np.ndarray, n_points: int) -> np.ndarray:
    """
    Compute DTFE weights for each point.

    DTFE density at a point is inversely proportional to the sum of volumes
    of all tetrahedra containing that point.

    Implemented as a single sequential loop over simplices with direct
    accumulation into the per-point array: this avoids read-modify-write
    data races between threads (which would corrupt the sums), and is
    already JIT-compiled, so no Python-level loops are involved.

    Parameters
    ----------
    vertices : ndarray (n_simplices, 4)
        Indices of vertices for each tetrahedron
    volumes : ndarray (n_simplices,)
        Volume for each tetrahedron
    n_points : int
        Total number of points

    Returns
    -------
    dtfe : ndarray (n_points,)
        DTFE density estimate for each point
    """
    dtfe = np.zeros(n_points, dtype=np.double)

    n_simplices = volumes.shape[0]

    for k in range(n_simplices):
        vol = volumes[k]
        for i in range(4):
            idx = vertices[k, i]
            if idx < n_points:
                dtfe[idx] += vol

    # Convert to density (inverse of volume)
    for i in range(n_points):
        if dtfe[i] > 0:
            dtfe[i] = 4.0 / dtfe[i]  # Factor of 4 for tetrahedra

    return dtfe


@njit(parallel=True, fastmath=True)
def interpolate_dtfe_to_centers(vertices: np.ndarray, dtfe: np.ndarray, 
                                 centers: np.ndarray, points: np.ndarray,
                                 radii: np.ndarray) -> np.ndarray:
    """
    Interpolate DTFE values to void centers using inverse distance weighting.
    
    Parameters
    ----------
    vertices : ndarray (n_simplices, 4)
        Indices of vertices for each tetrahedron
    dtfe : ndarray (n_points,)
        DTFE values at each point
    centers : ndarray (n_simplices, 3)
        Circumcenter coordinates
    points : ndarray (N, 3)
        Point coordinates
    radii : ndarray (n_simplices,)
        Circumradius for each tetrahedron
        
    Returns
    -------
    dtfe_interp : ndarray (n_simplices,)
        Interpolated DTFE at each void center
    """
    n_simplices = centers.shape[0]
    dtfe_interp = np.zeros(n_simplices, dtype=np.double)
    
    for k in prange(n_simplices):
        r = radii[k]
        if r < 1e-15:
            continue
            
        w = 1.0 / (r * r)
        numerator = 0.0
        
        for i in range(4):
            idx = vertices[k, i]
            if idx < len(dtfe):
                numerator += w * dtfe[idx]
        
        dtfe_interp[k] = numerator / (4.0 * w)
    
    return dtfe_interp


def _mean_free_path(n_points: int, box_volume: float) -> float:
    """Mean inter-particle distance ``(n/V)**(-1/3)``, matching the CGAL backend."""
    return (n_points / box_volume) ** (-1.0 / 3.0)


def _pad_periodic_shell(points: np.ndarray, box_min: np.ndarray,
                        box_max: np.ndarray, cpy_range: float) -> np.ndarray:
    """
    Replicate only the boundary shells of the box (not the full 27 images).

    This mirrors ``cdelaunay_periodic_extend`` in the CGAL backend: points
    within ``cpy_range`` of each low/high face are copied across that face by
    exactly one box length. Corner/edge regions are duplicated once per axis,
    which is sufficient to recover the correct local neighbourhoods. Tiling
    all 27 periodic images would inflate memory and triangulation time by
    ~27x while producing the same simplices inside the box.

    Note: scipy.spatial.Delaunay (Qhull) has no native periodic Delaunay
    triangulation, so this shell padding is required for periodic boundaries.
    """
    box_size = box_max - box_min
    reps = [points]
    for ax in range(3):
        near_lo = points[:, ax] < box_min[ax] + cpy_range
        q = points[near_lo].copy()
        q[:, ax] += box_size[ax]
        reps.append(q)

        near_hi = points[:, ax] >= box_max[ax] - cpy_range
        q = points[near_hi].copy()
        q[:, ax] -= box_size[ax]
        reps.append(q)

    return np.ascontiguousarray(np.vstack(reps), dtype=np.double)


def get_void_catalog_scipy(points: np.ndarray, mode: str = 'open', boxsize: float = None,
                           cpy_range: float = 0.0) -> np.ndarray:
    """
    Get basic void catalog using SciPy backend.
    
    Parameters
    ----------
    points : ndarray (N, 3)
        Input point coordinates
    mode : str, optional
        Boundary mode: 'open' (no padding; also used for 'lightcone', since
        lightcones are not periodic) or 'periodic' (replicate boundary shells
        of the box). Default is 'open'.
    boxsize : float, optional
        Box size for periodic boundary conditions (box assumed to span
        [0, boxsize]). If None, estimated from point distribution as
        max(bbox_max - bbox_min).
    cpy_range : float, optional
        Width of the replicated boundary shell. If 0 (default), uses
        8 * mean inter-particle distance, matching the CGAL backend.

    Returns
    -------
    output : ndarray (n_simplices, 4)
        Array with columns [x, y, z, r] for each simplex circumcenter
    """
    points = np.ascontiguousarray(points, dtype=np.double)

    # Note: 'lightcone' is treated as 'open': a lightcone has no periodic
    # dimension to wrap around, so no padding is applied.
    if mode == 'periodic':
        # Estimate box size if not provided
        if boxsize is None:
            bbox_min = points.min(axis=0)
            bbox_max = points.max(axis=0)
            box_size_vec = bbox_max - bbox_min
            boxsize = max(box_size_vec)
        
        box_min = np.zeros(3, dtype=np.double)
        box_max = np.full(3, boxsize, dtype=np.double)

        # Copy range: default to 8x the mean inter-particle distance,
        # consistent with cdelaunay_periodic_extend in the CGAL backend.
        mfp = _mean_free_path(len(points), boxsize ** 3)
        copy_range = cpy_range if cpy_range > 0 else 8 * mfp

        # Replicate only the boundary shells (workaround for Qhull not
        # supporting periodic Delaunay triangulations natively)
        padded_points = _pad_periodic_shell(points, box_min, box_max, copy_range)
        
        # Define original box bounds (assume centered or from 0)
        original_min = box_min
        original_max = box_max
        
        # Compute Delaunay on padded points
        tri = Delaunay(padded_points)
        vertices = tri.simplices.astype(np.int64)
        
        # Compute circumcenters and radii
        centers, radii = compute_circumcenters_and_radii(vertices, padded_points)
        
        # Filter to keep only voids with centers in original box
        mask = ((centers[:, 0] >= original_min[0]) & (centers[:, 0] <= original_max[0]) &
                (centers[:, 1] >= original_min[1]) & (centers[:, 1] <= original_max[1]) &
                (centers[:, 2] >= original_min[2]) & (centers[:, 2] <= original_max[2]))
        
        centers = centers[mask]
        radii = radii[mask]
    else:
        # Compute Delaunay triangulation
        tri = Delaunay(points)
        
        # Get vertex indices for all tetrahedra
        vertices = tri.simplices.astype(np.int64)
        
        # Compute circumcenters and radii (all in numba)
        centers, radii = compute_circumcenters_and_radii(vertices, points)
    
    # Combine into output array
    output = np.column_stack([centers, radii])
    
    return output


def get_void_catalog_full_scipy(points: np.ndarray, mode: str = 'open', boxsize: float = None,
                                cpy_range: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get full void catalog with DTFE, volumes, and areas using SciPy backend.

    Parameters
    ----------
    points : ndarray (N, 3)
        Input point coordinates
    mode : str, optional
        Boundary mode: 'open' (no padding; also used for 'lightcone', since
        lightcones are not periodic) or 'periodic' (replicate boundary shells
        of the box). Default is 'open'.
    boxsize : float, optional
        Box size for periodic boundary conditions (box assumed to span
        [0, boxsize]). If None, estimated from point distribution as
        max(bbox_max - bbox_min).
    cpy_range : float, optional
        Width of the replicated boundary shell. If 0 (default), uses
        8 * mean inter-particle distance, matching the CGAL backend.

    Returns
    -------
    output : ndarray (n_simplices, 7)
        Array with columns [x, y, z, r, volume, dtfe_interp, area]
    dtfe : ndarray (N,)
        DTFE density estimate at each input point
    """
    points = np.ascontiguousarray(points, dtype=np.double)
    n_points_original = len(points)

    # Note: 'lightcone' is treated as 'open': a lightcone has no periodic
    # dimension to wrap around, so no padding is applied.
    if mode == 'periodic':
        # Estimate box size if not provided
        if boxsize is None:
            bbox_min = points.min(axis=0)
            bbox_max = points.max(axis=0)
            box_size_vec = bbox_max - bbox_min
            boxsize = max(box_size_vec)

        box_min = np.zeros(3, dtype=np.double)
        box_max = np.full(3, boxsize, dtype=np.double)

        # Copy range: default to 8x the mean inter-particle distance,
        # consistent with cdelaunay_periodic_full in the CGAL backend.
        mfp = _mean_free_path(n_points_original, boxsize ** 3)
        copy_range = cpy_range if cpy_range > 0 else 8 * mfp

        # Replicate only the boundary shells (workaround for Qhull not
        # supporting periodic Delaunay triangulations natively)
        padded_points = _pad_periodic_shell(points, box_min, box_max, copy_range)
        n_points_padded = len(padded_points)

        # Define original box bounds
        original_min = box_min
        original_max = box_max

        
        # Compute Delaunay on padded points
        tri = Delaunay(padded_points)
        vertices = tri.simplices.astype(np.int64)
        n_simplices = len(vertices)
        
        # Compute all properties (all in numba)
        centers, radii = compute_circumcenters_and_radii(vertices, padded_points)
        volumes = compute_tetrahedron_volumes(vertices, padded_points)
        areas = compute_tetrahedron_areas(vertices, padded_points)
        dtfe_padded = compute_dtfe_weights(vertices, volumes, n_points_padded)
        dtfe_interp = interpolate_dtfe_to_centers(vertices, dtfe_padded, centers, padded_points, radii)
        
        # Filter to keep only voids with centers in original box
        mask = ((centers[:, 0] >= original_min[0]) & (centers[:, 0] <= original_max[0]) &
                (centers[:, 1] >= original_min[1]) & (centers[:, 1] <= original_max[1]) &
                (centers[:, 2] >= original_min[2]) & (centers[:, 2] <= original_max[2]))
        
        centers = centers[mask]
        radii = radii[mask]
        volumes = volumes[mask]
        areas = areas[mask]
        dtfe_interp = dtfe_interp[mask]
        
        # Return DTFE for original points only
        dtfe = dtfe_padded[:n_points_original]
    else:
        n_points = len(points)
        
        # Compute Delaunay triangulation
        tri = Delaunay(points)
        
        # Get vertex indices for all tetrahedra
        vertices = tri.simplices.astype(np.int64)
        n_simplices = len(vertices)
        
        # Compute all properties (all in numba)
        centers, radii = compute_circumcenters_and_radii(vertices, points)
        volumes = compute_tetrahedron_volumes(vertices, points)
        areas = compute_tetrahedron_areas(vertices, points)
        dtfe = compute_dtfe_weights(vertices, volumes, n_points)
        dtfe_interp = interpolate_dtfe_to_centers(vertices, dtfe, centers, points, radii)
    
    # Combine into output array
    output = np.column_stack([centers, radii, volumes, dtfe_interp, areas])
    
    return output, dtfe


if __name__ == '__main__':
    # Quick test
    print("Testing scipy+numba backend...")
    np.random.seed(42)
    points = np.random.random((1000, 3)).astype(np.double) * 100
    
    print("Running basic catalog...")
    voids = get_void_catalog_scipy(points)
    print(f"  Found {len(voids)} simplices")
    
    print("Running full catalog...")
    voids_full, dtfe = get_void_catalog_full_scipy(points)
    print(f"  Found {len(voids_full)} simplices")
    print(f"  DTFE shape: {dtfe.shape}")
    
    print("\n✓ SciPy+Numba backend working!")
