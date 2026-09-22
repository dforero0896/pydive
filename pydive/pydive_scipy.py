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
        
        # Compute circumcenter using the formula from geometric primitives
        # Based on: https://en.wikipedia.org/wiki/Circumscribed_sphere#Cartesian_coordinates
        ax, ay, az = p0
        bx, by, bz = p1
        cx, cy, cz = p2
        dx, dy, dz = p3
        
        # Compute differences
        bax = bx - ax
        bay = by - ay
        baz = bz - az
        cax = cx - ax
        cay = cy - ay
        caz = cz - az
        dax = dx - ax
        day = dy - ay
        daz = dz - az
        
        # Compute cross products and dot products
        cross_bc_x = bay * caz - baz * cay
        cross_bc_y = baz * cax - bax * caz
        cross_bc_z = bax * cay - bay * cax
        
        cross_bd_x = bay * daz - baz * day
        cross_bd_y = baz * dax - bax * daz
        cross_bd_z = bax * day - bay * dax
        
        cross_cd_x = cay * daz - caz * day
        cross_cd_y = caz * dax - cax * daz
        cross_cd_z = cax * day - cay * dax
        
        # Compute determinant
        det = 2.0 * (bax * cross_cd_x + bay * cross_cd_y + baz * cross_cd_z)
        
        if abs(det) < 1e-15:
            # Degenerate tetrahedron, skip
            continue
        
        # Compute squared lengths
        bax2 = bax*bax + bay*bay + baz*baz
        cax2 = cax*cax + cay*cay + caz*caz
        dax2 = dax*dax + day*day + daz*daz
        
        # Compute circumcenter
        centers[k, 0] = ax + (bax2 * cross_cd_x + cax2 * cross_bd_x + dax2 * cross_bc_x) / det
        centers[k, 1] = ay + (bax2 * cross_cd_y + cax2 * cross_bd_y + dax2 * cross_bc_y) / det
        centers[k, 2] = az + (bax2 * cross_cd_z + cax2 * cross_bd_z + dax2 * cross_bc_z) / det
        
        # Compute radius
        dx_center = centers[k, 0] - ax
        dy_center = centers[k, 1] - ay
        dz_center = centers[k, 2] - az
        radii[k] = np.sqrt(dx_center*dx_center + dy_center*dy_center + dz_center*dz_center)
    
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


@njit(parallel=True, fastmath=True)
def compute_dtfe_weights(vertices: np.ndarray, volumes: np.ndarray, n_points: int) -> np.ndarray:
    """
    Compute DTFE weights for each point.
    
    DTFE density at a point is inversely proportional to the sum of volumes
    of all tetrahedra containing that point.
    
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


def get_void_catalog_scipy(points: np.ndarray, periodic: bool = False) -> np.ndarray:
    """
    Get basic void catalog using SciPy backend.
    
    Parameters
    ----------
    points : ndarray (N, 3)
        Input point coordinates
    periodic : bool
        Whether to use periodic boundary conditions (not implemented)
        
    Returns
    -------
    output : ndarray (n_simplices, 4)
        Array with columns [x, y, z, r] for each simplex circumcenter
    """
    if periodic:
        raise NotImplementedError("Periodic boundary conditions not implemented in scipy backend")
    
    points = np.ascontiguousarray(points, dtype=np.double)
    
    # Compute Delaunay triangulation
    tri = Delaunay(points)
    
    # Get vertex indices for all tetrahedra
    vertices = tri.simplices.astype(np.int64)
    
    # Compute circumcenters and radii (all in numba)
    centers, radii = compute_circumcenters_and_radii(vertices, points)
    
    # Combine into output array
    output = np.column_stack([centers, radii])
    
    return output


def get_void_catalog_full_scipy(points: np.ndarray, periodic: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get full void catalog with DTFE, volumes, and areas using SciPy backend.
    
    Parameters
    ----------
    points : ndarray (N, 3)
        Input point coordinates
    periodic : bool
        Whether to use periodic boundary conditions (not implemented)
        
    Returns
    -------
    output : ndarray (n_simplices, 7)
        Array with columns [x, y, z, r, volume, dtfe_interp, area]
    dtfe : ndarray (N,)
        DTFE density estimate at each input point
    """
    if periodic:
        raise NotImplementedError("Periodic boundary conditions not implemented in scipy backend")
    
    points = np.ascontiguousarray(points, dtype=np.double)
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
