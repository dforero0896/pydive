#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False
import cython
import numpy as np
from libc.stdlib cimport malloc, free
from libcpp.vector cimport vector
from libcpp.cstring cimport memcpy

cdef extern from "delaunay_backend_standalone.cpp":
    cdef struct DelaunayOutput:
        vector[double] x
        vector[double] y
        vector[double] z
        vector[double] r
        vector[double] volume
        vector[double] area
        vector[double] dtfe
        vector[size_t] vertices[4]
        size_t n_simplices
    
    DelaunayOutput* cdelaunay_basic(double* X, double* Y, double* Z, size_t n_points) nogil
    DelaunayOutput* cdelaunay_full_basic(double* X, double* Y, double* Z, size_t n_points) nogil
    void free_delaunay_output(DelaunayOutput* output) nogil

@cython.boundscheck(False)
@cython.wraparound(False)
def get_void_catalog_cgal(double[:, :] points, bint periodic=False):
    """Get basic void catalog using CGAL backend.
    
    Parameters
    ----------
    points : ndarray (N, 3)
        Input point coordinates
    periodic : bool
        Whether to use periodic boundary conditions (not yet implemented in standalone)
        
    Returns
    -------
    output : ndarray (n_simplices, 4)
        Array with columns [x, y, z, r] for each simplex circumcenter
    """
    if periodic:
        raise NotImplementedError("Periodic boundary conditions not yet implemented in portable backend")
    
    cdef Py_ssize_t i, k
    cdef size_t n_points = points.shape[0]
    cdef double[:] x_arr = np.ascontiguousarray(points[:, 0])
    cdef double[:] y_arr = np.ascontiguousarray(points[:, 1])
    cdef double[:] z_arr = np.ascontiguousarray(points[:, 2])
    
    cdef DelaunayOutput* result = cdelaunay_basic(
        <double*>&x_arr[0],
        <double*>&y_arr[0],
        <double*>&z_arr[0],
        n_points
    )
    
    cdef size_t n_simplices = result.n_simplices
    output = np.zeros((n_simplices, 4), dtype=np.double)
    cdef double[:, :] output_view = output
    
    for k in range(n_simplices):
        output_view[k, 0] = result.x[k]
        output_view[k, 1] = result.y[k]
        output_view[k, 2] = result.z[k]
        output_view[k, 3] = result.r[k]
    
    free_delaunay_output(result)
    return output

@cython.boundscheck(False)
@cython.wraparound(False)
def get_void_catalog_full(double[:, :] points, bint periodic=False):
    """Get full void catalog with DTFE, volumes, and areas using CGAL backend.
    
    Parameters
    ----------
    points : ndarray (N, 3)
        Input point coordinates
    periodic : bool
        Whether to use periodic boundary conditions (not yet implemented in standalone)
        
    Returns
    -------
    output : ndarray (n_simplices, 7)
        Array with columns [x, y, z, r, volume, dtfe_interp, area]
    dtfe : ndarray (N,)
        DTFE density estimate at each input point
    """
    if periodic:
        raise NotImplementedError("Periodic boundary conditions not yet implemented in portable backend")
    
    cdef Py_ssize_t i, k, j
    cdef size_t n_points = points.shape[0]
    cdef double[:] x_arr = np.ascontiguousarray(points[:, 0])
    cdef double[:] y_arr = np.ascontiguousarray(points[:, 1])
    cdef double[:] z_arr = np.ascontiguousarray(points[:, 2])
    
    cdef DelaunayOutput* result = cdelaunay_full_basic(
        <double*>&x_arr[0],
        <double*>&y_arr[0],
        <double*>&z_arr[0],
        n_points
    )
    
    cdef size_t n_simplices = result.n_simplices
    output = np.zeros((n_simplices, 7), dtype=np.double)
    dtfe = np.zeros(n_points, dtype=np.double)
    
    cdef double[:, :] output_view = output
    cdef double[:] dtfe_view = dtfe
    cdef double w, numerator
    
    # Copy DTFE values
    for k in range(n_points):
        dtfe_view[k] = 4.0 / result.dtfe[k]
    
    # Copy void properties and interpolate DTFE
    for k in range(n_simplices):
        output_view[k, 0] = result.x[k]
        output_view[k, 1] = result.y[k]
        output_view[k, 2] = result.z[k]
        output_view[k, 3] = result.r[k]
        output_view[k, 4] = result.volume[k]
        
        # Interpolate DTFE to void center using inverse distance weighting
        w = 1.0 / (result.r[k] ** 2)
        numerator = 0.0
        for i in range(4):
            if result.vertices[i][k] < n_points:
                numerator += w * dtfe_view[result.vertices[i][k]]
        output_view[k, 5] = numerator / (4.0 * w)
        output_view[k, 6] = result.area[k]
    
    free_delaunay_output(result)
    return output, dtfe
