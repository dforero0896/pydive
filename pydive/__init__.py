"""
pydive - Python Delaunay triangulation-based void finder.

This package provides tools for computing void catalogs from point distributions
using Delaunay triangulation with multiple backends (CGAL, SciPy+Numba).

Backends:
- CGAL: Fastest implementation, requires CGAL library installation
- SciPy+Numba: Pure Python with JIT compilation, easier to install

Usage:
    from pydive import get_void_catalog, get_void_catalog_full
    
    # Auto-select best available backend
    voids = get_void_catalog(points)
    
    # Explicit backend selection
    voids = get_void_catalog(points, backend='cgal')  # or 'scipy'
    
    # Full catalog with DTFE, volumes, areas
    voids, dtfe = get_void_catalog_full(points)
    
    # Check backend availability
    from pydive import check_backend_status
    check_backend_status()
"""

__version__ = '1.0.0'

# Backend management
_available_backends = {}

def _check_cgal():
    """Check if CGAL backend is available."""
    try:
        from .delaunay_backend import get_void_catalog_cgal
        return True
    except (ImportError, ModuleNotFoundError):
        return False

def _check_scipy():
    """Check if SciPy+Numba backend is available."""
    try:
        from .pydive_scipy import get_void_catalog_scipy
        return True
    except (ImportError, ModuleNotFoundError):
        return False

def check_backend_status():
    """Print status of all backends."""
    print("Backend Status:")
    cgal_ok = _check_cgal()
    scipy_ok = _check_scipy()
    print(f"  CGAL: {'✓ Available' if cgal_ok else '✗ Not available'}")
    print(f"  SciPy+Numba: {'✓ Available' if scipy_ok else '✗ Not available'}")
    
    if not cgal_ok and not scipy_ok:
        print("\n⚠ Warning: No backends available!")
        print("   Install CGAL or SciPy+Numba to use pydive.")
    elif not cgal_ok:
        print("\nNote: Using SciPy+Numba backend (CGAL not available)")
    return cgal_ok or scipy_ok

def _get_best_backend():
    """Get the best available backend."""
    if _check_cgal():
        return 'cgal'
    elif _check_scipy():
        return 'scipy'
    else:
        raise ImportError(
            "No backends available. Install CGAL or SciPy+Numba.\n"
            "See INSTALL.md for installation instructions."
        )

def get_void_catalog(points, backend=None, **kwargs):
    """
    Get void catalog using specified or auto-selected backend.
    
    Parameters
    ----------
    points : ndarray (N, 3)
        Input point coordinates
    backend : str, optional
        Backend to use ('cgal' or 'scipy'). If None, auto-selects best available.
    **kwargs
        Additional arguments passed to backend function
        
    Returns
    -------
    output : ndarray (n_simplices, 4)
        Array with columns [x, y, z, r] for each simplex circumcenter
    """
    if backend is None:
        backend = _get_best_backend()
    
    if backend == 'cgal':
        from .delaunay_backend import get_void_catalog_cgal
        return get_void_catalog_cgal(points, **kwargs)
    elif backend == 'scipy':
        from .pydive_scipy import get_void_catalog_scipy
        return get_void_catalog_scipy(points, **kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'cgal' or 'scipy'.")

def get_void_catalog_full(points, backend=None, **kwargs):
    """
    Get full void catalog with DTFE, volumes, and areas.
    
    Parameters
    ----------
    points : ndarray (N, 3)
        Input point coordinates
    backend : str, optional
        Backend to use ('cgal' or 'scipy'). If None, auto-selects best available.
    **kwargs
        Additional arguments passed to backend function
        
    Returns
    -------
    output : ndarray (n_simplices, 7)
        Array with columns [x, y, z, r, volume, dtfe_interp, area]
    dtfe : ndarray (N,)
        DTFE density estimate at each input point
    """
    if backend is None:
        backend = _get_best_backend()
    
    if backend == 'cgal':
        from .delaunay_backend import get_void_catalog_full
        return get_void_catalog_full(points, **kwargs)
    elif backend == 'scipy':
        from .pydive_scipy import get_void_catalog_full_scipy
        return get_void_catalog_full_scipy(points, **kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'cgal' or 'scipy'.")

__all__ = [
    'get_void_catalog',
    'get_void_catalog_full',
    'check_backend_status'
]
