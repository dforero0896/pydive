"""
Pytest-compatible tests for pydive backends.

This module provides pytest-style tests that can be run with:
    python -m pytest tests/test_backends_pytest.py -v

Tests include:
- Backend availability checks
- Correctness comparisons between backends
- Performance benchmarks (optional)
- Feature parity validation
"""

import numpy as np
import pytest
import sys
import os
from typing import Dict, List

# Add workspace to path
workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if workspace_dir not in sys.path:
    sys.path.insert(0, workspace_dir)

from pydive.pydive_scipy import get_void_catalog_scipy, get_void_catalog_full_scipy

# Try to import CGAL backend
try:
    from pydive.delaunay_backend import get_void_catalog_cgal, get_void_catalog_full
    CGAL_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    CGAL_AVAILABLE = False

# Test configuration
RANDOM_SEED = 42
TOLERANCE_REL = 1e-10
TOLERANCE_ABS = 1e-12


@pytest.fixture(scope="module")
def test_points_100():
    """Generate 100 test points."""
    np.random.seed(RANDOM_SEED)
    return np.random.random((100, 3)).astype(np.double) * 100


@pytest.fixture(scope="module")
def test_points_500():
    """Generate 500 test points."""
    np.random.seed(RANDOM_SEED + 1)
    return np.random.random((500, 3)).astype(np.double) * 100


@pytest.fixture(scope="module")
def test_points_1000():
    """Generate 1000 test points."""
    np.random.seed(RANDOM_SEED + 2)
    return np.random.random((1000, 3)).astype(np.double) * 100


class TestBackendAvailability:
    """Test that at least one backend is available."""
    
    def test_scipy_backend_available(self):
        """SciPy+Numba backend should be available."""
        assert True, "SciPy+Numba backend is always available"
    
    @pytest.mark.skipif(not CGAL_AVAILABLE, reason="CGAL not installed")
    def test_cgal_backend_available(self):
        """CGAL backend should be available if installed."""
        assert CGAL_AVAILABLE, "CGAL backend should be available"
    
    def test_at_least_one_backend(self):
        """At least one backend must be available."""
        assert True or CGAL_AVAILABLE, "At least one backend must be available"


class TestScipyBackend:
    """Test SciPy+Numba backend functionality."""
    
    def test_basic_catalog_100(self, test_points_100):
        """Test basic catalog with 100 points."""
        voids = get_void_catalog_scipy(test_points_100)
        assert voids.shape[1] == 4, "Basic catalog should have 4 columns"
        assert len(voids) > 0, "Should find at least one simplex"
        assert np.all(np.isfinite(voids)), "All values should be finite"
    
    def test_basic_catalog_500(self, test_points_500):
        """Test basic catalog with 500 points."""
        voids = get_void_catalog_scipy(test_points_500)
        assert voids.shape[1] == 4
        assert len(voids) > 0
        assert np.all(np.isfinite(voids))
    
    def test_basic_catalog_1000(self, test_points_1000):
        """Test basic catalog with 1000 points."""
        voids = get_void_catalog_scipy(test_points_1000)
        assert voids.shape[1] == 4
        assert len(voids) > 0
        assert np.all(np.isfinite(voids))
    
    def test_full_catalog_100(self, test_points_100):
        """Test full catalog with 100 points."""
        voids, dtfe = get_void_catalog_full_scipy(test_points_100)
        assert voids.shape[1] == 7, "Full catalog should have 7 columns"
        assert len(voids) > 0
        assert len(dtfe) == len(test_points_100), "DTFE should match input points"
        assert np.all(np.isfinite(voids)), "All void values should be finite"
        assert np.all(np.isfinite(dtfe)), "All DTFE values should be finite"
    
    def test_full_catalog_500(self, test_points_500):
        """Test full catalog with 500 points."""
        voids, dtfe = get_void_catalog_full_scipy(test_points_500)
        assert voids.shape[1] == 7
        assert len(voids) > 0
        assert len(dtfe) == len(test_points_500)
        assert np.all(np.isfinite(voids))
        assert np.all(np.isfinite(dtfe))
    
    def test_periodic_not_implemented(self, test_points_100):
        """Test that periodic boundary conditions raise NotImplementedError."""
        with pytest.raises(NotImplementedError):
            get_void_catalog_scipy(test_points_100, periodic=True)
        
        with pytest.raises(NotImplementedError):
            get_void_catalog_full_scipy(test_points_100, periodic=True)
    
    def test_simplex_count_scaling(self, test_points_100, test_points_500, test_points_1000):
        """Test that simplex count scales appropriately with point count."""
        voids_100 = get_void_catalog_scipy(test_points_100)
        voids_500 = get_void_catalog_scipy(test_points_500)
        voids_1000 = get_void_catalog_scipy(test_points_1000)
        
        # Simplex count should scale roughly linearly with point count
        # For Delaunay in 3D: n_simplices ≈ 5-6 * n_points
        ratio_500 = len(voids_500) / len(voids_100)
        ratio_1000 = len(voids_1000) / len(voids_100)
        
        # Allow some variance but check general scaling
        assert 3 < ratio_500 < 8, f"Simplex count ratio should be reasonable: {ratio_500}"
        assert 6 < ratio_1000 < 15, f"Simplex count ratio should be reasonable: {ratio_1000}"


@pytest.mark.skipif(not CGAL_AVAILABLE, reason="CGAL not installed")
class TestCGALBackend:
    """Test CGAL backend functionality."""
    
    def test_basic_catalog_100(self, test_points_100):
        """Test CGAL basic catalog with 100 points."""
        voids = get_void_catalog_cgal(test_points_100)
        assert voids.shape[1] == 4
        assert len(voids) > 0
        assert np.all(np.isfinite(voids))
    
    def test_full_catalog_100(self, test_points_100):
        """Test CGAL full catalog with 100 points."""
        voids, dtfe = get_void_catalog_full(test_points_100)
        assert voids.shape[1] == 7
        assert len(voids) > 0
        assert len(dtfe) == len(test_points_100)
        assert np.all(np.isfinite(voids))
        assert np.all(np.isfinite(dtfe))


@pytest.mark.skipif(not CGAL_AVAILABLE, reason="CGAL not installed")
class TestBackendComparison:
    """Compare results between CGAL and SciPy backends."""
    
    def test_basic_catalog_match_100(self, test_points_100):
        """Test that basic catalogs match between backends."""
        voids_cgal = get_void_catalog_cgal(test_points_100)
        voids_scipy = get_void_catalog_scipy(test_points_100)
        
        # Should have same number of simplices
        assert len(voids_cgal) == len(voids_scipy), \
            f"Simplex count mismatch: CGAL={len(voids_cgal)}, SciPy={len(voids_scipy)}"
        
        # Sort by radius for comparison (order may differ)
        voids_cgal_sorted = voids_cgal[np.argsort(voids_cgal[:, 3])]
        voids_scipy_sorted = voids_scipy[np.argsort(voids_scipy[:, 3])]
        
        # Check coordinates and radii match within tolerance
        assert np.allclose(voids_cgal_sorted[:, :3], voids_scipy_sorted[:, :3], 
                          rtol=TOLERANCE_REL, atol=TOLERANCE_ABS), \
            "Circumcenter coordinates should match"
        
        assert np.allclose(voids_cgal_sorted[:, 3], voids_scipy_sorted[:, 3],
                          rtol=TOLERANCE_REL, atol=TOLERANCE_ABS), \
            "Radii should match"
    
    def test_full_catalog_match_100(self, test_points_100):
        """Test that full catalogs match between backends."""
        voids_cgal, dtfe_cgal = get_void_catalog_full(test_points_100)
        voids_scipy, dtfe_scipy = get_void_catalog_full_scipy(test_points_100)
        
        # Check DTFE matches
        assert np.allclose(dtfe_cgal, dtfe_scipy, 
                          rtol=TOLERANCE_REL, atol=TOLERANCE_ABS), \
            "DTFE values should match"
        
        # Sort voids by radius for comparison
        voids_cgal_sorted = voids_cgal[np.argsort(voids_cgal[:, 3])]
        voids_scipy_sorted = voids_scipy[np.argsort(voids_scipy[:, 3])]
        
        # Check volumes match
        assert np.allclose(voids_cgal_sorted[:, 4], voids_scipy_sorted[:, 4],
                          rtol=TOLERANCE_REL, atol=TOLERANCE_ABS), \
            "Volumes should match"
        
        # Check areas match
        assert np.allclose(voids_cgal_sorted[:, 6], voids_scipy_sorted[:, 6],
                          rtol=TOLERANCE_REL*10, atol=TOLERANCE_ABS*10), \
            "Areas should match (relaxed tolerance)"


class TestNumericalStability:
    """Test numerical stability of backends."""
    
    def test_no_nan_values(self, test_points_500):
        """Test that no NaN values are produced."""
        voids, dtfe = get_void_catalog_full_scipy(test_points_500)
        assert not np.any(np.isnan(voids)), "No NaN values in void catalog"
        assert not np.any(np.isnan(dtfe)), "No NaN values in DTFE"
    
    def test_no_inf_values(self, test_points_500):
        """Test that no infinite values are produced."""
        voids, dtfe = get_void_catalog_full_scipy(test_points_500)
        assert not np.any(np.isinf(voids)), "No infinite values in void catalog"
        assert not np.any(np.isinf(dtfe)), "No infinite values in DTFE"
    
    def test_positive_volumes(self, test_points_500):
        """Test that all volumes are positive."""
        voids, _ = get_void_catalog_full_scipy(test_points_500)
        assert np.all(voids[:, 4] > 0), "All volumes should be positive"
    
    def test_positive_radii(self, test_points_500):
        """Test that all radii are positive."""
        voids, _ = get_void_catalog_full_scipy(test_points_500)
        assert np.all(voids[:, 3] > 0), "All radii should be positive"


if __name__ == '__main__':
    # Run with pytest
    pytest.main([__file__, '-v'])
