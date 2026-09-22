"""
Comprehensive backend comparison tests for pydive.

This module tests all available backends (CGAL, scipy+numba) and compares:
- Correctness: Results should match within numerical tolerance
- Performance: Timing comparisons for different dataset sizes
- Memory usage: Memory footprint comparisons
- Feature parity: All backends should support the same features

Usage:
    python -m pytest tests/test_backends.py -v
    python tests/test_backends.py  # Run as script
"""

import numpy as np
import sys
import time
import os
from typing import Tuple, Optional, Dict, Any
import warnings

# Test configuration
TEST_SIZES = [100, 500, 1000]  # Number of points for testing
RANDOM_SEED = 42
TOLERANCE_REL = 1e-10  # Relative tolerance for numerical comparisons
TOLERANCE_ABS = 1e-12  # Absolute tolerance for numerical comparisons


class BackendResult:
    """Container for backend test results."""
    
    def __init__(self, name: str):
        self.name = name
        self.available = False
        self.voids = None
        self.dtfe = None
        self.time_basic = None
        self.time_full = None
        self.n_simplices = None
        self.error = None
    
    def __repr__(self):
        status = "✓" if self.available else "✗"
        return f"{status} {self.name}: {self.n_simplices} simplices"


def generate_test_data(n_points: int, seed: int = RANDOM_SEED) -> np.ndarray:
    """Generate reproducible test point data."""
    np.random.seed(seed)
    # Generate points in a unit cube
    points = np.random.random((n_points, 3)).astype(np.double)
    # Scale to reasonable size
    points *= 100.0
    return points


def check_backend_availability() -> Dict[str, bool]:
    """Check which backends are available."""
    import sys
    import os
    # Add workspace to path for testing
    workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if workspace_dir not in sys.path:
        sys.path.insert(0, workspace_dir)
    
    availability = {}
    
    # Check CGAL backend
    try:
        from pydive.delaunay_backend import get_void_catalog_cgal
        availability['cgal'] = True
    except (ImportError, ModuleNotFoundError) as e:
        availability['cgal'] = False
        print(f"  CGAL backend not available: {e}")
    
    # Check scipy+numba backend
    try:
        from pydive.pydive_scipy import get_void_catalog_scipy
        availability['scipy'] = True
    except (ImportError, ModuleNotFoundError) as e:
        availability['scipy'] = False
        print(f"  SciPy backend not available: {e}")
    
    return availability


def test_backend_basic(backend_name: str, points: np.ndarray) -> BackendResult:
    """Test basic void catalog computation for a backend."""
    result = BackendResult(backend_name)
    
    try:
        if backend_name == 'cgal':
            from pydive.delaunay_backend import get_void_catalog_cgal
            start = time.perf_counter()
            voids = get_void_catalog_cgal(points)
            result.time_basic = time.perf_counter() - start
            result.voids = voids
            
        elif backend_name == 'scipy':
            from pydive.pydive_scipy import get_void_catalog_scipy
            start = time.perf_counter()
            voids = get_void_catalog_scipy(points)
            result.time_basic = time.perf_counter() - start
            result.voids = voids
            
        else:
            raise ValueError(f"Unknown backend: {backend_name}")
        
        result.available = True
        result.n_simplices = len(voids)
        
    except Exception as e:
        result.error = str(e)
        result.available = False
    
    return result


def test_backend_full(backend_name: str, points: np.ndarray) -> BackendResult:
    """Test full void catalog computation (with DTFE, volumes, areas)."""
    result = BackendResult(backend_name)
    
    try:
        if backend_name == 'cgal':
            from pydive.delaunay_backend import get_void_catalog_full
            start = time.perf_counter()
            voids, dtfe = get_void_catalog_full(points)
            result.time_full = time.perf_counter() - start
            result.voids = voids
            result.dtfe = dtfe
            
        elif backend_name == 'scipy':
            from pydive.pydive_scipy import get_void_catalog_full_scipy
            start = time.perf_counter()
            voids, dtfe = get_void_catalog_full_scipy(points)
            result.time_full = time.perf_counter() - start
            result.voids = voids
            result.dtfe = dtfe
            
        else:
            raise ValueError(f"Unknown backend: {backend_name}")
        
        result.available = True
        result.n_simplices = len(voids)
        
    except Exception as e:
        result.error = str(e)
        result.available = False
    
    return result


def compare_voids(voids1: np.ndarray, voids2: np.ndarray, 
                  name1: str = "Backend 1", name2: str = "Backend 2") -> bool:
    """Compare void catalogs from two backends."""
    if voids1 is None or voids2 is None:
        print(f"  ✗ Cannot compare: one or both void catalogs is None")
        return False
    
    if voids1.shape != voids2.shape:
        print(f"  ✗ Shape mismatch: {voids1.shape} vs {voids2.shape}")
        return False
    
    # Check if shapes match
    n_simplices = voids1.shape[0]
    print(f"  Comparing {n_simplices} simplices...")
    
    # Check each column with appropriate tolerances
    columns = ['x', 'y', 'z', 'radius']
    all_match = True
    
    for i, col_name in enumerate(columns):
        if i >= voids1.shape[1]:
            break
        
        col1 = voids1[:, i]
        col2 = voids2[:, i]
        
        # Use both relative and absolute tolerance
        match = np.allclose(col1, col2, rtol=TOLERANCE_REL, atol=TOLERANCE_ABS)
        
        if not match:
            all_match = False
            max_diff = np.max(np.abs(col1 - col2))
            mean_diff = np.mean(np.abs(col1 - col2))
            print(f"    ✗ {col_name}: max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}")
        else:
            print(f"    ✓ {col_name}: arrays match within tolerance")
    
    return all_match


def compare_dtfe(dtfe1: np.ndarray, dtfe2: np.ndarray,
                 name1: str = "Backend 1", name2: str = "Backend 2") -> bool:
    """Compare DTFE values from two backends."""
    if dtfe1 is None or dtfe2 is None:
        print(f"  ✗ Cannot compare DTFE: one or both is None")
        return False
    
    if dtfe1.shape != dtfe2.shape:
        print(f"  ✗ DTFE shape mismatch: {dtfe1.shape} vs {dtfe2.shape}")
        return False
    
    n_points = len(dtfe1)
    print(f"  Comparing DTFE for {n_points} points...")
    
    match = np.allclose(dtfe1, dtfe2, rtol=TOLERANCE_REL, atol=TOLERANCE_ABS)
    
    if not match:
        max_diff = np.max(np.abs(dtfe1 - dtfe2))
        mean_diff = np.mean(np.abs(dtfe1 - dtfe2))
        print(f"    ✗ DTFE: max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}")
    else:
        print(f"    ✓ DTFE: arrays match within tolerance")
    
    return match


def run_correctness_tests():
    """Run correctness comparison tests between backends."""
    print("\n" + "="*70)
    print("CORRECTNESS TESTS")
    print("="*70)
    
    availability = check_backend_availability()
    
    if not any(availability.values()):
        print("\n✗ No backends available for testing!")
        return False
    
    available_backends = [k for k, v in availability.items() if v]
    print(f"\nAvailable backends: {', '.join(available_backends)}")
    
    all_passed = True
    
    for n_points in TEST_SIZES:
        print(f"\n{'-'*70}")
        print(f"Testing with {n_points} points")
        print(f"{'-'*70}")
        
        points = generate_test_data(n_points)
        results = {}
        
        # Test basic functionality
        print("\nBasic void catalog:")
        for backend in available_backends:
            print(f"  Testing {backend}...", end=" ")
            results[backend] = test_backend_basic(backend, points)
            if results[backend].available:
                print(f"✓ ({results[backend].time_basic:.4f}s, {results[backend].n_simplices} simplices)")
            else:
                print(f"✗ {results[backend].error}")
        
        # Compare results if multiple backends available
        if len(available_backends) >= 2:
            print("\nComparing basic results:")
            compare_voids(
                results[available_backends[0]].voids,
                results[available_backends[1]].voids,
                available_backends[0],
                available_backends[1]
            )
        
        # Test full functionality
        print("\nFull void catalog (with DTFE, volumes, areas):")
        for backend in available_backends:
            print(f"  Testing {backend}...", end=" ")
            results[backend] = test_backend_full(backend, points)
            if results[backend].available:
                print(f"✓ ({results[backend].time_full:.4f}s, {results[backend].n_simplices} simplices)")
            else:
                print(f"✗ {results[backend].error}")
        
        # Compare full results
        if len(available_backends) >= 2:
            print("\nComparing full results:")
            voids_match = compare_voids(
                results[available_backends[0]].voids,
                results[available_backends[1]].voids,
                available_backends[0],
                available_backends[1]
            )
            
            dtfe_match = compare_dtfe(
                results[available_backends[0]].dtfe,
                results[available_backends[1]].dtfe,
                available_backends[0],
                available_backends[1]
            )
            
            if voids_match and dtfe_match:
                print(f"\n  ✓ All comparisons passed for {n_points} points")
            else:
                print(f"\n  ✗ Some comparisons failed for {n_points} points")
                all_passed = False
    
    return all_passed


def run_performance_tests():
    """Run performance benchmark tests."""
    print("\n" + "="*70)
    print("PERFORMANCE BENCHMARKS")
    print("="*70)
    
    availability = check_backend_availability()
    available_backends = [k for k, v in availability.items() if v]
    
    if len(available_backends) == 0:
        print("\n✗ No backends available for benchmarking!")
        return
    
    # Larger test sizes for performance
    perf_sizes = [500, 1000, 2000, 5000]
    
    print(f"\nBenchmarking backends: {', '.join(available_backends)}")
    print(f"Test sizes: {perf_sizes}")
    
    results = {backend: {'basic': [], 'full': []} for backend in available_backends}
    
    for n_points in perf_sizes:
        print(f"\n{'-'*70}")
        print(f"Benchmarking with {n_points} points")
        print(f"{'-'*70}")
        
        points = generate_test_data(n_points)
        
        for backend in available_backends:
            # Basic benchmark
            try:
                if backend == 'cgal':
                    from pydive.delaunay_backend import get_void_catalog_cgal
                    start = time.perf_counter()
                    voids = get_void_catalog_cgal(points)
                    elapsed = time.perf_counter() - start
                    results[backend]['basic'].append(elapsed)
                    print(f"  {backend:8s} basic: {elapsed:8.4f}s ({len(voids)} simplices)")
                    
                elif backend == 'scipy':
                    from pydive.pydive_scipy import get_void_catalog_scipy
                    start = time.perf_counter()
                    voids = get_void_catalog_scipy(points)
                    elapsed = time.perf_counter() - start
                    results[backend]['basic'].append(elapsed)
                    print(f"  {backend:8s} basic: {elapsed:8.4f}s ({len(voids)} simplices)")
            except Exception as e:
                print(f"  {backend:8s} basic: FAILED - {e}")
            
            # Full benchmark
            try:
                if backend == 'cgal':
                    from pydive.delaunay_backend import get_void_catalog_full
                    start = time.perf_counter()
                    voids, dtfe = get_void_catalog_full(points)
                    elapsed = time.perf_counter() - start
                    results[backend]['full'].append(elapsed)
                    print(f"  {backend:8s} full:  {elapsed:8.4f}s")
                    
                elif backend == 'scipy':
                    from pydive.pydive_scipy import get_void_catalog_full_scipy
                    start = time.perf_counter()
                    voids, dtfe = get_void_catalog_full_scipy(points)
                    elapsed = time.perf_counter() - start
                    results[backend]['full'].append(elapsed)
                    print(f"  {backend:8s} full:  {elapsed:8.4f}s")
            except Exception as e:
                print(f"  {backend:8s} full:  FAILED - {e}")
    
    # Print summary
    print(f"\n{'='*70}")
    print("PERFORMANCE SUMMARY")
    print(f"{'='*70}")
    
    for backend in available_backends:
        if results[backend]['basic']:
            avg_basic = np.mean(results[backend]['basic'])
            avg_full = np.mean(results[backend]['full']) if results[backend]['full'] else None
            print(f"\n{backend.upper()}:")
            print(f"  Average basic: {avg_basic:.4f}s")
            if avg_full:
                print(f"  Average full:  {avg_full:.4f}s")


def run_memory_tests():
    """Run memory usage tests (if psutil available)."""
    try:
        import psutil
        import os
    except ImportError:
        print("\nSkipping memory tests (psutil not available)")
        print("Install with: pip install psutil")
        return
    
    print("\n" + "="*70)
    print("MEMORY USAGE TESTS")
    print("="*70)
    
    availability = check_backend_availability()
    available_backends = [k for k, v in availability.items() if v]
    
    if len(available_backends) == 0:
        print("\n✗ No backends available for memory testing!")
        return
    
    n_points = 5000
    points = generate_test_data(n_points)
    
    process = psutil.Process(os.getpid())
    
    print(f"\nTesting memory with {n_points} points")
    print(f"{'-'*70}")
    
    for backend in available_backends:
        # Get initial memory
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        
        try:
            if backend == 'cgal':
                from pydive.delaunay_backend import get_void_catalog_full
                voids, dtfe = get_void_catalog_full(points)
                
            elif backend == 'scipy':
                from pydive.pydive_scipy import get_void_catalog_full_scipy
                voids, dtfe = get_void_catalog_full_scipy(points)
            
            mem_after = process.memory_info().rss / 1024 / 1024  # MB
            mem_used = mem_after - mem_before
            
            print(f"  {backend:8s}: {mem_used:8.2f} MB (total: {mem_after:.2f} MB)")
            
        except Exception as e:
            print(f"  {backend:8s}: FAILED - {e}")


def main():
    """Run all tests."""
    print("="*70)
    print("PYDIVE BACKEND COMPARISON TESTS")
    print("="*70)
    print(f"Test sizes: {TEST_SIZES}")
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Tolerance: rel={TOLERANCE_REL}, abs={TOLERANCE_ABS}")
    
    # Run all test suites
    correctness_passed = run_correctness_tests()
    run_performance_tests()
    run_memory_tests()
    
    # Final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    
    if correctness_passed:
        print("✓ All correctness tests PASSED")
    else:
        print("✗ Some correctness tests FAILED")
    
    print("\nNote: Performance differences are expected.")
    print("      CGAL is typically faster but harder to install.")
    print("      SciPy+Numba is easier to install with good performance.")
    
    return 0 if correctness_passed else 1


if __name__ == '__main__':
    sys.exit(main())
