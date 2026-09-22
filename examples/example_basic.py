#!/usr/bin/env python
"""
Example: Basic void catalog computation with different backends.

This script demonstrates how to use pydive with automatic backend selection
and explicit backend specification.
"""

import numpy as np
import sys

# Add parent directory to path for development
sys.path.insert(0, '..')

from pydive import get_void_catalog, check_backend_status


def main():
    print("=" * 60)
    print("PyDIVE - Basic Void Catalog Example")
    print("=" * 60)
    
    # Check available backends
    print("\nBackend availability:")
    check_backend_status()
    
    # Generate sample point distribution
    print("\nGenerating random point distribution...")
    np.random.seed(42)
    n_points = 2000
    points = np.random.random((n_points, 3)) * 100.0
    
    print(f"  Number of points: {n_points}")
    print(f"  Box size: 100 x 100 x 100")
    
    # Method 1: Auto-select best available backend
    print("\n" + "-" * 60)
    print("Method 1: Auto-select backend")
    print("-" * 60)
    try:
        voids_auto = get_void_catalog(points)
        print(f"✓ Successfully computed void catalog")
        print(f"  Number of voids (tetrahedra): {len(voids_auto)}")
        print(f"  Output shape: {voids_auto.shape}")
        print(f"  Columns: [x, y, z, radius]")
        print(f"\n  Sample void centers and radii:")
        for i in range(min(5, len(voids_auto))):
            x, y, z, r = voids_auto[i]
            print(f"    Void {i}: center=({x:6.2f}, {y:6.2f}, {z:6.2f}), radius={r:6.2f}")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Method 2: Explicitly use SciPy backend
    print("\n" + "-" * 60)
    print("Method 2: Explicit SciPy backend")
    print("-" * 60)
    try:
        voids_scipy = get_void_catalog(points, backend='scipy')
        print(f"✓ Successfully computed void catalog with SciPy")
        print(f"  Number of voids: {len(voids_scipy)}")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Method 3: Try CGAL backend (may not be available)
    print("\n" + "-" * 60)
    print("Method 3: Explicit CGAL backend")
    print("-" * 60)
    try:
        voids_cgal = get_void_catalog(points, backend='cgal')
        print(f"✓ Successfully computed void catalog with CGAL")
        print(f"  Number of voids: {len(voids_cgal)}")
        
        # Compare results if both backends worked
        if 'voids_scipy' in locals():
            diff = np.abs(voids_scipy - voids_cgal).max()
            print(f"\n  Comparison with SciPy backend:")
            print(f"    Max absolute difference: {diff:.2e}")
            if diff < 1e-10:
                print(f"    ✓ Results match within tolerance!")
    except ImportError as e:
        print(f"⚠ CGAL backend not available: {e}")
        print(f"   Install CGAL library to use this backend.")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()
