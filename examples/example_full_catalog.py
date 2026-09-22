#!/usr/bin/env python
"""
Example: Full void catalog with DTFE, volumes, and areas.

This script demonstrates how to compute the full void catalog including:
- Circumcenters and radii
- Tetrahedron volumes
- DTFE density estimates
- Surface areas
"""

import numpy as np
import sys

# Add parent directory to path for development
sys.path.insert(0, '..')

from pydive import get_void_catalog_full, check_backend_status


def main():
    print("=" * 60)
    print("PyDIVE - Full Void Catalog Example")
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
    
    # Compute full catalog with auto-selected backend
    print("\n" + "-" * 60)
    print("Computing full void catalog...")
    print("-" * 60)
    try:
        voids_full, dtfe = get_void_catalog_full(points)
        
        print(f"✓ Successfully computed full void catalog")
        print(f"  Number of voids (tetrahedra): {len(voids_full)}")
        print(f"  Output shape: {voids_full.shape}")
        print(f"  Columns: [x, y, z, radius, volume, dtfe_interp, area]")
        print(f"  DTFE shape: {dtfe.shape}")
        
        # Display statistics
        print(f"\n  Void properties statistics:")
        print(f"    Radius:   min={voids_full[:,3].min():8.3f}, max={voids_full[:,3].max():8.3f}, mean={voids_full[:,3].mean():8.3f}")
        print(f"    Volume:   min={voids_full[:,4].min():8.3f}, max={voids_full[:,4].max():8.3f}, mean={voids_full[:,4].mean():8.3f}")
        print(f"    Area:     min={voids_full[:,6].min():8.3f}, max={voids_full[:,6].max():8.3f}, mean={voids_full[:,6].mean():8.3f}")
        print(f"    DTFE:     min={voids_full[:,5].min():8.3f}, max={voids_full[:,5].max():8.3f}, mean={voids_full[:,5].mean():8.3f}")
        
        # Display sample voids
        print(f"\n  Sample voids (first 5):")
        print(f"    {'ID':>4} | {'X':>8} {'Y':>8} {'Z':>8} | {'R':>7} | {'Vol':>8} | {'DTFE':>7} | {'Area':>8}")
        print(f"    {'-'*4}-+-{'-'*8} {'-'*8} {'-'*8}-+-{'-'*7}-+-{'-'*8}-+-{'-'*7}-+-{'-'*8}")
        for i in range(min(5, len(voids_full))):
            x, y, z, r, vol, dtfe_val, area = voids_full[i]
            print(f"    {i:4d} | {x:8.2f} {y:8.2f} {z:8.2f} | {r:7.3f} | {vol:8.3f} | {dtfe_val:7.3f} | {area:8.3f}")
        
        # DTFE at original points
        print(f"\n  DTFE density at input points:")
        print(f"    min={dtfe.min():8.3f}, max={dtfe.max():8.3f}, mean={dtfe.mean():8.3f}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Compare backends if both available
    print("\n" + "-" * 60)
    print("Comparing backends (if both available)...")
    print("-" * 60)
    
    scipy_ok = False
    cgal_ok = False
    
    try:
        voids_scipy, dtfe_scipy = get_void_catalog_full(points, backend='scipy')
        scipy_ok = True
        print("✓ SciPy backend works")
    except Exception as e:
        print(f"✗ SciPy backend error: {e}")
    
    try:
        voids_cgal, dtfe_cgal = get_void_catalog_full(points, backend='cgal')
        cgal_ok = True
        print("✓ CGAL backend works")
    except ImportError:
        print("⚠ CGAL backend not available")
    except Exception as e:
        print(f"✗ CGAL backend error: {e}")
    
    if scipy_ok and cgal_ok:
        print(f"\n  Backend comparison:")
        diff_centers = np.abs(voids_scipy[:, :3] - voids_cgal[:, :3]).max()
        diff_radii = np.abs(voids_scipy[:, 3] - voids_cgal[:, 3]).max()
        diff_volumes = np.abs(voids_scipy[:, 4] - voids_cgal[:, 4]).max()
        diff_areas = np.abs(voids_scipy[:, 6] - voids_cgal[:, 6]).max()
        diff_dtfe = np.abs(dtfe_scipy - dtfe_cgal).max()
        
        print(f"    Center difference:   {diff_centers:.2e}")
        print(f"    Radius difference:   {diff_radii:.2e}")
        print(f"    Volume difference:   {diff_volumes:.2e}")
        print(f"    Area difference:     {diff_areas:.2e}")
        print(f"    DTFE difference:     {diff_dtfe:.2e}")
        
        tolerance = 1e-8
        if all(d < tolerance for d in [diff_centers, diff_radii, diff_volumes, diff_areas, diff_dtfe]):
            print(f"\n    ✓ All results match within tolerance ({tolerance})!")
        else:
            print(f"\n    ⚠ Some differences exceed tolerance")
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()
