#!/usr/bin/env python
"""
Example: Backend performance comparison.

This script benchmarks the performance of different backends (CGAL vs SciPy+Numba)
across various point counts to help users understand the trade-offs.
"""

import numpy as np
import time
import sys

# Add parent directory to path for development
sys.path.insert(0, '..')

from pydive import get_void_catalog, get_void_catalog_full, check_backend_status


def benchmark_backend(backend_name, points, full=False):
    """
    Benchmark a specific backend.
    
    Parameters
    ----------
    backend_name : str
        Name of backend ('cgal' or 'scipy')
    points : ndarray
        Input points
    full : bool
        Whether to run full catalog computation
        
    Returns
    -------
    elapsed : float
        Time in seconds
    n_voids : int
        Number of voids found
    """
    start = time.perf_counter()
    
    if full:
        result = get_void_catalog_full(points, backend=backend_name)
        voids = result[0]
    else:
        voids = get_void_catalog(points, backend=backend_name)
    
    elapsed = time.perf_counter() - start
    
    return elapsed, len(voids)


def run_benchmark(point_counts, n_runs=3, full=False):
    """
    Run benchmarks across multiple point counts.
    
    Parameters
    ----------
    point_counts : list
        List of point counts to test
    n_runs : int
        Number of runs per configuration (for averaging)
    full : bool
        Whether to run full catalog computation
    """
    print("=" * 70)
    print("PyDIVE - Backend Performance Benchmark")
    print("=" * 70)
    
    # Check available backends
    print("\nBackend availability:")
    check_backend_status()
    
    scipy_available = False
    cgal_available = False
    
    try:
        test_points = np.random.random((100, 3))
        get_void_catalog(test_points, backend='scipy')
        scipy_available = True
    except:
        pass
    
    try:
        test_points = np.random.random((100, 3))
        get_void_catalog(test_points, backend='cgal')
        cgal_available = True
    except:
        pass
    
    if not scipy_available and not cgal_available:
        print("\n✗ No backends available! Cannot run benchmark.")
        return
    
    # Run benchmarks
    results = {}
    
    for n_points in point_counts:
        print(f"\n{'-'*70}")
        print(f"Benchmarking with {n_points:,} points...")
        print(f"{'-'*70}")
        
        # Generate random points
        np.random.seed(42)
        points = np.random.random((n_points, 3)) * 100.0
        
        results[n_points] = {}
        
        # Test SciPy backend
        if scipy_available:
            times = []
            n_voids_list = []
            for run in range(n_runs):
                np.random.seed(run)
                pts = np.random.random((n_points, 3)) * 100.0
                elapsed, n_voids = benchmark_backend('scipy', pts, full=full)
                times.append(elapsed)
                n_voids_list.append(n_voids)
            
            avg_time = np.mean(times)
            std_time = np.std(times)
            results[n_points]['scipy'] = {
                'time': avg_time,
                'std': std_time,
                'n_voids': n_voids_list[0]
            }
            print(f"  SciPy:      {avg_time:8.4f} ± {std_time:.4f} s  ({n_voids_list[0]:,} voids)")
        
        # Test CGAL backend
        if cgal_available:
            times = []
            n_voids_list = []
            for run in range(n_runs):
                np.random.seed(run)
                pts = np.random.random((n_points, 3)) * 100.0
                elapsed, n_voids = benchmark_backend('cgal', pts, full=full)
                times.append(elapsed)
                n_voids_list.append(n_voids)
            
            avg_time = np.mean(times)
            std_time = np.std(times)
            results[n_points]['cgal'] = {
                'time': avg_time,
                'std': std_time,
                'n_voids': n_voids_list[0]
            }
            print(f"  CGAL:       {avg_time:8.4f} ± {std_time:.4f} s  ({n_voids_list[0]:,} voids)")
            
            # Calculate speedup
            if scipy_available:
                speedup = results[n_points]['scipy']['time'] / avg_time
                print(f"  Speedup:    {speedup:8.2f}x faster than SciPy")
    
    # Print summary table
    print(f"\n{'='*70}")
    print("SUMMARY TABLE")
    print(f"{'='*70}")
    
    if full:
        print(f"{'Points':>10} | {'SciPy (s)':>12} | {'CGAL (s)':>12} | {'Speedup':>10}")
    else:
        print(f"{'Points':>10} | {'SciPy (s)':>12} | {'CGAL (s)':>12} | {'Speedup':>10}")
    print(f"{'-'*10}-+-{'-'*12}-+-{'-'*12}-+-{'-'*10}")
    
    for n_points in point_counts:
        row = f"{n_points:>10,} | "
        
        if scipy_available and n_points in results and 'scipy' in results[n_points]:
            row += f"{results[n_points]['scipy']['time']:>12.4f} | "
        else:
            row += f"{'N/A':>12} | "
        
        if cgal_available and n_points in results and 'cgal' in results[n_points]:
            row += f"{results[n_points]['cgal']['time']:>12.4f} | "
            if scipy_available and n_points in results and 'scipy' in results[n_points]:
                speedup = results[n_points]['scipy']['time'] / results[n_points]['cgal']['time']
                row += f"{speedup:>10.2f}x"
        else:
            row += f"{'N/A':>12} | {'N/A':>10}"
        
        print(row)
    
    print(f"{'='*70}\n")
    
    return results


def plot_results(results, save_path='benchmark_comparison.png'):
    """Create a plot comparing backend performance."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib not available, skipping plot generation.")
        return
    
    point_counts = sorted(results.keys())
    
    # Only plot if we have data for at least one backend
    scipy_counts = [n for n in point_counts if 'scipy' in results[n]]
    cgal_counts = [n for n in point_counts if 'cgal' in results[n]]
    
    if not scipy_counts and not cgal_counts:
        print("No benchmark data available for plotting.")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Use the union of point counts that have data
    all_counts = sorted(set(scipy_counts + cgal_counts))
    
    x = np.arange(len(all_counts))
    width = 0.35
    
    scipy_times = [results[n]['scipy']['time'] for n in all_counts if 'scipy' in results[n]]
    cgal_times = [results[n]['cgal']['time'] for n in all_counts if 'cgal' in results[n]]
    
    if scipy_times and cgal_times:
        # Both backends available
        ax.bar(x - width/2, scipy_times, width, label='SciPy+Numba', alpha=0.8)
        ax.bar(x + width/2, cgal_times, width, label='CGAL', alpha=0.8)
    elif scipy_times:
        # Only SciPy available
        ax.bar(x, scipy_times, width, label='SciPy+Numba', alpha=0.8)
    elif cgal_times:
        # Only CGAL available
        ax.bar(x, cgal_times, width, label='CGAL', alpha=0.8)
    
    ax.set_xlabel('Number of Points', fontsize=12)
    ax.set_ylabel('Time (seconds)', fontsize=12)
    ax.set_title('Backend Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{n:,}' for n in all_counts], rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved benchmark comparison plot to {save_path}")
    plt.show()


def main():
    print("\nSelect benchmark mode:")
    print("  1. Quick benchmark (small datasets)")
    print("  2. Full benchmark (larger datasets)")
    print("  3. Custom benchmark")
    
    choice = input("\nEnter choice (1-3) [default: 1]: ").strip() or '1'
    
    if choice == '1':
        point_counts = [500, 1000, 2000]
        full = False
    elif choice == '2':
        point_counts = [1000, 2000, 5000, 10000]
        full_mode = input("Run full catalog (with DTFE, volumes, areas)? (y/n) [default: n]: ").strip().lower()
        full = (full_mode == 'y')
    else:
        custom = input("Enter point counts (comma-separated, e.g., 500,1000,2000): ").strip()
        point_counts = [int(x.strip()) for x in custom.split(',')]
        full_mode = input("Run full catalog? (y/n) [default: n]: ").strip().lower()
        full = (full_mode == 'y')
    
    mode_str = "full catalog" if full else "basic catalog"
    print(f"\nRunning {mode_str} benchmark with point counts: {point_counts}")
    
    results = run_benchmark(point_counts, n_runs=3, full=full)
    
    if results:
        # Check if running interactively
        import sys
        if sys.stdin.isatty():
            plot_choice = input("\nGenerate comparison plot? (y/n) [default: y]: ").strip().lower()
            if plot_choice != 'n':
                try:
                    plot_results(results)
                except Exception as e:
                    print(f"Could not generate plot: {e}")
        else:
            # Non-interactive mode, skip plot prompt
            print("\nSkipping plot generation (non-interactive mode)")
            try:
                plot_results(results)
            except Exception as e:
                print(f"Could not generate plot: {e}")
    
    print("\nBenchmark completed!")


if __name__ == '__main__':
    main()
