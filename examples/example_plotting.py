#!/usr/bin/env python
"""
Example: Plotting void statistics distributions.

This script demonstrates how to create publication-quality plots of void
statistics including radius distribution, volume distribution, and DTFE
density estimates.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys

# Add parent directory to path for development
sys.path.insert(0, '..')

from pydive import get_void_catalog_full, check_backend_status


def plot_radius_distribution_filtered(voids, save_path=None, percentile=99):
    """Plot the distribution of void radii filtered to a given percentile."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    radii = voids[:, 3]
    r_max = np.percentile(radii, percentile)
    radii_filtered = radii[radii <= r_max]
    
    # Histogram
    ax1 = axes[0]
    n, bins, patches = ax1.hist(radii_filtered, bins=50, alpha=0.7, color='steelblue', 
                                 edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('Void Radius', fontsize=12)
    ax1.set_ylabel('Number of Voids', fontsize=12)
    ax1.set_title(f'Distribution of Void Radii (filtered ≤ {percentile}th pct)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add mean and median lines
    mean_r = radii_filtered.mean()
    median_r = np.median(radii_filtered)
    ax1.axvline(mean_r, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_r:.2f}')
    ax1.axvline(median_r, color='green', linestyle=':', linewidth=2, label=f'Median: {median_r:.2f}')
    ax1.legend()
    
    # Add stats text box
    stats_text = f"Count: {len(radii_filtered):,}\n"
    stats_text += f"Min: {radii_filtered.min():.2f}\n"
    stats_text += f"Max ({percentile}th): {r_max:.2f}\n"
    stats_text += f"Mean: {mean_r:.2f}\n"
    stats_text += f"Median: {median_r:.2f}"
    ax1.text(0.98, 0.95, stats_text, transform=ax1.transAxes, 
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=10)
    
    # Cumulative distribution
    ax2 = axes[1]
    sorted_radii = np.sort(radii_filtered)
    cumulative = np.arange(1, len(sorted_radii) + 1) / len(sorted_radii)
    ax2.plot(sorted_radii, cumulative, linewidth=2, color='darkblue')
    ax2.set_xlabel('Void Radius', fontsize=12)
    ax2.set_ylabel('Cumulative Fraction', fontsize=12)
    ax2.set_title(f'Cumulative Distribution (filtered ≤ {percentile}th pct)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.05)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved radius distribution plot to {save_path}")
    
    plt.show()


def plot_volume_distribution_filtered(voids, save_path=None, percentile=99):
    """Plot the distribution of void volumes filtered to a given percentile."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    volumes = voids[:, 4]
    valid_vols = volumes[volumes > 0]
    v_max = np.percentile(valid_vols, percentile)
    vols_filtered = valid_vols[valid_vols <= v_max]
    
    # Histogram with log scale
    ax1 = axes[0]
    n, bins, patches = ax1.hist(vols_filtered, bins=50, alpha=0.7, color='coral',
                                 edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('Void Volume', fontsize=12)
    ax1.set_ylabel('Number of Voids', fontsize=12)
    ax1.set_title(f'Distribution of Void Volumes (filtered ≤ {percentile}th pct)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add statistics
    mean_v = vols_filtered.mean()
    median_v = np.median(vols_filtered)
    ax1.axvline(mean_v, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_v:.2f}')
    ax1.axvline(median_v, color='green', linestyle=':', linewidth=2, label=f'Median: {median_v:.2f}')
    ax1.legend()
    
    # Add stats text box
    stats_text = f"Count: {len(vols_filtered):,}\n"
    stats_text += f"Min: {vols_filtered.min():.2f}\n"
    stats_text += f"Max ({percentile}th): {v_max:.2f}\n"
    stats_text += f"Mean: {mean_v:.2f}\n"
    stats_text += f"Median: {median_v:.2f}"
    ax1.text(0.98, 0.95, stats_text, transform=ax1.transAxes, 
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=10)
    
    # Log-scale histogram
    ax2 = axes[1]
    ax2.hist(np.log10(vols_filtered), bins=50, alpha=0.7, color='teal',
             edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('log₁₀(Volume)', fontsize=12)
    ax2.set_ylabel('Number of Voids', fontsize=12)
    ax2.set_title(f'Distribution of Void Volumes (Log Scale, filtered)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved volume distribution plot to {save_path}")
    
    plt.show()


def plot_area_vs_radius_filtered(voids, save_path=None, percentile=99):
    """Plot surface area vs radius relationship filtered to a given percentile."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    radii = voids[:, 3]
    areas = voids[:, 6]
    
    r_max = np.percentile(radii, percentile)
    mask = radii <= r_max
    radii_filtered = radii[mask]
    areas_filtered = areas[mask]
    
    # Scatter plot with transparency
    ax.scatter(radii_filtered, areas_filtered, alpha=0.3, s=10, c='navy', edgecolors='none')
    
    # Add theoretical sphere surface area curve (4πr²)
    r_theory = np.linspace(radii_filtered.min(), radii_filtered.max(), 100)
    area_sphere = 4 * np.pi * r_theory**2
    ax.plot(r_theory, area_sphere, 'r--', linewidth=2, label='Sphere: $4\\pi r^2$')
    
    ax.set_xlabel('Void Radius', fontsize=12)
    ax.set_ylabel('Surface Area', fontsize=12)
    ax.set_title(f'Surface Area vs Radius (filtered ≤ {percentile}th pct)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved area vs radius plot to {save_path}")
    
    plt.show()


def plot_all_statistics_filtered(voids, dtfe, save_dir='.', show=True, percentile=99):
    """Create a comprehensive figure with all statistics filtered to a given percentile."""
    fig = plt.figure(figsize=(14, 10))
    
    radii = voids[:, 3]
    volumes = voids[:, 4]
    dtfe_interp = voids[:, 5]
    areas = voids[:, 6]
    
    # Filter to percentile
    r_max = np.percentile(radii, percentile)
    valid_vols = volumes[volumes > 0]
    v_max = np.percentile(valid_vols, percentile)
    mask = (radii <= r_max) & (volumes <= v_max)
    
    radii_f = radii[mask]
    volumes_f = volumes[mask]
    dtfe_f = dtfe_interp[mask]
    areas_f = areas[mask]
    
    # Radius distribution
    ax1 = plt.subplot(2, 2, 1)
    ax1.hist(radii_f, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax1.axvline(radii_f.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {radii_f.mean():.2f}')
    ax1.set_xlabel('Radius')
    ax1.set_ylabel('Count')
    ax1.set_title(f'Void Radius Distribution (≤{percentile}th pct)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Volume distribution (log scale)
    ax2 = plt.subplot(2, 2, 2)
    valid_vols_f = volumes_f[volumes_f > 0]
    ax2.hist(np.log10(valid_vols_f), bins=50, alpha=0.7, color='coral', edgecolor='black')
    ax2.set_xlabel('log₁₀(Volume)')
    ax2.set_ylabel('Count')
    ax2.set_title(f'Void Volume Distribution (Log Scale, filtered)')
    ax2.grid(True, alpha=0.3)
    
    # DTFE distribution
    ax3 = plt.subplot(2, 2, 3)
    ax3.hist(dtfe_f, bins=50, alpha=0.7, color='purple', edgecolor='black')
    ax3.axvline(dtfe_f.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {dtfe_f.mean():.2f}')
    ax3.set_xlabel('DTFE Density')
    ax3.set_ylabel('Count')
    ax3.set_title('DTFE Density at Void Centers')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Area vs Radius
    ax4 = plt.subplot(2, 2, 4)
    ax4.scatter(radii_f, areas_f, alpha=0.3, s=10, c='navy')
    r_theory = np.linspace(radii_f.min(), radii_f.max(), 100)
    ax4.plot(r_theory, 4*np.pi*r_theory**2, 'r--', linewidth=2, label='$4\\pi r^2$')
    ax4.set_xlabel('Radius')
    ax4.set_ylabel('Area')
    ax4.set_title(f'Area vs Radius (filtered)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(f'Void Statistics Summary (filtered to {percentile}th percentile)', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    save_path = f"{save_dir}/void_statistics.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved comprehensive statistics plot to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_dtfe_distribution(voids, dtfe, save_path=None):
    """Plot the distribution of DTFE density estimates."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    dtfe_interp = voids[:, 5]
    
    # Histogram of DTFE at void centers
    ax1 = axes[0]
    n, bins, patches = ax1.hist(dtfe_interp, bins=50, alpha=0.7, color='purple',
                                 edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('DTFE Density (at void centers)', fontsize=12)
    ax1.set_ylabel('Number of Voids', fontsize=12)
    ax1.set_title('Distribution of DTFE Density at Void Centers', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add statistics
    mean_d = dtfe_interp.mean()
    median_d = np.median(dtfe_interp)
    ax1.axvline(mean_d, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_d:.2f}')
    ax1.axvline(median_d, color='green', linestyle=':', linewidth=2, label=f'Median: {median_d:.2f}')
    ax1.legend()
    
    # Compare DTFE at points vs at void centers
    ax2 = axes[1]
    ax2.hist(dtfe, bins=50, alpha=0.6, color='orange', label='At Points', edgecolor='black')
    ax2.hist(dtfe_interp, bins=50, alpha=0.6, color='purple', label='At Void Centers', edgecolor='black')
    ax2.set_xlabel('DTFE Density', fontsize=12)
    ax2.set_ylabel('Number', fontsize=12)
    ax2.set_title('DTFE Density: Points vs Void Centers', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved DTFE distribution plot to {save_path}")
    
    plt.show()


def plot_area_vs_radius(voids, save_path=None):
    """Plot surface area vs radius relationship."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    radii = voids[:, 3]
    areas = voids[:, 6]
    
    # Scatter plot with transparency
    ax.scatter(radii, areas, alpha=0.3, s=10, c='navy', edgecolors='none')
    
    # Add theoretical sphere surface area curve (4πr²)
    r_theory = np.linspace(radii.min(), radii.max(), 100)
    area_sphere = 4 * np.pi * r_theory**2
    ax.plot(r_theory, area_sphere, 'r--', linewidth=2, label='Sphere: $4\\pi r^2$')
    
    ax.set_xlabel('Void Radius', fontsize=12)
    ax.set_ylabel('Surface Area', fontsize=12)
    ax.set_title('Surface Area vs Radius', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved area vs radius plot to {save_path}")
    
    plt.show()


def plot_all_statistics(voids, dtfe, save_dir='.', show=True):
    """Create a comprehensive figure with all statistics."""
    fig = plt.figure(figsize=(14, 10))
    
    # Radius distribution
    ax1 = plt.subplot(2, 2, 1)
    radii = voids[:, 3]
    ax1.hist(radii, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax1.axvline(radii.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {radii.mean():.2f}')
    ax1.set_xlabel('Radius')
    ax1.set_ylabel('Count')
    ax1.set_title('Void Radius Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Volume distribution (log scale)
    ax2 = plt.subplot(2, 2, 2)
    volumes = voids[:, 4]
    valid_vols = volumes[volumes > 0]
    ax2.hist(np.log10(valid_vols), bins=50, alpha=0.7, color='coral', edgecolor='black')
    ax2.set_xlabel('log₁₀(Volume)')
    ax2.set_ylabel('Count')
    ax2.set_title('Void Volume Distribution (Log Scale)')
    ax2.grid(True, alpha=0.3)
    
    # DTFE distribution
    ax3 = plt.subplot(2, 2, 3)
    dtfe_interp = voids[:, 5]
    ax3.hist(dtfe_interp, bins=50, alpha=0.7, color='purple', edgecolor='black')
    ax3.axvline(dtfe_interp.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {dtfe_interp.mean():.2f}')
    ax3.set_xlabel('DTFE Density')
    ax3.set_ylabel('Count')
    ax3.set_title('DTFE Density at Void Centers')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Area vs Radius
    ax4 = plt.subplot(2, 2, 4)
    areas = voids[:, 6]
    ax4.scatter(radii, areas, alpha=0.3, s=10, c='navy')
    r_theory = np.linspace(radii.min(), radii.max(), 100)
    ax4.plot(r_theory, 4*np.pi*r_theory**2, 'r--', linewidth=2, label='$4\\pi r^2$')
    ax4.set_xlabel('Radius')
    ax4.set_ylabel('Area')
    ax4.set_title('Area vs Radius')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Void Statistics Summary', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    save_path = f"{save_dir}/void_statistics.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved comprehensive statistics plot to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Plot void statistics')
    parser.add_argument('--npoints', type=int, default=10000, 
                        help='Number of points (default: 10000). Note: >50k may require significant memory in periodic mode.')
    parser.add_argument('--boxsize', type=float, default=100.0,
                        help='Box size (default: 100.0)')
    parser.add_argument('--output-dir', type=str, default='.',
                        help='Output directory for plots')
    parser.add_argument('--mode', type=str, default='periodic',
                        choices=['open', 'periodic', 'lightcone'],
                        help='Boundary mode (default: periodic)')
    args = parser.parse_args()
    
    # Warn about memory usage for large datasets with periodic mode
    if args.npoints > 50000 and args.mode in ['periodic', 'lightcone']:
        print(f"⚠ WARNING: {args.npoints:,} points with '{args.mode}' mode will create ~{27*args.npoints:,} padded points.")
        print("  This may require significant memory (>10GB). Consider using --npoints 10000 or --mode open.\n")
    
    print("=" * 60)
    print("PyDIVE - Void Statistics Visualization Example")
    print("=" * 60)
    
    # Check available backends
    print("\nBackend availability:")
    check_backend_status()
    
    # Generate sample point distribution
    print(f"\nGenerating random point distribution...")
    np.random.seed(42)
    n_points = args.npoints
    points = np.random.random((n_points, 3)) * args.boxsize
    
    print(f"  Number of points: {n_points:,}")
    print(f"  Box size: {args.boxsize} x {args.boxsize} x {args.boxsize}")
    print(f"  Mode: {args.mode}")
    
    # Compute full catalog
    print(f"\nComputing void catalog with mode='{args.mode}'...")
    try:
        if args.mode == 'periodic' or args.mode == 'lightcone':
            voids, dtfe = get_void_catalog_full(points, mode=args.mode, boxsize=args.boxsize)
        else:
            voids, dtfe = get_void_catalog_full(points)
        print(f"✓ Found {len(voids):,} voids (tetrahedra)")
    except MemoryError:
        print(f"✗ Error: Out of memory. Try reducing --npoints or using --mode open")
        return
    except Exception as e:
        print(f"✗ Error computing void catalog: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Compute and print robust statistics
    radii = voids[:, 3]
    volumes = voids[:, 4]
    
    print("\n" + "=" * 60)
    print("VOID STATISTICS (with quantiles)")
    print("=" * 60)
    
    print("\nRADIUS STATISTICS:")
    print(f"  Count: {len(radii):,}")
    print(f"  Min: {radii.min():.4f}")
    print(f"  Max: {radii.max():.4f}")
    print(f"  Mean: {radii.mean():.4f}")
    print(f"  Median (50th): {np.percentile(radii, 50):.4f}")
    for q in [1, 10, 90, 99, 99.9]:
        print(f"  {q}th percentile: {np.percentile(radii, q):.4f}")
    
    print("\nVOLUME STATISTICS:")
    valid_vols = volumes[volumes > 0]
    if len(valid_vols) > 0:
        print(f"  Count: {len(valid_vols):,}")
        print(f"  Min: {valid_vols.min():.4f}")
        print(f"  Max: {valid_vols.max():.4f}")
        print(f"  Mean: {valid_vols.mean():.4f}")
        print(f"  Median (50th): {np.percentile(valid_vols, 50):.4f}")
        for q in [1, 10, 90, 99, 99.9]:
            print(f"  {q}th percentile: {np.percentile(valid_vols, q):.4f}")
    
    # Filter to 99th percentile for plotting
    r99 = np.percentile(radii, 99)
    v99 = np.percentile(valid_vols, 99) if len(valid_vols) > 0 else 0
    print(f"\n  99th percentile radius: {r99:.4f}")
    print(f"  99th percentile volume: {v99:.4f}")
    print(f"  Voids with radius < {r99:.2f}: {np.sum(radii < r99):,} ({100*np.sum(radii < r99)/len(radii):.1f}%)")
    print("=" * 60)
    
    # Create plots
    print("\nCreating visualization plots...")
    
    # Individual plots with 99th percentile filtering
    print("\n1. Radius distribution (filtered to 99th percentile)...")
    plot_radius_distribution_filtered(voids, save_path=f"{args.output_dir}/radius_distribution.png")
    
    print("\n2. Volume distribution (filtered to 99th percentile)...")
    plot_volume_distribution_filtered(voids, save_path=f"{args.output_dir}/volume_distribution.png")
    
    print("\n3. DTFE distribution...")
    plot_dtfe_distribution(voids, dtfe, save_path=f"{args.output_dir}/dtfe_distribution.png")
    
    print("\n4. Area vs Radius...")
    plot_area_vs_radius_filtered(voids, save_path=f"{args.output_dir}/area_vs_radius.png")
    
    print("\n5. Comprehensive summary...")
    plot_all_statistics_filtered(voids, dtfe, save_dir=args.output_dir, show=False)
    
    print("\n" + "=" * 60)
    print("All plots generated successfully!")
    print("=" * 60)
    print(f"\nGenerated files in {args.output_dir}/:")
    print("  - radius_distribution.png")
    print("  - volume_distribution.png")
    print("  - dtfe_distribution.png")
    print("  - area_vs_radius.png")
    print("  - void_statistics.png")


if __name__ == '__main__':
    main()
