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


def plot_radius_distribution(voids, save_path=None):
    """Plot the distribution of void radii."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    radii = voids[:, 3]
    
    # Histogram
    ax1 = axes[0]
    n, bins, patches = ax1.hist(radii, bins=50, alpha=0.7, color='steelblue', 
                                 edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('Void Radius', fontsize=12)
    ax1.set_ylabel('Number of Voids', fontsize=12)
    ax1.set_title('Distribution of Void Radii', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add mean and median lines
    mean_r = radii.mean()
    median_r = np.median(radii)
    ax1.axvline(mean_r, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_r:.2f}')
    ax1.axvline(median_r, color='green', linestyle=':', linewidth=2, label=f'Median: {median_r:.2f}')
    ax1.legend()
    
    # Cumulative distribution
    ax2 = axes[1]
    sorted_radii = np.sort(radii)
    cumulative = np.arange(1, len(sorted_radii) + 1) / len(sorted_radii)
    ax2.plot(sorted_radii, cumulative, linewidth=2, color='darkblue')
    ax2.set_xlabel('Void Radius', fontsize=12)
    ax2.set_ylabel('Cumulative Fraction', fontsize=12)
    ax2.set_title('Cumulative Distribution of Void Radii', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.05)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved radius distribution plot to {save_path}")
    
    plt.show()


def plot_volume_distribution(voids, save_path=None):
    """Plot the distribution of void volumes."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    volumes = voids[:, 4]
    
    # Histogram with log scale
    ax1 = axes[0]
    n, bins, patches = ax1.hist(volumes, bins=50, alpha=0.7, color='coral',
                                 edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('Void Volume', fontsize=12)
    ax1.set_ylabel('Number of Voids', fontsize=12)
    ax1.set_title('Distribution of Void Volumes', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add statistics
    mean_v = volumes.mean()
    median_v = np.median(volumes)
    ax1.axvline(mean_v, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_v:.2f}')
    ax1.axvline(median_v, color='green', linestyle=':', linewidth=2, label=f'Median: {median_v:.2f}')
    ax1.legend()
    
    # Log-scale histogram
    ax2 = axes[1]
    valid_volumes = volumes[volumes > 0]
    ax2.hist(np.log10(valid_volumes), bins=50, alpha=0.7, color='teal',
             edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('log₁₀(Volume)', fontsize=12)
    ax2.set_ylabel('Number of Voids', fontsize=12)
    ax2.set_title('Distribution of Void Volumes (Log Scale)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved volume distribution plot to {save_path}")
    
    plt.show()


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
    print("=" * 60)
    print("PyDIVE - Void Statistics Visualization Example")
    print("=" * 60)
    
    # Check available backends
    print("\nBackend availability:")
    check_backend_status()
    
    # Generate sample point distribution
    print("\nGenerating random point distribution...")
    np.random.seed(42)
    n_points = 3000
    points = np.random.random((n_points, 3)) * 100.0
    
    print(f"  Number of points: {n_points}")
    print(f"  Box size: 100 x 100 x 100")
    
    # Compute full catalog
    print("\nComputing void catalog...")
    try:
        voids, dtfe = get_void_catalog_full(points)
        print(f"✓ Found {len(voids)} voids (tetrahedra)")
    except Exception as e:
        print(f"✗ Error computing void catalog: {e}")
        return
    
    # Create plots
    print("\nCreating visualization plots...")
    
    # Individual plots
    print("\n1. Radius distribution...")
    plot_radius_distribution(voids, save_path='radius_distribution.png')
    
    print("\n2. Volume distribution...")
    plot_volume_distribution(voids, save_path='volume_distribution.png')
    
    print("\n3. DTFE distribution...")
    plot_dtfe_distribution(voids, dtfe, save_path='dtfe_distribution.png')
    
    print("\n4. Area vs Radius...")
    plot_area_vs_radius(voids, save_path='area_vs_radius.png')
    
    print("\n5. Comprehensive summary...")
    plot_all_statistics(voids, dtfe, save_dir='.', show=False)
    
    print("\n" + "=" * 60)
    print("All plots generated successfully!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - radius_distribution.png")
    print("  - volume_distribution.png")
    print("  - dtfe_distribution.png")
    print("  - area_vs_radius.png")
    print("  - void_statistics.png")


if __name__ == '__main__':
    main()
