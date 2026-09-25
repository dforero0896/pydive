#!/usr/bin/env python
"""
Example: Post-processing a DIVE void catalog with pydive.postprocess.

This script demonstrates the full workflow from a raw (overlapping) void
catalog produced by ``get_void_catalog_full`` to a *pruned*, non-overlapping
void catalog using :func:`pydive.postprocess.filter_non_overlapping` and
:func:`pydive.postprocess.central_satellite_split`, and plots the resulting
void size function plus the density, area and volume distributions of the
pruned catalog.

Catalog column layout (both backends):
    [x, y, z, radius, volume, dtfe_interp, area]

All figures and text outputs are written to this directory (examples/) or a
subdirectory of it -- see ``OUTPUT_DIR`` below.

Run:
    python example_postprocess.py
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')  # headless-safe; must be set before pyplot
import matplotlib.pyplot as plt

# Add parent directory to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pydive import get_void_catalog_full, check_backend_status
from pydive.postprocess import filter_non_overlapping, central_satellite_split

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(HERE, 'results_postprocess')   # all results go here

N_POINTS = 5000        # number of particles in the simulation box
BOXSIZE = 100.0        # periodic box side length
SEED = 42              # reproducibility

# Column indices of the full catalog
X, Y, Z, R, VOL, DTFE, AREA = range(7)


def make_points(n=N_POINTS, box=BOXSIZE, seed=SEED):
    """A mildly clustered point set so the voids look realistic."""
    rng = np.random.default_rng(seed)
    # 80% uniform background + 20% in a few dense clumps -> real under/overdensities
    pts = rng.random((n, 3)) * box
    n_clump = int(0.2 * n)
    n_clumps = 6
    idx = rng.choice(n, n_clump, replace=False)
    centers = rng.random((n_clumps, 3)) * box
    assign = rng.integers(0, n_clumps, n_clump)
    pts[idx] = centers[assign] + rng.normal(0, box * 0.03, (n_clump, 3))
    return pts % box


def main():
    print("=" * 64)
    print("PyDIVE - Postprocessing example (pruned void catalog)")
    print("=" * 64)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    check_backend_status()

    # ------------------------------------------------------------------
    # 1. Raw (overlapping) void catalog
    # ------------------------------------------------------------------
    print("\n[1] Generating points and raw void catalog ...")
    points = make_points()
    catalog, dtfe_pts = get_void_catalog_full(points, backend='scipy')
    # discard degenerate boundary tetrahedra (huge circumspheres outside box)
    inside = np.all((catalog[:, :3] > -0.5 * BOXSIZE) &
                    (catalog[:, :3] < 1.5 * BOXSIZE), axis=1)
    catalog = catalog[inside]
    print(f"    raw catalog: {len(catalog)} voids")

    # ------------------------------------------------------------------
    # 2. Prune: greedy non-overlapping filter (drop satellites)
    # ------------------------------------------------------------------
    print("\n[2] filter_non_overlapping (merge_threshold=0, PBC on) ...")
    pruned = filter_non_overlapping(catalog, radius_col=R, boxsize=BOXSIZE)
    print(f"    pruned catalog: {len(pruned)} voids "
          f"({100.0 * len(pruned) / len(catalog):.1f}% kept)")

    # ------------------------------------------------------------------
    # 3. Prune with merging of shallow overlaps
    # ------------------------------------------------------------------
    print("\n[3] filter_non_overlapping (merge_threshold=0.2) ...")
    merged, n_members = filter_non_overlapping(catalog, radius_col=R,
                                               boxsize=BOXSIZE,
                                               merge_threshold=0.2,
                                               return_labels=True)
    families = merged[n_members >= 0]
    n_absorbed = int(n_members.sum())
    print(f"    merged catalog: {len(merged)} rows, "
          f"{n_absorbed} satellites absorbed into hosts")

    # ------------------------------------------------------------------
    # 4. Central / satellite decomposition
    # ------------------------------------------------------------------
    print("\n[4] central_satellite_split ...")
    centrals, satellites, host_of = central_satellite_split(
        catalog, radius_col=R, boxsize=BOXSIZE)
    print(f"    centrals: {len(centrals)}, satellites: {len(satellites)}")

    # persist catalogs as plain text for reuse
    np.savetxt(os.path.join(OUTPUT_DIR, 'catalog_raw.txt'), catalog)
    np.savetxt(os.path.join(OUTPUT_DIR, 'catalog_pruned.txt'), pruned)
    np.savetxt(os.path.join(OUTPUT_DIR, 'catalog_merged.txt'), merged)

    # ------------------------------------------------------------------
    # 5. Plots of the PRUNED catalog
    # ------------------------------------------------------------------
    print("\n[5] Making plots (saved under examples/results_postprocess/) ...")

    # --- 5a. Void size function -----------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    bins_r = np.logspace(np.log10(pruned[:, R].min()),
                         np.log10(pruned[:, R].max()), 25)
    hist, edges = np.histogram(pruned[:, R], bins=bins_r)
    width = np.diff(edges)
    rmid = 0.5 * (edges[:-1] + edges[1:])
    # void abundance normalised per log-radius bin ("size function")
    dndlnr = hist / width * rmid
    ax.step(rmid, dndlnr, where='mid', color='tab:blue',
            label=f'pruned ({len(pruned)} voids)')
    hist_raw, _ = np.histogram(catalog[:, R], bins=bins_r)
    ax.step(rmid, hist_raw / width * rmid, where='mid', ls='--',
            color='grey', label=f'raw ({len(catalog)} voids)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Void radius $R$ [Mpc/h]')
    ax.set_ylabel('$dN/d\\ln R$')
    ax.set_title('Void size function (pruned vs raw catalog)')
    ax.legend()
    ax.grid(alpha=0.3, which='both')
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'void_size_function.png'), dpi=150)
    plt.close(fig)
    print("    - void_size_function.png")

    # --- 5b. Density (DTFE) distribution ---------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].hist(pruned[:, DTFE], bins=40, color='tab:red', alpha=0.85)
    axes[0].set_yscale('log')
    axes[0].set_xlabel(r'DTFE density at void centre $\delta_{\rm DTFE}$')
    axes[0].set_ylabel(r'$n(\delta)$')
    axes[0].set_title('Density distribution of pruned voids')
    axes[0].axvline(0.0, color='k', lw=0.8, ls=':')
    axes[0].grid(alpha=0.3)

    axes[1].hist(dtfe_pts, bins=60, color='tab:purple', alpha=0.6,
                 label='particles')
    axes[1].hist(pruned[:, DTFE], bins=60, color='tab:red', alpha=0.6,
                 label='void centres')
    axes[1].set_yscale('log')
    axes[1].set_xlabel(r'$\delta_{\rm DTFE}$')
    axes[1].set_ylabel('counts')
    axes[1].set_title('Void-centre density vs particle density')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'density_distribution.png'), dpi=150)
    plt.close(fig)
    print("    - density_distribution.png")

    # --- 5c. Area and volume distributions -------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].hist(pruned[:, VOL], bins=40, color='tab:green', alpha=0.85)
    axes[0].set_yscale('log')
    axes[0].set_xlabel('Void volume [Mpc$^3$/h$^3$]')
    axes[0].set_ylabel('counts')
    axes[0].set_title('Volume distribution (pruned catalog)')
    axes[0].grid(alpha=0.3)

    axes[1].hist(pruned[:, AREA], bins=40, color='tab:orange', alpha=0.85)
    axes[1].set_yscale('log')
    axes[1].set_xlabel('Void surface area [Mpc$^2$/h$^2$]')
    axes[1].set_ylabel('counts')
    axes[1].set_title('Area distribution (pruned catalog)')
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'area_volume_distribution.png'),
                dpi=150)
    plt.close(fig)
    print("    - area_volume_distribution.png")

    # --- 5d. Summary panel: area & volume vs radius ----------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].scatter(pruned[:, R], pruned[:, AREA], s=8, alpha=0.5,
                    color='tab:orange', label='pruned')
    axes[0].scatter(catalog[:, R], catalog[:, AREA], s=2, alpha=0.15,
                    color='grey', label='raw')
    rr = np.linspace(pruned[:, R].min(), pruned[:, R].max(), 50)
    axes[0].plot(rr, 4 * np.pi * rr ** 2, 'k--', lw=1, label='$4\\pi R^2$')
    axes[0].set_xscale('log')
    axes[0].set_yscale('log')
    axes[0].set_xlabel('Radius $R$')
    axes[0].set_ylabel('Surface area $A$')
    axes[0].set_title('Area vs radius')
    axes[0].legend()
    axes[0].grid(alpha=0.3, which='both')

    axes[1].scatter(pruned[:, R], pruned[:, VOL], s=8, alpha=0.5,
                    color='tab:green', label='pruned')
    axes[1].scatter(catalog[:, R], catalog[:, VOL], s=2, alpha=0.15,
                    color='grey', label='raw')
    axes[1].plot(rr, 4 / 3 * np.pi * rr ** 3, 'k--', lw=1,
                 label=r'$\frac{4}{3}\pi R^3$')
    axes[1].set_xscale('log')
    axes[1].set_yscale('log')
    axes[1].set_xlabel('Radius $R$')
    axes[1].set_ylabel('Volume $V$')
    axes[1].set_title('Volume vs radius')
    axes[1].legend()
    axes[1].grid(alpha=0.3, which='both')
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'area_volume_vs_radius.png'), dpi=150)
    plt.close(fig)
    print("    - area_volume_vs_radius.png")

    # ------------------------------------------------------------------
    # 6. Quick stats table
    # ------------------------------------------------------------------
    print("\n[6] Statistics of the pruned catalog:")
    print(f"    {'quantity':>10} | {'min':>10} | {'mean':>10} | {'max':>10}")
    for name, col in (('radius', R), ('volume', VOL), ('area', AREA),
                      ('dtfe', DTFE)):
        v = pruned[:, col]
        print(f"    {name:>10} | {v.min():10.3f} | {v.mean():10.3f} | {v.max():10.3f}")

    print(f"\nAll results written to: {OUTPUT_DIR}")
    print("Example completed successfully!")


if __name__ == '__main__':
    main()
