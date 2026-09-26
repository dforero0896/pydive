#!/usr/bin/env python
"""
Recreation of the example plot shipped with the original README
(`tests/dtfe.png` in the upstream pydive repository).

The original figure was produced by `tests/test_dtfe.py` on a PATCHY halo
catalog and combined, in a single 2x4 panel figure:

  (0) void-center coordinate histograms (x, y, z)
  (1) void radius histogram, split by void overdensity delta_v, r_cut = 16
  (2) simplex volume distribution (log x)
  (3) simplex surface-area distribution (log x)
  (4) slice-averaged density field delta+1 computed with CIC
  (5) slice-averaged density field delta+1 computed with DTFE
  (6) void sphericity distribution
  (7) two-point correlation functions (CIC / DTFE / cross) -- SKIPPED HERE

As requested, the field painting (panels 4-5, which used the external
MASL/SL libraries) and the 2pcf computation (panel 7, which used Pk_library)
are omitted; the remaining panels are reproduced using the public pydive API.
Since no compiled CGAL extension or halo file is assumed here, the SciPy
backend runs on a random Poisson point sample (same style as the README
example), so the statistics are those of a uniform catalog rather than a
PATCHY mock. The CIC grid in panel (4) is built with a small internal NumPy
helper purely to keep the reference layout (it is not part of pydive).

Usage:
    python tests/example_readme_plots.py [--n 100000] [--grid 256]
                                         [--boxsize 2500.] [--out tests/dtfe_recreated.png]
"""

import argparse
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pydive import get_void_catalog_full, check_backend_status


def cic_grid(points, boxsize, grid):
    """Cloud-in-cell assignment onto a grid x grid x grid mesh.

    Minimal stand-in for MASL.MA (only used to reproduce the CIC panel of
    the original figure). Returns the delta field (mean-normalised).
    """
    delta = np.zeros((grid, grid, grid), dtype=np.float64)
    pos = np.ascontiguousarray(points[:, :3], dtype=np.float64) % boxsize
    scaled = pos * (grid / boxsize)
    i0 = np.floor(scaled).astype(np.int64)
    frac = scaled - i0
    w = np.stack([frac[..., 0], frac[..., 1], frac[..., 2]], axis=-1)
    # distribute each point over its 8 nearest grid nodes
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                wx = (1 - frac[:, 0]) if dx == 0 else frac[:, 0]
                wy = (1 - frac[:, 1]) if dy == 0 else frac[:, 1]
                wz = (1 - frac[:, 2]) if dz == 0 else frac[:, 2]
                weight = wx * wy * wz
                idx = (i0 + np.array([dx, dy, dz])) % grid
                flat = (idx[:, 0] * grid + idx[:, 1]) * grid + idx[:, 2]
                np.add.at(delta.reshape(-1), flat, weight)
    delta /= delta.mean(dtype=np.float64)
    delta -= 1.0
    return delta.astype(np.float32)


def nearest_interp_to_grid(src_points, values, grid, boxsize):
    """Nearest-neighbour interpolation of point values onto a regular grid.

    Reproduces the NearestNDInterpolator step of the original script but
    chunked so it stays memory-friendly for large grids.
    """
    from scipy.spatial import cKDTree
    tree = cKDTree(src_points[:, :3])
    x = (np.arange(grid) + 0.5) * boxsize / grid
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    out = np.empty(X.shape, dtype=np.float64)
    # query slab-by-slab along the last axis
    for i in range(grid):
        pts = np.stack([X[:, :, i].ravel(), Y[:, :, i].ravel(),
                        Z[:, :, i].ravel()], axis=1)
        _, idx = tree.query(pts, k=1)
        out[:, :, i] = values[idx].reshape(X[:, :, i].shape)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=int(1e5), help="number of points")
    ap.add_argument("--boxsize", type=float, default=2500.0)
    ap.add_argument("--grid", type=int, default=256, help="mesh for the CIC panel")
    ap.add_argument("--rcut", type=float, default=16.0, help="void radius cut [Mpc/h]")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__),
                                                  "dtfe_recreated.png"))
    args = ap.parse_args()
    boxsize = args.boxsize

    check_backend_status()

    # ---- data: same recipe as the README example -------------------------
    np.random.seed(42)
    points_raw = np.random.random((args.n, 3)) * boxsize

    # ---- void catalog with extra features --------------------------------
    s = time.time()
    try:
        voids, dtfe = get_void_catalog_full(points_raw, backend="cgal",
                                            periodic=True,
                                            box_min=[0., 0., 0.],
                                            box_max=[boxsize] * 3)
    except Exception:
        print("CGAL backend unavailable -> falling back to SciPy+Numba",
              flush=True)
        voids, dtfe = get_void_catalog_full(points_raw, backend="scipy",
                                            mode="periodic", boxsize=boxsize)
    print(f"Void catalog took {time.time() - s} s", flush=True)

    # columns: x, y, z, r, volume, dtfe_interp, area
    mask = ((voids[:, :3] > 0).all(axis=1) & (voids[:, :3] < boxsize).all(axis=1))
    voids = voids[mask]
    mean_dtfe = voids[:, 5].mean()
    void_delta = voids[:, 5] / mean_dtfe - 1.0
    sphericity = (36.0 * np.pi * voids[:, 4] ** 2) ** (1.0 / 3.0) / voids[:, 6]

    # ---- figure -----------------------------------------------------------
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    ax = axes.T.ravel()

    # (0) void center coordinates
    ax[0].hist(voids[:, 0], bins=100, histtype="step", label="x")
    ax[0].hist(voids[:, 1], bins=100, histtype="step", label="y")
    ax[0].hist(voids[:, 2], bins=100, histtype="step", label="z")
    ax[0].set_xlabel("$x$ [Mpc/$h$]")
    ax[0].legend()
    ax[0].set_title("Void centers")

    # (1) radius distribution split by void overdensity
    log_bins = np.logspace(-3, 5, 100)
    lin_bins = np.linspace(0, 100, 100)
    ax[1].hist(voids[:, 3], bins=lin_bins, histtype="step", label="r")
    for t in [0, -0.3, -0.5, -0.7, -0.9]:
        ax[1].hist(voids[void_delta < t, 3], bins=lin_bins, histtype="step",
                   label="$\\delta_v < %.1f$" % t)
    ax[1].axvline(args.rcut, ls=":", color="k", label="$r=%.0f$ Mpc/$h$" % args.rcut)
    ax[1].set_xlabel("$r$ [Mpc/$h$]")
    ax[1].legend(fontsize=8)
    ax[1].set_title("Void radii")

    # (2) simplex volume
    ax[2].hist(voids[:, 4], bins=log_bins, histtype="step", label="vol")
    ax[2].set_xscale("log")
    ax[2].set_xlabel("$V$ [(Mpc/$h$)$^3$]")
    ax[2].legend()
    ax[2].set_title("Simplex volume")

    # (3) simplex surface area
    ax[3].hist(voids[:, 6], bins=log_bins, histtype="step", label="area")
    ax[3].set_xscale("log")
    ax[3].set_xlabel("$A$ [(Mpc/$h$)$^2$]")
    ax[3].legend()
    ax[3].set_title("Simplex area")

    # (4) CIC density field, slice-averaged (kept only as a reference panel;
    #     the original painted fields with MASL -- skipped per request)
    delta_cic = cic_grid(points_raw, boxsize, args.grid)
    sl = slice(args.grid // 4, 3 * args.grid // 4)
    p = ax[4].imshow((1.0 + delta_cic)[:, :, sl].mean(axis=2), vmin=0.5, vmax=1.5)
    ax[4].set_title("$\\delta+1$ CIC")
    ax[4].set_xlabel("$X$ [Mpc/$h$]")
    ax[4].set_ylabel("$Y$ [Mpc/$h$]")
    fig.colorbar(p, ax=ax[4])

    # (5) DTFE density field interpolated to the mesh, slice-averaged
    dtfe_grid = nearest_interp_to_grid(points_raw, np.asarray(dtfe).squeeze(),
                                       args.grid, boxsize)
    mean = dtfe_grid.mean(dtype=np.float64)
    dtfe_grid = dtfe_grid / mean - 1.0
    p = ax[5].imshow((1.0 + dtfe_grid)[:, :, sl].mean(axis=2), vmin=0.5, vmax=1.5)
    ax[5].set_title("$\\delta+1$ DTFE")
    ax[5].set_xlabel("$X$ [Mpc/$h$]")
    ax[5].set_ylabel("$Y$ [Mpc/$h$]")
    fig.colorbar(p, ax=ax[5])

    # (6) sphericity
    ax[6].hist(sphericity, bins=100, histtype="step", label="sphericity")
    ax[6].set_xlabel("$\\mathcal{S}$")
    ax[6].legend()
    ax[6].set_title("Void sphericity")

    # (7) 2pcf panel intentionally left blank (computation skipped)
    ax[7].axis("off")
    ax[7].text(0.5, 0.5, "2pcf computation\nnot included",
               ha="center", va="center", fontsize=12, color="gray",
               transform=ax[7].transAxes)

    fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    print(f"Saved figure to {args.out}", flush=True)


if __name__ == "__main__":
    main()
