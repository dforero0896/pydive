from pydive.pydive import get_void_catalog_dtfe, get_sphericity, extend_boundaries_box, get_void_catalog_parallel
from scipy.spatial import Delaunay
import os
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numba

def rr_analytic(bin_low_bound, bin_high_bound, box_size):
    volume = 4 * np.pi * (bin_high_bound**3 - bin_low_bound**3) / 3
    normed_volume = volume / box_size **3
    return normed_volume

def pickle_saver(object, filename):
    print(f"==> Saving data to {filename}", flush=True)
    with open(filename+".TMP", "wb") as f:
        pickle.dump(object, f)
    os.rename(filename+".TMP", filename)


def pickle_loader(filename):
    print(f"==> Loading data from {filename}", flush=True)
    with  open(filename, "rb") as f:
        result = pickle.load(f)
    return result

def get_edge_len(tess, void_cat, rcut, dmask = True):
    mask = void_cat[:,3] >= rcut
    mask*=dmask
    coords = void_cat[tess.neighbors[mask],:3]
    edge_len = np.sqrt(((void_cat[mask,:3][:,None,:] - coords)**2).sum(axis=-1)).flatten()
    print("==> Computed edge lengths", edge_len.mean(), flush=True)
    print(void_cat[mask, 3].mean())
    return edge_len

if __name__ == '__main__':

    np.random.seed(42)

    tess = pickle_loader("tests/tess.pkl")

    if not os.path.isfile("tests/tess_rand.pkl"):
        random_sample = extend_boundaries_box(np.random.random((10000,3)) * 2500, box_size=2500, cpy_range=80, low_range=0)
        tess_rand = Delaunay(random_sample)
        pickle_saver(tess_rand, "tests/tess_rand.pkl")
    else:
        tess_rand = pickle_loader("tests/tess_rand.pkl")

    mask_dat = ((tess.points >= 0) & (tess.points <= 2500)).all(axis=1)
    mask_ran = ((tess_rand.points >= 0) & (tess_rand.points <= 2500)).all(axis=1)

    mfp_dat =  (2500**3 / tess.points[mask_dat].shape[0])**(1./3)
    mfp_ran =  (2500**3 / tess_rand.points[mask_ran].shape[0])**(1./3)

    simplex_coords=tess.points[tess.simplices[:,:], :]
    sphericities = np.zeros((simplex_coords.shape[0],3), dtype=np.double)
    void_cat = np.zeros((simplex_coords.shape[0],4), dtype=np.double)
    void_ran = np.zeros((tess_rand.simplices.shape[0], 4), dtype=np.double)
    n_threads = 32
    
    get_void_catalog_parallel(simplex_coords, void_cat, simplex_coords.shape[0], n_threads)
    get_void_catalog_parallel(tess_rand.points[tess_rand.simplices[:,:], :], void_ran, tess_rand.simplices.shape[0], n_threads)
    dmask_dat = ((void_cat[:,:3] >= 0) & (void_cat[:,:3] <= 2500)).all(axis=1)
    dmask_ran = ((void_ran[:,:3] >= 0) & (void_ran[:,:3] <= 2500)).all(axis=1)

    #bins = np.logspace(-1, 2.8, 1000)
    bins = np.linspace(0, 100, 100)
    centers = bins[:-1] + 0.5 * (bins[1:] - bins[:-1])
    fig, axes = plt.subplots(1, 3, figsize=(15,5))
    ax = axes.ravel()
    rcuts = [0, 16, 20]
    density = True
    [a.axvline(100, ls=':') for a in ax]
    [a.axvline(110, ls=':') for a in ax]
    rr_cnt = rr_analytic(bins[:-1], bins[1:], 2500)
    for rcut in rcuts[:1]:
        
        edge_len = get_edge_len(tess, void_cat, rcut, dmask = dmask_dat) #/ mfp_dat
        edges_rand = get_edge_len(tess_rand, void_ran, rcut, dmask = dmask_ran) * mfp_dat  / mfp_ran
        print(mfp_dat, mfp_ran)
        counts, _ = np.histogram(edge_len, bins=bins, density=density)
        #counts_rand, _ = np.histogram(edges_rand, bins=bins, density=density)
        counts_rand, _ = np.histogram(void_cat[dmask_dat, 3], bins=bins, density=density)
        
        ratio = counts / counts_rand
        
        ax[0].plot(centers, ratio, label=f"{rcut:.1f}")
        ax[1].plot(centers, centers**2*ratio)
        ax[2].plot(centers, centers*ratio)

        ax[0].plot(centers, counts_rand)

       

        [a.axvline(rcut, ls=':', c='k') for a in ax]
    #[a.set_xscale('log') for a in ax]
    [a.set_yscale('log') for a in ax]
    #[a.set_xlim(1e-2, 1e3) for a in ax]
    [a.set_xlim(1e-2, 60) for a in ax]
    #[a.set_ylim(1e-1, 1e2) for a in ax]
    fig.savefig("tests/edges.png")
    