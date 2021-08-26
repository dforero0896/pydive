#!/usr/bin/env python
import sys
import time
#sys.path.append("./pydive")
sys.path.append("/home/astro/dforero/codes/pydive/pydive")
from pydive.pydive import get_void_catalog_cgal, get_void_catalog_full
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pickle
import os

import MAS_library as MASL
import smoothing_library as SL
import Pk_library as PKL
import readfof

HALOS="/hpcstorage/zhaoc/PATCHY_BOX/pre-recon/halo/BDM_Apk/CATALPTCICz0.562G960S1010008301.dat"
DM_FIELD="/hpcstorage/zhaoc/PATCHY_BOX/pre-recon/DMfield/1010008301.dat"
BOX_SIZE=1000
GRID_SIZE=512

N = int(1e6)
np.random.seed(42)
points_raw = np.random.random((N,4)) * BOX_SIZE


grid    = GRID_SIZE    #the 3D field will have grid x grid x grid voxels
MAS     = 'CIC'  #mass-assigment scheme
R       = 5 #Mpc.h
verbose = True   #print information on progress
Filter  = 'Gaussian'
fig, axes = plt.subplots(2, 2, figsize=(10,5))
ax = axes.ravel()
print(f"==> Interpolating to grid", flush=True)
# define 3D density field
for i, factor in enumerate([1, -1]):
    density_cic_grid = np.zeros((grid,grid,grid), dtype=np.float32)

    # construct 3D density field
    MASL.MA(points_raw[:,:3].astype(np.float32), density_cic_grid, BOX_SIZE, MAS, verbose=verbose, W=factor*np.ones(points_raw.shape[0], dtype=np.float32))


    print(np.min(density_cic_grid), np.max(density_cic_grid))
    density_cic_grid -= np.min(density_cic_grid)
    print(np.min(density_cic_grid), np.max(density_cic_grid))
    # at this point, delta contains the effective number of particles in each voxel
    # now compute overdensity and density constrast

    density_cic_grid /= np.mean(density_cic_grid, dtype=np.float64)
    density_cic_grid -= 1.0
    print(np.min(density_cic_grid), np.max(density_cic_grid))
    W_k = SL.FT_filter(BOX_SIZE, R, grid, Filter, 8)
    density_cic_grid = SL.field_smoothing(density_cic_grid, W_k, 8)
    print(np.min(density_cic_grid), np.max(density_cic_grid))
    vmin, vmax = np.min(density_cic_grid), np.max(density_cic_grid)


    

    p = ax[i].imshow((1+density_cic_grid)[:,:,200:500].mean(axis=2))#, norm=mcolors.SymLogNorm(linthresh=1e-3))#, vmin=vmin, vmax=vmax), label = 'CIC')
    ax[i].set_title(f'CIC {factor}')
    fig.colorbar(p, ax = ax[i])

    fig.savefig('tests/void_delta.png', dpi=200)


    MAS     = 'CIC'
    axis    = 0
    threads = 2

    # compute the correlation function
    
    CF_cic     = PKL.Xi(density_cic_grid.astype(np.float32), BOX_SIZE, MAS, axis, threads)
    cic_mask = CF_cic.r3D < 200
    
    

    
    ax[i+2].plot(CF_cic.r3D[cic_mask], (CF_cic.r3D**2*CF_cic.xi[:,0])[cic_mask], label='CIC', ls='-')
    ax[i+2].legend(loc=0)

    fig.savefig('tests/void_delta.png', dpi=200)