#!/usr/bin/env python
import sys
import time
sys.path.append("./pydive")
from pydive.pydive import get_void_catalog_wdensity, interpolate_at_circumcenters
from dive import extend_boundaries_box
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numba

import MAS_library as MASL
import smoothing_library as SL

@numba.njit(numba.float64(numba.float64[:,:]))
def vol_tetrahedron(vertices):
    # Accepts points defining the simplex/tetrahedron (4, 3)
    matrix = np.empty((4, 4), dtype=np.double)
    for i in range(3):
        for j in range(4):
            elem = vertices[j,i] 
            matrix[i,j] = elem
    matrix[3,:] = 1
    #print(matrix)
    
    volume = np.abs(np.linalg.det(matrix) / 6)

    return volume
@numba.njit(parallel=False)
def get_density3d_tracer(volumes_buffer, densities_buffer, simplex_indices, simplex_coordinates, simplex_volumes, average_density = 4e-4):
    factor = 4 / average_density
    n_simplices = simplex_indices.shape[0]
    n_galaxies = volumes_buffer.shape[0]
    
    for i in numba.prange(n_simplices):
        #volume = vol_tetrahedron(simplex_coordinates[i,:,:])
        volume = simplex_volumes[i]
        assert(volume>0)
        for k in range(simplex_indices.shape[1]):
            vertex_index = simplex_indices[i,k]
            volumes_buffer[vertex_index] += volume    
    for i in numba.prange(n_galaxies):
        if volumes_buffer[i] > 0:
            densities_buffer[i] = factor / volumes_buffer[i]
    return volumes_buffer, densities_buffer


if __name__ == '__main__':

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    ax = axes.ravel()
    
    points_raw = pd.read_csv('tests/points.dat', engine='c', delim_whitespace=True,
                        names=['x', 'y', 'z'], usecols=(0, 1, 2)).values
    points = extend_boundaries_box(points_raw, box_size=1000, cpy_range=80, low_range=0)
    ngal = 400215 / 1000**3
    print(ngal, flush=True)
    print(f"==> Doing tesselation", flush=True)
    tess = Delaunay(points)
    print("\tDone", flush=True)
    simplex_coordinates = tess.points[tess.simplices[:,:], :]
    dummy_one = np.array([1], dtype=np.double)
    weights = np.lib.stride_tricks.as_strided(dummy_one, (points.shape[0],), (0,))
    selection = weights

    voids = np.zeros((simplex_coordinates.shape[0], 5), dtype=np.double)
    volumes = np.zeros((points.shape[0],))
    density = np.empty((points.shape[0],))
    density[:] = np.nan
    
    get_void_catalog_wdensity(simplex_coordinates, 
                                tess.simplices,
                                weights,
                                selection,
                                voids,
                                volumes,
                                density,
                                simplex_coordinates.shape[0], 
                                points.shape[0],
                                ngal,
                                4)
    print((volumes==0).sum(), tess.coplanar.shape, flush=True)
    vertices_domain = ((points>0) & (points<1000)).reshape(points.shape[0], 3).all(axis=1)
    simplices_domain = ((voids[:,:3]>0) & (voids[:,:3]<1000)).reshape(voids[:,:3].shape[0], 3).all(axis=1)
    finite_density_mask = volumes!=0
    print(np.nanmean(density[vertices_domain & finite_density_mask]))
    print(volumes[volumes!=0])
    print(density[volumes!=0])
    print(density[volumes==0])
    density -= 1
    print(np.nanmean(density[vertices_domain & finite_density_mask]))

    
    interp = LinearNDInterpolator(tess, density)
    print(f"==> Interpolating DTFE densities at circumcenters", flush=True)
    s = time.time()
    density_at_voids_dtfe = interp(voids[:,:3])
    print(f"\t Done in {time.time() - s} s", flush=True)
    print(np.nanmin(density_at_voids_dtfe), np.nanmax(density_at_voids_dtfe), np.nanmean(density_at_voids_dtfe))
    nan_mask = np.isnan(density_at_voids_dtfe)
    print(f"==> Interpolating DTFE densities at circumcenters mine", flush=True)
    s = time.time()
    interpolate_at_circumcenters(density,
                                 tess.points,
                                 tess.simplices,
                                 voids[:,:4],
                                 density_at_voids_dtfe,
                                 1,
                                 16)
    print(f"\t Done in {time.time() - s} s", flush=True)
    print(np.nanmin(density_at_voids_dtfe[~nan_mask]), np.nanmax(density_at_voids_dtfe[~nan_mask]), np.nanmean(density_at_voids_dtfe[~nan_mask]))
    print("\tDone", flush=True)
    


    # density field parameters
    grid    = 256    #the 3D field will have grid x grid x grid voxels
    BoxSize = 1000.0 #Mpc/h ; size of box
    MAS     = 'CIC'  #mass-assigment scheme
    verbose = True   #print information on progress

    # define 3D density field
    density_cic_grid = np.zeros((grid,grid,grid), dtype=np.float32)

    # construct 3D density field
    MASL.MA(points_raw.astype(np.float32), density_cic_grid, BoxSize, MAS, verbose=verbose)

    # at this point, delta contains the effective number of particles in each voxel
    # now compute overdensity and density constrast
    density_cic_grid /= np.mean(density_cic_grid, dtype=np.float64);  density_cic_grid -= 1.0
    
    print(np.mean(density_cic_grid))

    BoxSize = 1000
    R       = 20.0  #Mpc.h
    grid    = density_cic_grid.shape[0]
    Filter  = 'Top-Hat'
    threads = 16

    # compute FFT of the filter
    W_k = SL.FT_filter(BoxSize, R, grid, Filter, threads)

    # smooth the field
    field_smoothed = SL.field_smoothing(density_cic_grid.astype(np.float32), W_k, threads)
    density_cic_grid = field_smoothed

    density_at_voids_cic = np.zeros(voids.shape[0], dtype=np.float32)
    MASL.CIC_interp(density_cic_grid.astype(np.float32), BoxSize, voids[:,:3].astype(np.float32), density_at_voids_cic)


    voids = voids[simplices_domain]
    density_at_voids_cic = density_at_voids_cic[simplices_domain]
    density_at_voids_dtfe = density_at_voids_dtfe[simplices_domain]
    bins_delta = np.concatenate((-np.logspace(0.1, -5, 101), np.logspace(-5, 3, 201)))
    bins_r = np.linspace(0, 50, 101)

    ax[0].hist2d(density_at_voids_cic, voids[:,3], bins=(bins_delta, bins_r), norm=mcolors.LogNorm())
    ax[0].set_xlabel("$\delta$ CIC")
    ax[0].set_ylabel("$R$")
    ax[0].set_xscale('symlog')
    ax[1].hist2d(density_at_voids_dtfe, voids[:,3], bins=(bins_delta, bins_r), norm=mcolors.LogNorm())
    ax[1].set_xlabel("$\delta$ DTFE")
    ax[1].set_ylabel("$R$")
    ax[1].set_xscale('symlog')

    fig.savefig('tests/dtfe.png', dpi=200)


