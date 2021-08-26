import numpy as np
import pandas as pd
import sys
import os
import time
sys.path.append("pydive")
from pydive.pydive import get_void_catalog_cgal, get_void_catalog, extend_boundaries_box
import MAS_library as MASL
import smoothing_library as SL
import Pk_library as PKL
import matplotlib.pyplot as plt

#data_fn = "/home/daniel/scratch/projects/baosystematics/patchy_results/box1/real/nosyst/mocks_gal_xyz/CATALPTCICz0.466G960S1005638091.dat"


#data = pd.read_csv(data_fn, delim_whitespace=True, engine='c', names = ['x', 'y', 'z'], usecols=(0,1,2)).values
data = np.random.random((100000, 3)) * 2500
rcut = 16
# density field parameters
grid    = 512    #the 3D field will have grid x grid x grid voxels
BoxSize = 2500.0 #Mpc/h ; size of box
MAS     = 'CIC'  #mass-assigment scheme
verbose = True   #print information on progress
axis = 0
threads = 8

opath = "/tmp/"
for func in [get_void_catalog_cgal]:
    oname = opath + func.__name__ + '.npy'
    if not os.path.isfile(oname) or 1:
        s = time.time()
        void_cc = func(data, periodic=False, box_size=BoxSize, cpy_range=50, compute_dtfe=False)
        print("DT took", time.time() - s, "s", flush=True)
        np.save(oname,void_cc)
    else:
        void_cc = np.load(oname)

    
    mask_cc = (void_cc[:,3] > rcut) & (void_cc[:,:3] > 0).all(axis=1) & (void_cc[:,:3] < BoxSize).all(axis=1)


    void_cc = void_cc[mask_cc]

    print(void_cc)



    delta_cc = np.zeros((grid,grid,grid), dtype=np.float32)
    MASL.MA(void_cc[:,:3].astype(np.float32), delta_cc, BoxSize, MAS, verbose=verbose)
    delta_cc /= np.mean(delta_cc, dtype=np.float64);  delta_cc -= 1.0


    CF_cc     = PKL.Xi(delta_cc, BoxSize, MAS, axis, threads)


    mask_cc = CF_cc.r3D < 200


    plt.plot(CF_cc.r3D[mask_cc], (CF_cc.r3D**2*CF_cc.xi[:,0])[mask_cc], label=func.__name__)
    plt.legend()

    plt.savefig("tests/centers.png")









