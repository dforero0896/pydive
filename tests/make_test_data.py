import numpy as np
from nbodykit.lab import *

def make_test_data(n_elements, box_size=2500, seed=42):
    np.random.seed(seed)
    output = box_size * np.random.random((n_elements, 3))
    np.savetxt('tests/points.dat', output, fmt="%.5f")
def make_lognormal_data(box_size=1000, seed=42):
    redshift = 0.55
    cosmo = cosmology.Planck15
    Plin = cosmology.LinearPower(cosmo, redshift, transfer='EisensteinHu')
    b1 = 2.0

    cat = LogNormalCatalog(Plin=Plin, nbar=4e-4, BoxSize=box_size, Nmesh=512, bias=b1, seed=42)

    real_mesh = cat.to_mesh(compensated=True, resampler='cic', position='Position')
    mesh_array = real_mesh.compute()
    print(mesh_array.shape)
    np.save('tests/delta_cic_lognorm.npy', mesh_array)
    np.savetxt('tests/points.dat', cat['Position'].compute(), fmt="%.5f")

if __name__=='__main__':

    #make_test_data(int(1e4))
    make_lognormal_data(box_size=1000, seed=42)