# pydive
## Python version of [DIVE](https://github.com/cheng-zhao/DIVE)

This code offers both a Scipy (Qhull-based) and [CGAL](https://www.cgal.org/) backends. The CGAL backend offers faster Delaunay Triangulations and is likely to be increasingly better incorporated in the code. While the Scipy triangulation is slower, this code offers more features such that computing volumes, areas and sphericity of the tetrahedra. Both backends have support to compute the Delaunay Tesselation Field Estimation (DTFE) of the input point set, which is not offered by the original implementation. 

Some computations use GSL, so make sure you link and include the library appropriately.

Given that CGAL is used in this code, the GMP, MPFR, BOOST and (of course) CGAL libraries are necessary. Given that the code is called from Python, the CGAL library must be built beforehand see the compilation/installation guide for CGAL [here](https://doc.cgal.org/latest/Manual/installation.html). When compilig `pydive` make sure to edit `setup.py` to your `lib` and `include` dirs for all libraries needed and add the flags `-gsl -gslcblas -CGAL -gmp -mpfr`

For information about the motivation, references and original implementation, please visit [DIVE's repository](https://github.com/cheng-zhao/DIVE). If you use this implementation in a scientific publication, please link to this repository and cive the DIVE paper.

### Usage examples

CGAL backend:
```python
sys.path.append("/home/astro/dforero/codes/pydive/pydive") # add library location to path
from pydive.pydive import get_void_catalog, extend_boundaries_box
from scipy.spatial import Delaunay
N = int(1e5)
np.random.seed(42)
# Uniform set of points in a cubic box of side 2500Mpc/h
# The code accepts a point set in comoving coordinates (xmin, ymin, zmin) >= 0
points_raw = np.random.random((N,4)) * 2500 
buffer_ones = np.ones_like(points_raw[:,0])
s = time.time()
periodic=True
compute_dtfe = False

voids = get_void_catalog_cgal(points_raw, 
                            periodic=periodic, 
                            box_size=2500, 
                            cpy_range=100, # Not used if not periodic
                            compute_dtfe=compute_dtfe,
                            weights = buffer_ones, #Not used if not dtfe
                            selection = buffer_ones, #Not used if not dtfe
                            average_density = points_raw.shape[0] / 2500**3) #Not used if not dtfe
mask = (voids[:,:3] > 0).all(axis=1) & (voids[:,:3] < 2500).all(axis=1) #Get voids within the box
voids = voids[mask]

```


Scipy backend:
```python
sys.path.append("/home/astro/dforero/codes/pydive/pydive") # add library location to path
from pydive.pydive import get_void_catalog, extend_boundaries_box
from scipy.spatial import Delaunay
N = int(1e5)
np.random.seed(42)
# Uniform set of points in a cubic box of side 2500Mpc/h
# The code accepts a point set in comoving coordinates (xmin, ymin, zmin) >= 0
points_raw = np.random.random((N,4)) * 2500 
buffer_ones = np.ones_like(points_raw[:,0])
s = time.time()
periodic=True
compute_dtfe = False
if periodic:
    # For periodic boxes duplicate points at boundaries by `cpu_range` Mpc/h
    points = extend_boundaries_box(points_raw, box_size=2500, cpy_range=100).astype(np.double)
else:
    points = points_raw

tess = Delaunay(points[:,:3]) # DT using Scipy
vertices = tess.points[tess.simplices[:,:], :]
n_simplices = vertices.shape[0]
voids = np.zeros((n_simplices, 4), dtype=np.double)
get_void_catalog(vertices,  voids, n_simplices)
mask = (voids[:,:3] > 0).all(axis=1) & (voids[:,:3] < 2500).all(axis=1) # voids within box
voids = voids[mask]

```




