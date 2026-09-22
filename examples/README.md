# PyDIVE Examples

This directory contains example scripts demonstrating how to use the pydive package with different backends.

## Available Examples

### 1. Basic Usage (`example_basic.py`)
Demonstrates:
- Automatic backend selection
- Explicit backend specification (CGAL vs SciPy)
- Basic void catalog computation
- Comparing results between backends

**Run:**
```bash
python example_basic.py
```

### 2. Full Catalog (`example_full_catalog.py`)
Demonstrates:
- Computing full void catalog with all features
- Accessing DTFE density estimates
- Computing tetrahedron volumes and surface areas
- Backend comparison with numerical tolerance checking

**Run:**
```bash
python example_full_catalog.py
```

### 3. Visualization (`example_plotting.py`)
Demonstrates:
- Creating publication-quality plots of void statistics
- Radius distribution histograms
- Volume distribution (linear and log scale)
- DTFE density distributions
- Area vs radius relationships
- Comprehensive summary figures

**Requirements:** matplotlib
```bash
pip install matplotlib
python example_plotting.py
```

**Output files:**
- `radius_distribution.png`
- `volume_distribution.png`
- `dtfe_distribution.png`
- `area_vs_radius.png`
- `void_statistics.png`

### 4. Performance Benchmark (`example_benchmark.py`)
Demonstrates:
- Benchmarking backend performance
- Comparing CGAL vs SciPy+Numba speed
- Generating performance comparison plots
- Interactive benchmark configuration

**Run:**
```bash
python example_benchmark.py
```

## Quick Start

If you're new to pydive, start with the basic example:

```bash
cd examples
python example_basic.py
```

This will show you:
1. Which backends are available on your system
2. How to compute a basic void catalog
3. The difference between backends (if both are available)

## Backend Selection

All examples support both backends:

```python
from pydive import get_void_catalog

# Auto-select best available backend (recommended)
voids = get_void_catalog(points)

# Explicitly use SciPy+Numba (easier installation)
voids = get_void_catalog(points, backend='scipy')

# Explicitly use CGAL (fastest, requires CGAL library)
voids = get_void_catalog(points, backend='cgal')
```

## Requirements

**Core requirements (all examples):**
- numpy
- scipy
- numba

**For plotting examples:**
- matplotlib

**For CGAL backend (optional but recommended for performance):**
- CGAL library
- Boost libraries
- GMP and MPFR libraries

See the main README.md and INSTALL_CGAL.md for detailed installation instructions.

## Understanding the Output

### Basic Catalog
Returns an array with shape `(n_simplices, 4)`:
- Columns: `[x, y, z, radius]`
- Each row represents a tetrahedron's circumcenter and circumradius

### Full Catalog
Returns two arrays:
1. Void catalog with shape `(n_simplices, 7)`:
   - Columns: `[x, y, z, radius, volume, dtfe_interp, area]`
2. DTFE density at input points with shape `(n_points,)`

## Tips

1. **Start small**: Begin with 1000-2000 points to test, then scale up
2. **Use auto-selection**: Let pydive choose the best backend unless you have specific needs
3. **Check backend status**: Use `check_backend_status()` to see what's available
4. **Reproduce results**: Set numpy random seed for reproducible point distributions
5. **Memory usage**: Full catalog uses more memory than basic catalog

## Troubleshooting

**No backends available:**
```python
from pydive import check_backend_status
check_backend_status()
```
Install at least one backend (SciPy+Numba is easiest).

**CGAL import errors:**
See INSTALL_CGAL.md for detailed CGAL installation instructions.

**Memory errors with large datasets:**
Try smaller point counts or use basic catalog instead of full catalog.

## Next Steps

After running these examples:
1. Try with your own point data
2. Explore the test suite in `tests/` for more advanced usage
3. Read the source code in `pydive/` to understand the implementation
4. Check the documentation for API reference
