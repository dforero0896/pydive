# PyDIVE Test Suite

This directory contains comprehensive tests for comparing and validating pydive backends.

## Test Files

### `test_backends.py` - Standalone Comparison Script
Run as a standalone script to compare all available backends:

```bash
python tests/test_backends.py
```

**Features:**
- Automatic backend detection (CGAL, SciPy+Numba)
- Correctness comparisons between backends
- Performance benchmarks with multiple dataset sizes
- Memory usage profiling
- Detailed output with timing and accuracy metrics

**Output includes:**
- Which backends are available
- Number of simplices found by each backend
- Timing comparisons (basic and full catalog)
- Numerical differences between backends
- Memory footprint measurements

### `test_backends_pytest.py` - Pytest-Compatible Tests
Run with pytest for CI/CD integration:

```bash
python -m pytest tests/test_backends_pytest.py -v
```

**Test categories:**

1. **TestBackendAvailability**
   - Checks which backends are installed
   - Ensures at least one backend is available

2. **TestScipyBackend**
   - Tests basic catalog generation (100, 500, 1000 points)
   - Tests full catalog with DTFE, volumes, areas
   - Validates periodic boundary conditions raise errors
   - Checks simplex count scaling with point count

3. **TestCGALBackend** (skipped if CGAL not installed)
   - Tests CGAL basic catalog
   - Tests CGAL full catalog

4. **TestBackendComparison** (skipped if CGAL not installed)
   - Compares circumcenter coordinates between backends
   - Compares radii, volumes, areas
   - Compares DTFE values
   - Uses tight numerical tolerances (rel=1e-10, abs=1e-12)

5. **TestNumericalStability**
   - Checks for NaN values
   - Checks for infinite values
   - Validates positive volumes and radii

## Running Tests

### Quick Test (SciPy backend only)
```bash
cd /workspace
python tests/test_backends.py
```

### Full Test Suite with pytest
```bash
cd /workspace
python -m pytest tests/test_backends_pytest.py -v
```

### Specific Test Classes
```bash
# Test only SciPy backend
python -m pytest tests/test_backends_pytest.py::TestScipyBackend -v

# Test numerical stability
python -m pytest tests/test_backends_pytest.py::TestNumericalStability -v

# Test backend comparison (requires CGAL)
python -m pytest tests/test_backends_pytest.py::TestBackendComparison -v
```

## Test Configuration

The tests use the following default settings:
- Random seed: 42 (reproducible results)
- Relative tolerance: 1e-10
- Absolute tolerance: 1e-12
- Test sizes: 100, 500, 1000 points (correctness)
- Benchmark sizes: 500, 1000, 2000, 5000 points (performance)

## Expected Results

### With SciPy+Numba Only
- All SciPy tests should PASS
- CGAL tests will be SKIPPED
- Backend comparison tests will be SKIPPED

### With Both Backends
- All tests should PASS
- Backend comparison should show matching results within tolerance
- CGAL should be faster but results should be numerically equivalent

## Troubleshooting

### "No backends available"
Install at least one backend:
```bash
# SciPy+Numba (easier)
pip install numpy scipy numba

# CGAL (faster, harder to install)
conda install -c conda-forge cgal boost-cpp
pip install .
```

### Tests failing with tolerance errors
This may indicate:
1. Different Delaunay implementations producing different triangulations
2. Numerical precision differences
3. Bugs in one of the backends

Check the detailed output to see which quantities differ and by how much.

### Performance issues
- First run includes Numba JIT compilation time (SciPy backend)
- Subsequent runs should be faster
- CGAL should consistently outperform SciPy for large datasets

## Adding New Tests

To add new test cases:

1. For pytest tests, add methods to existing classes or create new classes
2. Use fixtures for common test data
3. Mark CGAL-dependent tests with `@pytest.mark.skipif(not CGAL_AVAILABLE, ...)`
4. Use appropriate assertions with clear error messages

Example:
```python
def test_new_feature(self, test_points_500):
    """Test description."""
    result = some_function(test_points_500)
    assert expected_condition(result), "Clear error message"
```

## Performance Notes

- **SciPy+Numba**: First run includes JIT compilation (~2-3s overhead)
- **CGAL**: Consistent performance, no warmup needed
- Simplex count scales as ~5-6x point count in 3D
- Memory usage scales linearly with point count

## CI/CD Integration

Add to your CI pipeline:
```yaml
test:
  script:
    - pip install numpy scipy numba pytest
    - python -m pytest tests/test_backends_pytest.py -v
```

For full backend testing, also install CGAL in your CI environment.
