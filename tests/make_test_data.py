import numpy as np

def make_test_data(n_elements, box_size=2500, seed=42):
    np.random.seed(seed)
    output = box_size * np.random.random((n_elements, 3))
    np.savetxt('tests/points.dat', output, fmt="%.5f")

if __name__=='__main__':

    make_test_data(int(1e4))