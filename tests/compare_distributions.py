import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(42)
pydive_fn = "tests/voids_pydive.dat"
#dive_box_fn = "tests/CATALPTCICz0.638G960S1005638091_zspace.VOID.dat"
dive_box_fn = "tests/voids_dive_box.dat"

pydive_data = pd.read_csv(pydive_fn, delim_whitespace=True, engine="c", names = ['x', 'y', 'z', 'r', 'v']).fillna(0)
dive_box_data = pd.read_csv(dive_box_fn, delim_whitespace=True, engine="c", names = ['x', 'y', 'z', 'r', 'v']).fillna(0)
print(f"==> PyDIVE statistics:")
print(pydive_data.describe())
print(f"==> DIVE statistics:")
print(dive_box_data.describe())
bins = np.logspace(-20, 11, 101)
fig, ax = plt.subplots(3, 2)
axr = ax.ravel()
for i in range(len(pydive_data.columns)):
    axr[i].hist(pydive_data[pydive_data.columns[i]], histtype="step", color="b", label="PyDIVE", bins=bins)

    axr[i].hist(dive_box_data[dive_box_data.columns[i]], histtype="step", color="g", label="DIVE", ls="--", bins=bins)
    axr[i].set_title(pydive_data.columns[i])
    axr[i].set_xscale('log')
    axr[i].legend()

ids = np.random.randint(low=0, high=pydive_data.shape[0], size=200)
axr[-1].hist(pydive_data['v'] / pydive_data['r']**3, bins=bins)
axr[-1].set_xscale('log')
axr[-1].set_yscale('log')
fig.savefig("tests/distributions.png", dpi=300)

f = plt.figure()
pydive_fn = "tests/voids_pydive_lc.dat"
#dive_box_fn = "tests/CATALPTCICz0.638G960S1005638091_zspace.VOID.dat"
dive_box_fn = "tests/voids_dive_lc.dat"

pydive_data = pd.read_csv(pydive_fn, delim_whitespace=True, engine="c", names = ['x', 'y', 'z', 'r', 'v']).fillna(0)
dive_box_data = pd.read_csv(dive_box_fn, delim_whitespace=True, engine="c", names = ['x', 'y', 'z', 'r', 'v']).fillna(0)
print(f"==> PyDIVE statistics:")
print(pydive_data.describe())
print(f"==> DIVE statistics:")
print(dive_box_data.describe())
#bins = np.linspace(0, 1000, 101)
#bins = 10000
#bins = np.logspace(-11, 11, 101)
fig, ax = plt.subplots(3, 2)
axr = ax.ravel()
for i in range(len(pydive_data.columns)):
    axr[i].hist(pydive_data[pydive_data.columns[i]], histtype="step", color="b", label="PyDIVE", bins=bins, density=True)
    axr[i].hist(dive_box_data[dive_box_data.columns[i]], histtype="step", color="g", label="DIVE", ls="--", bins=bins, density=True)
    axr[i].set_title(pydive_data.columns[i])
    axr[i].set_xscale('log')
    axr[i].legend()

ids = np.random.randint(low=0, high=pydive_data.shape[0], size=200)
axr[-1].hist(pydive_data['v'] / pydive_data['r']**3, bins=bins)
axr[-1].set_xscale('log')
axr[-1].set_yscale('log')

fig.savefig("tests/lc_distributions.png", dpi=300)
