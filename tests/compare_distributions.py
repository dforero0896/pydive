import pandas as pd
import matplotlib.pyplot as plt

pydive_fn = "tests/voids_pydive.dat"
dive_box_fn = "tests/voids_dive_box.dat"

pydive_data = pd.read_csv(pydive_fn, delim_whitespace=True, engine="c", names = ['x', 'y', 'z', 'r'])
dive_box_data = pd.read_csv(dive_box_fn, delim_whitespace=True, engine="c", names = ['x', 'y', 'z', 'r'])
print(f"==> PyDIVE statistics:")
print(pydive_data.describe())
print(f"==> DIVE statistics:")
print(dive_box_data.describe())

ax = pydive_data.hist(histtype="step", color="b", label="PyDIVE")
dive_box_data.hist(histtype="step", color="g", label="DIVE", ax=ax, ls="--")
plt.legend()
plt.savefig("tests/distributions.png", dpi=300)