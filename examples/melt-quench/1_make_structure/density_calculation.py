

import numpy as np

natoms = 1000
nav = 6.02214e23
mw = 72.64
rho = 5.323 # g/cm^3
volume = (natoms*(1/nav)*mw)/rho # cm^3
# Convert to A
volume = volume*(1e8)**3
print(volume)
print(volume**(1./3.))
