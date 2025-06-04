import numpy as np
import matplotlib.pyplot as plt

import qopy.phase_space.wigner as wig
import qopy.phase_space.cross_wigner as xwig
import qopy.state_space.ket as qket
import qopy.state_space.density as qdens
import qopy.plotting as qplt

rl = 30
nr = 500

a = np.linspace(0, rl, nr)

plt.plot(a, np.ones(nr))
plt.show()
