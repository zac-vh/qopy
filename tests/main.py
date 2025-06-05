import numpy as np
import matplotlib.pyplot as plt

import qopy.phase_space.wigner as wig
import qopy.phase_space.cross_wigner as xwig
import qopy.state_space.ket as qket
import qopy.state_space.density as qdens
import qopy.plotting as wplt

rl = 30
nr = 500

gamma = 1

w = wig.cubic_phase(gamma, rl, nr)
wplt.grid_2d(w, rl)
