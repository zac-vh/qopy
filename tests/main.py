import qopy.phase_space.wavefunction as wfunc
import qopy.phase_space.wigner as wig
import qopy.phase_space.measures as meas
import qopy.plotting as wplot
import qopy.state_space.ket as qket
import qopy.state_space.density as qdens
import math
import matplotlib.pyplot as plt
import numpy as np
import scipy
import time
import random
import qopy.state_space.bosonic_operators as bos
import numpy as np
import matplotlib.pyplot as plt
from qopy.utils.grid import grid_square as grid
import qopy.utils.polynomials as poly

rl = 30
nr = 1000

N = 4
alist = 5*np.exp(2*np.pi*1j*np.arange(N)/N)
xilist = 0*1*np.exp(4*np.pi*1j*np.arange(N)/N)
nlist = 0*np.ones(N, dtype=int)
clist = np.ones(N)

w = wig.wigner_gaussian_fock_superposition(alist, xilist, nlist, clist, rl, nr)

print('Norm: ', meas.integrate_2d(w, rl))
print('Purity: ', meas.purity(np.abs(w), rl))

wplot.plot_2d(w, rl)
