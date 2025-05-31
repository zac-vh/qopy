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


rl = 30
nr = 1000

N = 8
alpha_list = 5*np.exp(2*np.pi*1j*np.arange(N)/N)
xi_list = -1*np.exp(4*np.pi*1j*np.arange(N)/N)
amp_list = np.ones(N)#np.exp(2*np.pi*1j*np.arange(N)/N)


w = wig.wigner_gaussian_superposition(alpha_list, xi_list, amp_list, rl, nr)
w24 = wig.wigner_fock(24, rl, nr)

print(meas.integrate_2d(w, rl))
print(meas.purity(w, rl))

wplot.plot_2d([w, w24, w-w24], rl)
