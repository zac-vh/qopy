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


rl = 10
nr = 500
N = 4
ket = qket.random_ket(N)
rho = qdens.ket_to_rho(ket)
w = wig.density_to_wigner(rho, rl, nr)


wplot.plot_marginal_with_slider(w, rl)