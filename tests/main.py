import qopy.phase_space.wavefunc as wfunc
import qopy.phase_space.wigner as wig
import qopy.phase_space.measures as meas
from qopy.phase_space.measures import negative_volume
import qopy.plotting as wplot
from qopy.utils.grid import grid_square as grid
import qopy.state_space.ket as qopyket
import qopy.state_space.density as qopydens
import math
import matplotlib.pyplot as plt
import numpy as np
import scipy

rl = 20
nr = 500
xl = wfunc.get_xl(rl, nr)


gamma = 2

dp = 0

psi0 = wfunc.psi_gauss(xl)
cubic = wfunc.cubic_phase(gamma, xl)
psi = cubic*psi0
wpsi = wig.wigner_psi(psi, rl, nr)



#wplot.plotw(wpsi, rl)


mx, mp = grid(rl, nr)
airy = scipy.special.airy((1+4*gamma*(mp+gamma*mx**2))/(2*gamma)**(4/3))[0]
wformula = 2**(2/3)*np.exp((1+6*gamma)*mp/(6*gamma**2))/(np.sqrt(math.pi)*np.abs(gamma)**(1/3))*airy


wplot.plotw([wpsi, airy], rl)
