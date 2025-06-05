from qopy.phase_space import wigner as wig
from qopy.phase_space import measures as meas
from qopy.plotting import plotw
from qopy.utils import majorization as maj
import matplotlib.pyplot as plt
import numpy as np


rl = 10 #size of phase-space area
nr = 200 #number of points

n = 4
wfock = wig.fock(n, rl, nr)

gamma = 0.5
disp = (0, -2.5) #displacement ensuring the cubic-phase-state is centered around the origin
sq = 1.5 #squeezing ensuring the cubic-phase-state is well contained within the grid_square
wcubic = wig.cubic_phase(gamma, rl, nr, disp, sq)

print(meas.integrate_2d(wcubic, rl)) #checking that norm is 1
print(meas.purity(wcubic, rl)) #checiking that purity is 1

plotw([wfock, wcubic], rl) #checking that all the Wigner function is contained



tol = 1e-4

vfock = wfock.ravel()
vfock = vfock[np.abs(vfock) > tol]

vcubic = wcubic.ravel()
vcubic = vcubic[np.abs(vcubic) > tol]

vfock_2mode = np.tensordot(vfock, vfock, axes=0)
vfock_2mode = vfock_2mode[np.abs(vfock_2mode) > tol]

vcubic_2mode = np.tensordot(vcubic, vcubic, axes=0)
vcubic_2mode = vcubic_2mode[np.abs(vcubic_2mode) > tol]

lorenz_cubic_2m_dec = maj.lorenz_decreasing(vcubic_2mode)*(rl/(nr-1))**4
lorenz_cubic_2m_inc = maj.lorenz_increasing(vcubic_2mode)*(rl/(nr-1))**4
lorenz_fock_2m_dec = maj.lorenz_decreasing(vfock_2mode)*(rl/(nr-1))**4
lorenz_fock_2m_inc = maj.lorenz_increasing(vfock_2mode)*(rl/(nr-1))**4

plt.plot(lorenz_fock_2m_dec, label='dec fock', color='tab:blue')
plt.plot(lorenz_fock_2m_inc, label='inc fock', color='tab:blue', linestyle='dashed')
plt.plot(lorenz_cubic_2m_dec, label='dec cubic', color='tab:orange')
plt.plot(lorenz_cubic_2m_inc, label='inc cubic', color='tab:orange', linestyle='dashed')
plt.legend()
plt.show()

'''
lorenz_dec_cubic = maj.lorenz_decreasing_2d(wcubic, rl)
lorenz_inc_cubic = maj.lorenz_increasing_2d(wcubic, rl)
lorenz_dec_fock = maj.lorenz_decreasing_2d(wfock, rl)
lorenz_inc_fock = maj.lorenz_increasing_2d(wfock, rl)

plt.plot(lorenz_inc_cubic, label='inc cubic')
plt.plot(lorenz_dec_cubic, label='dec cubic')
plt.plot(lorenz_inc_fock, label='inc fock')
plt.plot(lorenz_dec_fock, label='dec fock')
plt.legend()
plt.show()
'''