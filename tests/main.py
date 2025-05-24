import qopy.phase_space.wavefunc as wfunc
import qopy.phase_space.wigner as wig
from qopy.phase_space.measures import negative_volume
import qopy.plotting as wplot
import math
import matplotlib.pyplot as plt
import numpy as np

rl = 10
nr = 200
xl = wfunc.get_xl(rl, nr)


psi0 = wfunc.psi_gauss(xl)

gamma = 0.05
cubic = wfunc.cubic_phase(gamma, xl)

psi = cubic*psi0

w0 = wig.wigner_fock(0, rl, nr)

w1 = wig.wigner_fock(1, rl, nr)
w2 = wig.wigner_fock(2, rl, nr)

v0 = negative_volume(w0, rl)
v1 = negative_volume(w1, rl)
v2 = negative_volume(w2, rl)

print(v0)
print(v1)
print(v2)

#w = wig.wigner_psi(psi, rl, nr)

#wplot.plotw([w0, w], rl)
