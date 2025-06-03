import numpy as np

import qopy.phase_space.wigner as wig
import qopy.phase_space.cross_wigner as xwig
import qopy.state_space.ket as qket
import qopy.state_space.density as qdens
import qopy.plotting as qplt

rl = 30
nr = 1000

N = 20

wijset = xwig.fock.grid_set(N, rl, nr)

r = 1
for n in range(1, N+1):
    k = qket.squeezed_vacuum(n, r)
    rho = qdens.from_ket(k)
    w = wig.from_density_via_set(rho, wijset)
    qplt.grid_2d(w, rl)
    #w = wig.from_density_via_set(rho, wijset)

