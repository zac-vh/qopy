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
from qopy.phase_space.wavefunction import integrate_1d



narea = 60000
areamax = 10000
area = np.linspace(0, areamax, narea)

n = 10
area_fock = wig.fock_area(n, area)

plt.plot(area, area_fock)
plt.show()

nalpha = 500
alpha_big = np.linspace(1.1, 10, nalpha)
alpha_small = np.linspace(0.1, 0.9, nalpha)

renyi_big = np.zeros(nalpha)
renyi_small = np.zeros(nalpha)
pow = np.zeros(narea)

for i in range(nalpha):
    ai = alpha_big[i]
    pow = np.abs(area_fock/np.pi)**ai
    pnorm = np.pi*integrate_1d(pow, area)
    renyi_big[i] = np.log(pnorm)/(1-ai)

    bi = alpha_small[i]
    pow = np.abs(area_fock/np.pi)**bi
    pnorm = np.pi*integrate_1d(pow, area)
    renyi_small[i] = np.log(pnorm)/(1-bi)


print('Max Renyi: ', np.max(renyi_big))
print('Max alpha: ', alpha_big[np.argmax(renyi_big)])
print('------------')
print('Min Renyi: ', np.max(renyi_small))
print('Main alpha: ', alpha_small[np.argmax(renyi_small)])

plt.plot(alpha_big, renyi_big)
plt.plot(alpha_small, renyi_small)
plt.show()
