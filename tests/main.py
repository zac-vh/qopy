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



N = 10

alpha = 1+2j

m = 2
n = 4

Dalpha = bos.displacement(N, alpha)
disp = qdens.displacement_matrix(m, n, alpha)


print(Dalpha[n, m]-disp)