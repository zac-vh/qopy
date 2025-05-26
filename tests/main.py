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

rl = 50
nr = 2000



a = np.linspace(-5, 5, 10)

b = a.ravel()

print(a-b)