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

rl = 10
nr = 1000


gamma = 2
disp = (0, -3)
sq = 1.5

w = wig.wigner_fock(1, rl, nr)
wcubic = wig.wigner_cubic_phase(gamma, rl, nr, disp, sq)

wplot.plot_wigner_2d([w, wcubic], rl)

wplot.plot_wigner_zero_contour([w, wcubic], rl)

