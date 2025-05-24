import numpy as np
import scipy
import math
from qutip import clebsch
import matplotlib.pyplot as plt
from qopy.utils.grid import grid_sphere as grid

# Spherical Wigner functions for qudits
def spherical_harmonic_theta_phi(l, m, theta, phi):
    return scipy.special.sph_harm(m, l, phi, theta, out=None)


def sperical_harmonic(l, m, n_theta, n_phi=None):
    if n_phi is None:
        n_phi = 2*n_theta
    mth, mph = grid(n_theta, n_phi)
    return spherical_harmonic_theta_phi(l, m, mth, mph)


def dicke_spherical_wigner(j, m, n_theta, n_phi=None):
    sw = dicke_spherical_cross_wigner(j, m, m, n_theta, n_phi)
    return np.real(sw)


def dicke_spherical_cross_wigner(j, ma, mb, n_theta, n_phi=None):
    if n_phi is None:
        n_phi = 2*n_theta
    mth, mph = grid(n_theta, n_phi)
    sw = np.zeros([n_theta, n_phi])
    for l in np.arange(0, 2*j+1):
        for m in np.arange(-l, l+1):
            sw = sw + np.sqrt((2*l+1)/(2*j+1))*clebsch(j, l, j, ma, m, mb)*spherical_harmonic_theta_phi(l, m, mth, mph)
    sw = np.sqrt((4*math.pi)/(2*j+1))*sw
    return sw


def plot_spherical_wigner_2d(sw):
    if len(np.shape(sw)) == 2:
        sw = [sw]
    N = np.shape(sw)[0]
    fig, axs = plt.subplots(1, N)
    if N == 1:
        axs = [axs]
    for i in range(N):
        swi = sw[i]
        axs[i].set_xlabel('$\\phi$')
        axs[i].set_ylabel('$\\theta$')
        im = axs[i].imshow(swi, cmap='RdBu', vmin=-np.max(np.abs(swi)), vmax=np.max(np.abs(swi)), extent=[0, 2*math.pi, math.pi, 0])
        plt.colorbar(im, ax=axs[i])
    plt.show()

def integrate_spherical_wigner(sw, j=1/2):
    n_theta = sw.shape[0]
    n_phi = sw.shape[1]
    x_theta = np.linspace(0, math.pi, n_theta)
    x_phi = np.linspace(0, 2*math.pi, n_phi)
    res = scipy.integrate.simpson(scipy.integrate.simpson(sw, x=x_phi)*np.sin(x_theta), x=x_theta)
    return ((2*j+1)/(4*math.pi))*res


def plotsw(sw, view='2D'):
    if view == '2D' or view == '2d':
        plot_spherical_wigner_2d(sw)
    else:
        plot_spherical_wigner_3d(sw)
    return


def plot_spherical_wigner_3d(sw):
    if len(np.shape(sw)) == 2:
        sw = [sw]
    N = np.shape(sw)[0]
    n_theta = sw[0].shape[0]
    n_phi = sw[0].shape[1]
    phi = np.linspace(0, 2 * np.pi, num=n_phi, endpoint=False)
    theta = np.linspace(np.pi * (-1 / 2 + 1. / (n_theta + 1)), np.pi / 2, num=n_theta, endpoint=False)
    phi, theta = np.meshgrid(phi, theta)
    theta, phi = theta.ravel(), phi.ravel()
    mesh_x, mesh_y = ((np.pi / 2 - theta) * np.cos(phi), (np.pi / 2 - theta) * np.sin(phi))
    triangles = mtri.Triangulation(mesh_x, mesh_y).triangles
    x, y, z = np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi), np.sin(theta)

    fig = plt.figure()
    for i in range(N):
        ax = fig.add_subplot(1, N, i + 1, projection='3d')
        swi = sw[i]
        vals = swi.ravel()
        colors = np.mean(vals[triangles], axis=1)
        cmap = plt.get_cmap('RdBu')
        triang = mtri.Triangulation(x, y, triangles)
        collec = ax.plot_trisurf(triang, z, cmap=cmap, shade=False, linewidth=0.)
        collec.set_array(colors)
        collec.autoscale()
        ax.set_box_aspect([1, 1, 1])
    plt.show()
    return


def marginal_spherical(sw, angle='theta'):
    n_theta = sw.shape[0]
    n_phi = sw.shape[1]
    x_theta, x_phi = np.linspace(0, math.pi, n_theta), np.linspace(0, 2 * math.pi, n_phi)
    mesh_theta, mesh_phi = np.meshgrid(x_theta, x_phi, indexing='ij')
    if angle == 'theta' or angle == 'th':
        marg_theta = scipy.integrate.simpson(sw, x=x_phi, axis=1)/(2*math.pi)
        return marg_theta
    if angle == 'phi' or angle == 'ph':
        marg_phi = scipy.integrate.simpson(sw * np.sin(mesh_theta), x=x_theta, axis=0)/(2*math.pi)
        return marg_phi
    else:
        marg_theta = scipy.integrate.simpson(sw, x=x_phi, axis=1)/(2*math.pi)
        marg_phi = scipy.integrate.simpson(sw * np.sin(mesh_theta), x=x_theta, axis=0)/(2*math.pi)
        return [marg_theta, marg_phi]


def dicke_spherical_crosss_wigner_set(j, n_theta, n_phi=None):
    if n_phi is None:
        n_phi = 2*n_theta
    swijset = np.zeros([round(2*j+1), round(2*j+1), n_theta, n_phi], dtype=complex)
    mlist = np.arange(-j, j+1)
    for a in range(round(2*j+1)):
        ma = mlist[a]
        swijset[a, a] = dicke_spherical_wigner(j, ma, n_theta, n_phi)
        for b in range(a+1, round(2*j+1)):
            mb = mlist[b]
            swij = dicke_spherical_cross_wigner(j, ma, mb, n_theta, n_phi)
            swijset[a, b] = swij
            swijset[b, a] = np.conj(swij)
    return swijset

