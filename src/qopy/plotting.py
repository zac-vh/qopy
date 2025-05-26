import numpy as np
import matplotlib.pyplot as plt
from qopy.utils.grid import grid_square as grid



def plotr(rho):
    # Plot rho
    if len(np.shape(rho)) == 2:
        rho = [rho]
    N = len(rho)
    fig = plt.figure()
    for i in range(N):
        ax = fig.add_subplot(1, N, i + 1)
        ploti = ax.matshow(np.abs(rho[i]))
        plt.colorbar(ploti, ax=ax)
    plt.show()



def plot_wigner_2d(wlist, rl=None, titles=None, maxval=None, cmap='RdBu'):
    if isinstance(wlist, np.ndarray) and wlist.ndim == 2:
        wlist = [wlist]
    N = len(wlist)
    nr = wlist[0].shape[0]
    if rl is None:
        rl = nr

    fig, axs = plt.subplots(1, N, figsize=(5 * N, 4))
    axs = np.atleast_1d(axs)

    for i, w in enumerate(wlist):
        ax = axs[i]
        if maxval is None:
            vabs = np.max(np.abs(w))
        else:
            vabs = maxval
        im = ax.imshow(w.T[::-1], extent=[-rl/2, rl/2, -rl/2, rl/2], cmap=cmap, vmin=-vabs, vmax=vabs)
        ax.set_xlabel('$x$')
        ax.set_ylabel('$p$')
        ax.set_aspect('equal')
        if titles:
            ax.set_title(titles[i])
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.show()


def plot_wigner_3d(wlist, rl=None, titles=None, maxval=None, cmap='viridis', stride=None):
    if isinstance(wlist, np.ndarray) and wlist.ndim == 2:
        wlist = [wlist]

    N = len(wlist)
    nr = wlist[0].shape[0]
    if rl is None:
        rl = nr

    mx, mp = grid(rl, nr)

    fig = plt.figure(figsize=(6 * N, 5))
    for i, w in enumerate(wlist):
        ax = fig.add_subplot(1, N, i + 1, projection='3d')
        w_real = np.real(w)

        if maxval is None:
            vabs = np.max(np.abs(w_real))
        else:
            vabs = maxval

        kwargs = dict(
            cmap=cmap,
            vmin=-vabs,
            vmax=vabs,
            linewidth=0,
            antialiased=True
        )
        if stride is not None:
            kwargs["rstride"] = stride
            kwargs["cstride"] = stride

        ax.plot_surface(mx, mp, w_real, **kwargs)
        ax.set_xlabel('$x$')
        ax.set_ylabel('$p$')
        ax.set_zlim(-vabs, vabs)
        if titles:
            ax.set_title(titles[i])

    plt.tight_layout()
    plt.show()


def plot_wigner_contour(wlist, rl=None, titles=None, levels=20, cmap='RdBu', linewidths=0.8):
    if isinstance(wlist, np.ndarray) and wlist.ndim == 2:
        wlist = [wlist]

    N = len(wlist)
    nr = wlist[0].shape[0]
    if rl is None:
        rl = nr

    mx, mp = grid(rl, nr)

    fig, axs = plt.subplots(1, N, figsize=(5 * N, 4))
    axs = np.atleast_1d(axs)

    for i, w in enumerate(wlist):
        ax = axs[i]
        w_real = np.real(w)
        vmax = np.max(np.abs(w_real))

        cs = ax.contourf(mx, mp, w_real, levels=levels, cmap=cmap, vmin=-vmax, vmax=vmax)
        ax.contour(mx, mp, w_real, levels=levels, colors='k', linewidths=linewidths, linestyles='solid')

        ax.set_xlabel('$x$')
        ax.set_ylabel('$p$')
        ax.set_aspect('equal')
        if titles:
            ax.set_title(titles[i])

        plt.colorbar(cs, ax=ax)

    plt.tight_layout()
    plt.show()


def plot_wigner_lines(wlist, rl=None, titles=None, levels=20, colors='black', linewidths=1.0):
    """
    Plot one or more Wigner functions using contour lines only (no fill).

    Parameters
    ----------
    wlist : ndarray or list of ndarray
        2D Wigner function(s) to plot.
    rl : float, optional
        Range limit for both x and p axes. Defaults to array size.
    titles : list of str, optional
        Titles for each subplot.
    levels : int or list
        Number of contour levels or explicit level values.
    colors : str or list
        Color(s) of the contour lines.
    linewidths : float
        Thickness of the contour lines.
    """
    if isinstance(wlist, np.ndarray) and wlist.ndim == 2:
        wlist = [wlist]

    N = len(wlist)
    nr = wlist[0].shape[0]
    if rl is None:
        rl = nr

    mx, mp = grid(rl, nr)

    fig, axs = plt.subplots(1, N, figsize=(5 * N, 4))
    axs = np.atleast_1d(axs)

    for i, w in enumerate(wlist):
        ax = axs[i]
        w_real = np.real(w)
        vmax = np.max(np.abs(w_real))

        ax.contour(mx, mp, w_real, levels=levels, colors=colors, linewidths=linewidths)

        ax.set_xlabel('$x$')
        ax.set_ylabel('$p$')
        ax.set_aspect('equal')
        if titles:
            ax.set_title(titles[i])

    plt.tight_layout()
    plt.show()


def plot_wigner_zero_contour(wlist, rl=None, titles=None, color='black', linewidth=1.5, linestyle='solid'):
    """
    Plot the zero-level contour (nodal line) of one or more Wigner functions.

    Parameters
    ----------
    wlist : ndarray or list of ndarray
        2D Wigner function(s) to plot.
    rl : float, optional
        Range limit for both x and p axes. Defaults to array size.
    titles : list of str, optional
        Titles for each subplot.
    color : str
        Color of the nodal lines.
    linewidth : float
        Thickness of the nodal line.
    linestyle : str
        Style of the contour line ('solid', 'dashed', etc.).
    """
    if isinstance(wlist, np.ndarray) and wlist.ndim == 2:
        wlist = [wlist]

    N = len(wlist)
    nr = wlist[0].shape[0]
    if rl is None:
        rl = nr

    mx, mp = grid(rl, nr)

    fig, axs = plt.subplots(1, N, figsize=(5 * N, 4))
    axs = np.atleast_1d(axs)

    for i, w in enumerate(wlist):
        ax = axs[i]
        w_real = np.real(w)

        # Trace uniquement la courbe W(x, p) = 0
        ax.contour(mx, mp, w_real, levels=[0], colors=color, linewidths=linewidth, linestyles=linestyle)

        ax.set_xlabel('$x$')
        ax.set_ylabel('$p$')
        ax.set_aspect('equal')  # très important pour la précision visuelle
        if titles:
            ax.set_title(titles[i])

    plt.tight_layout()
    plt.show()