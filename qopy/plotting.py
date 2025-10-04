import numpy as np
import matplotlib.pyplot as plt
from qopy.utils.grid import grid_square as grid
from matplotlib.widgets import Slider
from qopy.phase_space.measures import marginal


def plot_2d(wlist, rl=None, titles=None, maxval=None, cmap='RdBu'):
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


def plot_3d(wlist, rl=None, titles=None, maxval=None, cmap='viridis', stride=None):
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


def plot_contour(wlist, rl=None, titles=None, levels=20, cmap='RdBu', linewidths=0.8):
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


def plot_lines(wlist, rl=None, titles=None, levels=20, colors='black', linewidths=1.0):
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


def plot_zero_contour(wlist, rl=None, titles=None, color='black', linewidth=1.5, linestyle='solid'):
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


def plot_marginal(W, rl, adaptive_maxval=False):
    nr = W.shape[0]
    x = np.linspace(-rl / 2, rl / 2, nr)
    initial_theta = 0.0

    # Initial marginal
    marg_init = marginal(W, rl, initial_theta)
    max_val = np.max(marg_init)

    # Plot setup
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)

    line, = ax.plot(x, marg_init, lw=2)
    ax.set_xlabel(r"$x_\theta$")
    ax.set_ylabel("$\\rho_{\\theta}$")
    ax.set_title("Rotated marginal distribution")
    ax.set_xlim(-rl/2, rl/2)

    if adaptive_maxval:
        ax.set_ylim(0, 1.1 * max_val)
    else:
        ax.set_ylim(0, 1.5 * max_val)

    # Slider
    ax_theta = plt.axes([0.2, 0.1, 0.65, 0.03])
    slider_theta = Slider(ax_theta, '$\\theta\\ \mathrm{(deg)}$', 0, 180, valinit=initial_theta)

    # Update function
    def update(val):
        theta = slider_theta.val*np.pi/180
        m = marginal(W, rl, theta)
        line.set_ydata(m)
        if adaptive_maxval:
            ax.set_ylim(0, 1.1 * np.max(m))
        else:
            ax.set_ylim(0, 1.5 * max_val)
        fig.canvas.draw_idle()

    slider_theta.on_changed(update)

    plt.show()
