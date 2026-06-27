import numpy as np
import matplotlib.pyplot as plt
from qopy.utils.grid import square as grid
from matplotlib.widgets import Slider
from qopy.phase_space.measures import marginal
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, LightSource

def plot_2d(wlist, rl=None, titles=None, maxval=None, cmap='RdBu'):
    if isinstance(wlist, np.ndarray):
        if wlist.ndim == 2:
            wlist = [wlist]
        elif wlist.ndim == 3:
            wlist = [wlist[:, :, i] for i in range(wlist.shape[2])]
            
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
    if isinstance(wlist, np.ndarray):
        if wlist.ndim == 2:
            wlist = [wlist]
        elif wlist.ndim == 3:
            wlist = [wlist[:, :, i] for i in range(wlist.shape[2])]

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
    if isinstance(wlist, np.ndarray):
        if wlist.ndim == 2:
            wlist = [wlist]
        elif wlist.ndim == 3:
            wlist = [wlist[:, :, i] for i in range(wlist.shape[2])]

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
    if isinstance(wlist, np.ndarray):
        if wlist.ndim == 2:
            wlist = [wlist]
        elif wlist.ndim == 3:
            wlist = [wlist[:, :, i] for i in range(wlist.shape[2])]

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
    if isinstance(wlist, np.ndarray):
        if wlist.ndim == 2:
            wlist = [wlist]
        elif wlist.ndim == 3:
            wlist = [wlist[:, :, i] for i in range(wlist.shape[2])]

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


def plot_3d_artistic(
    wlist,
    rl=None,
    titles=None,
    maxval=None,
    cmap=None,
    figsize=None,
    elev=24,
    azim=-58,
    z_aspect=0.42,
    normalize_height=True,
    surface_alpha=0.96,
    surface_stride=1,
    surface_grid=True,
    surface_grid_n=14,
    surface_grid_lw=0.55,
    plane=True,
    plane_color="#aeefff",
    plane_alpha=0.78,
    plane_grid=True,
    plane_grid_n=10,
    plane_grid_lw=0.65,
    contours=True,
    contour_levels=13,
    contour_lw=1.0,
    contour_color="white",
    title_color="black",
    hide_axes=True,
):
    """
    Représentation 3D artistique d'une ou plusieurs fonctions de phase,
    dans le style des visualisations de fonctions de Wigner :
      - surface colorée
      - plan z = 0 bleu clair
      - quadrillage noir fin sur le plan
      - maillage noir fin sur la surface
      - lignes de niveau blanches projetées sur le plan
      - fond transparent
    """

    # Conversion de l'entrée selon la convention du module
    if isinstance(wlist, np.ndarray):
        if wlist.ndim == 2:
            wlist = [wlist]
        elif wlist.ndim == 3:
            wlist = [wlist[:, :, i] for i in range(wlist.shape[2])]
        else:
            raise ValueError("wlist doit être un tableau 2D, 3D, ou une liste de tableaux 2D.")

    if len(wlist) == 0:
        raise ValueError("wlist est vide.")

    N = len(wlist)
    nr = wlist[0].shape[0]

    if rl is None:
        rl = nr

    if figsize is None:
        figsize = (7.0 * N, 5.0)

    # Colormap proche de ton image
    if cmap is None:
        cmap = LinearSegmentedColormap.from_list(
            "qopy_wigner_artistic",
            [
                "#3b2c85",  # violet
                "#5a67d8",  # bleu-violet
                "#a78bfa",  # mauve clair
                "#f4f1d0",  # crème
                "#f6d79d",  # beige/orangé
                "#f5a27a",  # saumon
                "#e96b5b",  # rouge-orangé
            ],
            N=512,
        )
    elif isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)

    mx, mp = grid(rl, nr)

    x_min, x_max = -rl / 2, rl / 2
    p_min, p_max = -rl / 2, rl / 2

    fig = plt.figure(figsize=figsize)
    fig.patch.set_alpha(0.0)  # fond transparent

    for i, w in enumerate(wlist):
        ax = fig.add_subplot(1, N, i + 1, projection="3d")
        ax.set_facecolor((1, 1, 1, 0))  # transparent

        W = np.real(w)

        if maxval is None:
            vabs = np.max(np.abs(W))
        else:
            vabs = maxval

        if vabs == 0:
            vabs = 1.0

        # Normalisation de la hauteur pour un rendu visuel stable
        if normalize_height:
            Z = W / vabs
            zlim = 1.05
            color_values = np.clip(W / vabs, -1, 1)
            norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
        else:
            Z = W
            zlim = 1.05 * vabs
            color_values = W
            norm = TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs)

        eps = 1e-3 * zlim

        # ------------------------------------------------------------------
        # Plan z = 0
        # ------------------------------------------------------------------
        if plane:
            Xp, Pp = np.meshgrid([x_min, x_max], [p_min, p_max])
            Zp = np.zeros_like(Xp)

            ax.plot_surface(
                Xp,
                Pp,
                Zp,
                color=plane_color,
                alpha=plane_alpha,
                linewidth=0,
                antialiased=False,
                shade=False,
            )

        # ------------------------------------------------------------------
        # Courbes de niveau blanches projetées
        # ------------------------------------------------------------------
        if contours:
            if isinstance(contour_levels, int):
                if normalize_height:
                    levels = np.linspace(-0.9, 0.9, contour_levels)
                else:
                    levels = np.linspace(-0.9 * vabs, 0.9 * vabs, contour_levels)
            else:
                levels = contour_levels

            ax.contour(
                mx,
                mp,
                Z,
                levels=levels,
                zdir="z",
                offset=0.0 + 3 * eps,
                colors=contour_color,
                linewidths=contour_lw,
                linestyles="solid",
            )

        # ------------------------------------------------------------------
        # Quadrillage du plan
        # ------------------------------------------------------------------
        if plane and plane_grid:
            xs = np.linspace(x_min, x_max, plane_grid_n + 1)
            ps = np.linspace(p_min, p_max, plane_grid_n + 1)

            for x in xs:
                ax.plot(
                    [x, x],
                    [p_min, p_max],
                    [0.0 + 2 * eps, 0.0 + 2 * eps],
                    color="black",
                    linewidth=plane_grid_lw,
                    solid_capstyle="round",
                )

            for p in ps:
                ax.plot(
                    [x_min, x_max],
                    [p, p],
                    [0.0 + 2 * eps, 0.0 + 2 * eps],
                    color="black",
                    linewidth=plane_grid_lw,
                    solid_capstyle="round",
                )

        # ------------------------------------------------------------------
        # Surface colorée avec éclairage artificiel
        # ------------------------------------------------------------------
        light = LightSource(azdeg=315, altdeg=42)

        facecolors = light.shade(
            color_values,
            cmap=cmap,
            norm=norm,
            vert_exag=0.9,
            blend_mode="soft",
        )

        if facecolors.shape[-1] == 3:
            alpha_channel = surface_alpha * np.ones(facecolors.shape[:2] + (1,))
            facecolors = np.concatenate([facecolors, alpha_channel], axis=-1)
        else:
            facecolors[..., -1] = surface_alpha

        ax.plot_surface(
            mx,
            mp,
            Z,
            rstride=surface_stride,
            cstride=surface_stride,
            facecolors=facecolors,
            linewidth=0,
            antialiased=True,
            shade=False,
        )

        # ------------------------------------------------------------------
        # Maillage noir sur la surface
        # ------------------------------------------------------------------
        if surface_grid:
            stride = max(1, nr // surface_grid_n)

            ax.plot_wireframe(
                mx,
                mp,
                Z + eps,
                rstride=stride,
                cstride=stride,
                color="black",
                linewidth=surface_grid_lw,
            )

        # ------------------------------------------------------------------
        # Réglages caméra / rendu
        # ------------------------------------------------------------------
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(p_min, p_max)
        ax.set_zlim(-zlim, zlim)

        ax.view_init(elev=elev, azim=azim)
        ax.set_box_aspect((1.0, 1.0, z_aspect))

        try:
            ax.set_proj_type("ortho")
        except Exception:
            pass

        if titles is not None:
            ax.set_title(titles[i], color=title_color, pad=2)

        if hide_axes:
            ax.set_axis_off()
        else:
            ax.set_xlabel("$x$")
            ax.set_ylabel("$p$")
            ax.set_zlabel("$W(x,p)$")

        # rendre totalement transparents les "panes"
        try:
            ax.xaxis.pane.set_alpha(0.0)
            ax.yaxis.pane.set_alpha(0.0)
            ax.zaxis.pane.set_alpha(0.0)
        except Exception:
            pass

    plt.tight_layout()
    plt.show()