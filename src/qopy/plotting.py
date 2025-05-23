import numpy as np
import matplotlib.pyplot as plt


def plotw(w, rl=None, view='2D'):
    if view == '2D' or view == '2d':
        plotw2d(w, rl)
    else:
        plotw3d(w, rl)
    return


def plotw2d(w, rl=None):
    if len(np.shape(w)) == 2:
        w = [w]
    N = np.shape(w)[0]
    nr = len(w[0])
    if rl is None:
        rl = nr
    fig, axs = plt.subplots(1, N)
    if N == 1:
        axs = [axs]
    for i in range(N):
        wi = w[i]
        axs[i].set_xlabel('$x$')
        axs[i].set_ylabel('$p$')
        im = axs[i].imshow(np.transpose(wi)[:][::-1], cmap='RdBu', vmin=-np.max(np.abs(wi)), vmax=np.max(np.abs(wi)), extent=[-rl/2, rl/2, -rl/2, rl/2])
        plt.colorbar(im, ax=axs[i])
    plt.show()


def plotw3d(w, rl=None, cmap='viridis', norm=None, titles=None):
    if len(np.shape(w)) == 2:
        w = [w]
    N = np.shape(w)[0]
    nr = len(w[0])
    if rl is None:
        rl = nr
    x = np.linspace(-rl / 2, rl / 2, nr)
    mx, my = np.meshgrid(x, x, indexing='ij')
    fig = plt.figure()
    for i in range(N):
        ax = fig.add_subplot(1, N, i + 1, projection='3d')
        ax.set_xlabel('$x$')
        ax.set_ylabel('$p$')
        ax.plot_surface(mx, my, np.real(w[i]), cmap=cmap, norm=norm)
        if titles is not None:
            ax.set_title(titles[i])
    plt.show()


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


def printk(ket):
    print(list(ket))
