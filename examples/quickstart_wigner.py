import os

import matplotlib
import numpy as np

import qopy

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    rl, nr = 8.0, 201

    w_vac = qopy.phase_space.wigner.fock(0, rl=rl, nr=nr)
    w_1 = qopy.phase_space.wigner.fock(1, rl=rl, nr=nr)
    w_cat = qopy.phase_space.superposition.coherent(
        alphas=[1.4, -1.4],
        cis=[1.0, 1.0],
        rl=rl,
        nr=nr,
        normalized=True,
    )

    neg_1 = qopy.phase_space.measures.negative_volume(w_1, rl=rl)
    neg_cat = qopy.phase_space.measures.negative_volume(w_cat, rl=rl)
    print(f"Negative volume |1>: {neg_1:.6f}")
    print(f"Negative volume cat: {neg_cat:.6f}")

    x = np.linspace(-rl / 2, rl / 2, nr)
    extent = [x[0], x[-1], x[0], x[-1]]

    fig, axs = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    for ax, w, title in zip(
        axs,
        [w_vac, w_1, w_cat],
        ["Vacuum |0>", "Fock |1>", "Even cat"],
    ):
        vmax = np.max(np.abs(w))
        im = ax.imshow(w.T[::-1], extent=extent, cmap="RdBu", vmin=-vmax, vmax=vmax)
        ax.set_xlabel("x")
        ax.set_ylabel("p")
        ax.set_title(title)
        ax.set_aspect("equal")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    output = "docs/assets/wigner_gallery.png"
    os.makedirs(os.path.dirname(output), exist_ok=True)
    fig.savefig(output, dpi=180)
    print(f"Saved illustration to {output}")


if __name__ == "__main__":
    main()
