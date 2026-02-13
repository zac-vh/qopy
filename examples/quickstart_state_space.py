import qopy


def main() -> None:
    N = 40
    alpha = 1.0 + 0.3j

    ket = qopy.state_space.ket.coherent(alpha=alpha, N=N)
    rho = qopy.state_space.density.from_ket(ket)

    eta = 0.7
    rho_after_loss = qopy.state_space.channels.pure_loss_channel(rho, eta=eta)

    purity_before = qopy.state_space.measures.purity(rho)
    purity_after = qopy.state_space.measures.purity(rho_after_loss)
    energy_before = qopy.state_space.measures.energy(rho)
    energy_after = qopy.state_space.measures.energy(rho_after_loss)

    print("Coherent-state example")
    print(f"N = {N}, alpha = {alpha}, eta = {eta}")
    print(f"purity before loss: {purity_before:.6f}")
    print(f"purity after loss : {purity_after:.6f}")
    print(f"energy before loss: {energy_before:.6f}")
    print(f"energy after loss : {energy_after:.6f}")


if __name__ == "__main__":
    main()
