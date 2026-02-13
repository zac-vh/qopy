# Quickstart

This guide shows a minimal path from installation to first results with `qopy`.

## 1. Install

```bash
pip install git+https://github.com/zac-vh/qopy.git
```

## 2. Wigner function of a non-classical state

```python
import qopy

rl, nr = 8.0, 201
w = qopy.phase_space.wigner.fock(n=1, rl=rl, nr=nr)
neg = qopy.phase_space.measures.negative_volume(w, rl=rl)

print(f"negative volume = {neg:.6f}")
qopy.plotting.plot_2d([w], rl=rl, titles=["Wigner of |1>"])
```

## 3. Density matrix workflow

```python
import qopy

N = 40
ket = qopy.state_space.ket.squeezed_vacuum(N=N, r=0.6)
rho = qopy.state_space.density.from_ket(ket)

purity = qopy.state_space.measures.purity(rho)
entropy = qopy.state_space.measures.von_neumann_entropy(rho)

print(f"purity = {purity.real:.6f}")
print(f"von Neumann entropy = {entropy:.6f}")
```

## 4. Apply a channel

```python
import qopy

N = 30
ket = qopy.state_space.ket.coherent(alpha=1.0, N=N)
rho = qopy.state_space.density.from_ket(ket)

eta = 0.75
rho_loss = qopy.state_space.channels.pure_loss_channel(rho, eta=eta)

print("energy before:", qopy.state_space.measures.energy(rho))
print("energy after :", qopy.state_space.measures.energy(rho_loss))
```

## 5. Reproducible example scripts

- `examples/quickstart_wigner.py`
- `examples/quickstart_state_space.py`

These scripts are intended as small templates for new users.
