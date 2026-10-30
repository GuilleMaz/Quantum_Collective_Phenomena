"""
Energía del estado fundamental del modelo de Ising en campo transversal 1D (condiciones periódicas).

H = -J * sum_i sigma_i^z sigma_{i+1}^z  -  Gamma * sum_i sigma_i^x
"""

import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
from pathlib import Path
from scipy.sparse import csr_matrix, identity, kron
from scipy.sparse.linalg import eigsh

N = 12      # number of spins
J = 1.0     # coupling

# Operadores de un solo espín en la base sigma^z (|0> -> +1, |1> -> -1)
I2 = identity(2, format="csr")
X = csr_matrix(np.array([[0.0, 1.0], [1.0, 0.0]]))
Z = csr_matrix(np.array([[1.0, 0.0], [0.0, -1.0]]))


def site_op(op, i):
    """Operador op sobre el espín i; identidad en los demás espines."""
    ops = [I2] * N
    ops[i] = op
    return reduce(lambda a, b: kron(a, b, format="csr"), ops)


# Términos sigma_i^z sigma_{i+1}^z y sigma_i^x. No dependen de Gamma, así que
# se construyen una sola vez y el barrido en Gamma solo recombina matrices.
ZZ_TERMS = [site_op(Z, i) @ site_op(Z, (i + 1) % N) for i in range(N)]
X_TERMS = [site_op(X, i) for i in range(N)]


def hamiltonian(gamma):
    """
    Hamiltoniano disperso en la base sigma^z (|0> -> +1, |1> -> -1).

    Cada término es un producto de Kronecker de operadores de un solo espín,
    sigma_i^z = Id (x) ... (x) Z (x) ... (x) Id, con Z en la posición i, de
    modo que la suma reproduce literalmente el Hamiltoniano del enunciado.
    """
    H = -J * sum(ZZ_TERMS) - gamma * sum(X_TERMS)
    return csr_matrix(H)  # CSR: multiplicación rápida por vectores (eigsh)


def ground_state(gamma):
    """Ground state of H(gamma): energy and eigenvector."""
    evals, evecs = eigsh(hamiltonian(gamma), k=1, which="SA")
    return evals[0], evecs[:, 0]


gammas = np.linspace(0.0, 2.5, 101)
energies = np.array([ground_state(g)[0] for g in gammas])

# Guardar los resultados del barrido en un CSV
data = np.column_stack((gammas, energies, energies / N))
csv_path = Path(__file__).with_name("ift_ground_energy_vs_gamma.csv")
np.savetxt(
    csv_path,
    data,
    delimiter=",",
    header="Gamma,E0,E0_over_N",
    comments="",
)
print(f"Saved data: {csv_path.name}")

plt.plot(gammas, energies, "o-", ms=3, label=rf"Exact diagonalization, $N={N}$")
plt.axvline(J, ls=":", color="0.4")
plt.xlabel(r"$\Gamma/J$")
plt.ylabel(r"$E_0$  (units of $J$)")
plt.title(rf"Ground-state energy of the IFT chain, $N={N}$")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(Path(__file__).with_name("ift_ground_energy_vs_gamma.png"), dpi=300)

# Limiting cases
print(f"Gamma -> 0 : E0/N = {ground_state(0.0)[0] / N:.6f}  (exact: -J = {-J:.6f})")
for g in (10.0, 100.0):
    e = ground_state(g)[0] / N + g
    print(f"Gamma = {g:5.0f} : E0/N + Gamma = {e:.2e},  Gamma*(E0/N + Gamma) = {g * e:.5f}  (exact: -J^2/4 = {-J**2 / 4:.5f})")

plt.show()
