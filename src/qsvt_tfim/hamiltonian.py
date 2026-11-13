from __future__ import annotations

from numbers import Real

import numpy as np

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


def _validate_real(name: str, value: Real) -> float:
    if not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite real number.")
    return float(value)


def local_operator(operator: np.ndarray, qubit: int, n_qubits: int) -> np.ndarray:
    """Embed a one-qubit operator using Qiskit-style rightmost qubit-0 ordering."""
    if n_qubits < 1:
        raise ValueError("n_qubits must be positive.")
    if not 0 <= qubit < n_qubits:
        raise ValueError("qubit is outside the register.")

    factors = [I2 for _ in range(n_qubits)]
    factors[n_qubits - 1 - qubit] = np.asarray(operator, dtype=complex)

    result = factors[0]
    for factor in factors[1:]:
        result = np.kron(result, factor)
    return result


def tfim_hamiltonian(
    n_qubits: int,
    J: Real = 1.0,
    g: Real = 0.5,
) -> np.ndarray:
    """Return the dense open-boundary transverse-field Ising Hamiltonian."""
    if not isinstance(n_qubits, int) or isinstance(n_qubits, bool):
        raise ValueError("n_qubits must be a positive integer.")
    if n_qubits < 1:
        raise ValueError("n_qubits must be a positive integer.")

    J = _validate_real("J", J)
    g = _validate_real("g", g)

    dimension = 2**n_qubits
    hamiltonian = np.zeros((dimension, dimension), dtype=complex)

    for qubit in range(n_qubits):
        hamiltonian -= g * local_operator(X, qubit, n_qubits)

    for qubit in range(n_qubits - 1):
        hamiltonian -= J * (
            local_operator(Z, qubit, n_qubits)
            @ local_operator(Z, qubit + 1, n_qubits)
        )

    return hamiltonian


def plus_state(n_qubits: int) -> np.ndarray:
    """Return |+>^{tensor n}."""
    if not isinstance(n_qubits, int) or isinstance(n_qubits, bool):
        raise ValueError("n_qubits must be a positive integer.")
    if n_qubits < 1:
        raise ValueError("n_qubits must be a positive integer.")

    dimension = 2**n_qubits
    return np.ones(dimension, dtype=complex) / np.sqrt(dimension)


def dense_qsvt_instance(
    n_qubits: int,
    J: Real = 1.0,
    g: Real = 0.5,
    safety_factor: Real = 1.05,
) -> dict[str, object]:
    """Construct the shifted, normalized small-system TFIM used for QSVT checks."""
    safety_factor = _validate_real("safety_factor", safety_factor)
    if safety_factor <= 1.0:
        raise ValueError("safety_factor must be greater than one.")

    hamiltonian = tfim_hamiltonian(n_qubits, J=J, g=g)
    energies, eigenvectors = np.linalg.eigh(hamiltonian)

    ground_energy = float(energies[0])
    alpha = float(safety_factor * (energies[-1] - energies[0]))

    dimension = 2**n_qubits
    contraction = (
        hamiltonian - ground_energy * np.eye(dimension, dtype=complex)
    ) / alpha
    lambdas, contraction_vectors = np.linalg.eigh(contraction)

    return {
        "n_qubits": n_qubits,
        "H": hamiltonian,
        "energies": energies,
        "eigenvectors": eigenvectors,
        "ground_energy": ground_energy,
        "gap": float(energies[1] - energies[0]),
        "alpha": alpha,
        "A": contraction,
        "lambdas": lambdas,
        "A_eigenvectors": contraction_vectors,
        "normalized_gap": float(lambdas[1] - lambdas[0]),
    }


def tfim_pauli_terms(
    n_qubits: int,
    J: Real = 1.0,
    g: Real = 0.5,
    energy_shift: Real | None = None,
) -> dict[str, float]:
    """Return the TFIM as Qiskit-ordered Pauli strings and real coefficients."""
    if not isinstance(n_qubits, int) or isinstance(n_qubits, bool):
        raise ValueError("n_qubits must be a positive integer.")
    if n_qubits < 1:
        raise ValueError("n_qubits must be a positive integer.")

    J = _validate_real("J", J)
    g = _validate_real("g", g)
    terms: dict[str, float] = {}

    if g != 0.0:
        for qubit in range(n_qubits):
            pauli = ["I"] * n_qubits
            pauli[n_qubits - 1 - qubit] = "X"
            terms["".join(pauli)] = -g

    if J != 0.0:
        for qubit in range(n_qubits - 1):
            pauli = ["I"] * n_qubits
            pauli[n_qubits - 1 - qubit] = "Z"
            pauli[n_qubits - 2 - qubit] = "Z"
            terms["".join(pauli)] = -J

    if energy_shift is not None:
        shift = _validate_real("energy_shift", energy_shift)
        if shift != 0.0:
            terms["I" * n_qubits] = -shift

    return terms
