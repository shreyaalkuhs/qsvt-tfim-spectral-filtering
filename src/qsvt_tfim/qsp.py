from __future__ import annotations

from numbers import Real

import numpy as np


def signal_unitary(x: Real) -> np.ndarray:
    """Return the scalar QSP signal unitary W(x)."""
    x = float(x)
    if not np.isfinite(x):
        raise ValueError("x must be finite.")
    if abs(x) > 1.0 + 1e-12:
        raise ValueError("|x| must be at most one.")

    x = float(np.clip(x, -1.0, 1.0))
    complement = np.sqrt(max(0.0, 1.0 - x**2))

    return np.array(
        [
            [x, 1j * complement],
            [1j * complement, x],
        ],
        dtype=complex,
    )


def phase_rotation(phi: Real) -> np.ndarray:
    """Return exp(i phi Z)."""
    phi = float(phi)
    if not np.isfinite(phi):
        raise ValueError("phi must be finite.")

    return np.array(
        [
            [np.exp(1j * phi), 0.0],
            [0.0, np.exp(-1j * phi)],
        ],
        dtype=complex,
    )


def qsp_scalar(x: Real, phases: np.ndarray) -> np.ndarray:
    """Return the scalar QSP unitary for one signal value x."""
    phases = np.asarray(phases, dtype=float)
    if phases.ndim != 1 or phases.size == 0:
        raise ValueError("phases must be a nonempty one-dimensional array.")

    unitary = phase_rotation(phases[0])
    signal = signal_unitary(x)

    for phi in phases[1:]:
        unitary = unitary @ signal @ phase_rotation(phi)

    return unitary


def dense_signal_block_encoding(A: np.ndarray) -> np.ndarray:
    """Return [[A, i sqrt(I-A^2)], [i sqrt(I-A^2), A]]."""
    A = np.asarray(A, dtype=complex)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square.")
    if not np.allclose(A, A.conj().T):
        raise ValueError("A must be Hermitian.")

    eigenvalues, eigenvectors = np.linalg.eigh(A)
    if np.max(np.abs(eigenvalues)) > 1.0 + 1e-10:
        raise ValueError("A must be a Hermitian contraction.")

    complement = (
        eigenvectors
        @ np.diag(np.sqrt(np.clip(1.0 - eigenvalues**2, 0.0, None)))
        @ eigenvectors.conj().T
    )

    return np.block(
        [
            [A, 1j * complement],
            [1j * complement, A],
        ]
    )


def matrix_phase_rotation(phi: Real, system_dim: int) -> np.ndarray:
    """Return exp(i phi Z) tensor I_system."""
    if system_dim < 1:
        raise ValueError("system_dim must be positive.")
    return np.kron(phase_rotation(phi), np.eye(system_dim, dtype=complex))


def qsvt_matrix(W: np.ndarray, phases: np.ndarray, system_dim: int) -> np.ndarray:
    """Return the dense matrix QSVT sequence for a signal-unitary W."""
    W = np.asarray(W, dtype=complex)
    phases = np.asarray(phases, dtype=float)

    expected_shape = (2 * system_dim, 2 * system_dim)
    if W.shape != expected_shape:
        raise ValueError(f"W must have shape {expected_shape}.")
    if phases.ndim != 1 or phases.size == 0:
        raise ValueError("phases must be a nonempty one-dimensional array.")

    unitary = matrix_phase_rotation(phases[0], system_dim)
    for phi in phases[1:]:
        unitary = unitary @ W @ matrix_phase_rotation(phi, system_dim)
    return unitary


def spectral_qsp_matrix(A: np.ndarray, phases: np.ndarray) -> np.ndarray:
    """Evaluate the scalar QSP response on the eigenvalues of Hermitian A."""
    A = np.asarray(A, dtype=complex)
    eigenvalues, eigenvectors = np.linalg.eigh(A)
    responses = np.array([qsp_scalar(x, phases)[0, 0] for x in eigenvalues])
    return eigenvectors @ np.diag(responses) @ eigenvectors.conj().T
