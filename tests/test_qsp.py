import numpy as np

from qsvt_tfim.hamiltonian import dense_qsvt_instance
from qsvt_tfim.qsp import (
    dense_signal_block_encoding,
    qsp_scalar,
    qsvt_matrix,
    signal_unitary,
    spectral_qsp_matrix,
)


def test_scalar_signal_unitary_is_unitary():
    W = signal_unitary(0.37)
    assert np.allclose(W.conj().T @ W, np.eye(2))


def test_qsp_sequence_is_unitary():
    phases = np.array([0.2, -0.3, 0.7, 0.1])
    U = qsp_scalar(0.4, phases)
    assert np.allclose(U.conj().T @ U, np.eye(2))


def test_dense_block_encoding_is_unitary_and_contains_A():
    instance = dense_qsvt_instance(2, J=1.0, g=0.5)
    A = np.asarray(instance["A"])
    W = dense_signal_block_encoding(A)
    dimension = A.shape[0]

    assert np.allclose(W.conj().T @ W, np.eye(2 * dimension), atol=1e-12)
    assert np.allclose(W[:dimension, :dimension], A)


def test_qsvt_selected_block_matches_spectral_response():
    instance = dense_qsvt_instance(2, J=1.0, g=0.5)
    A = np.asarray(instance["A"])
    W = dense_signal_block_encoding(A)
    phases = np.array([0.2, -0.7, 1.1, 0.4, -0.3])

    U = qsvt_matrix(W, phases, system_dim=4)
    selected_block = U[:4, :4]
    expected = spectral_qsp_matrix(A, phases)

    assert np.allclose(selected_block, expected, atol=1e-12)
