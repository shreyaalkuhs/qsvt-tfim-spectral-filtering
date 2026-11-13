import numpy as np

from qsvt_tfim.hamiltonian import (
    dense_qsvt_instance,
    plus_state,
    tfim_hamiltonian,
    tfim_pauli_terms,
)


def test_two_qubit_tfim_matches_explicit_matrix():
    expected = np.array(
        [
            [-1.0, -0.5, -0.5, 0.0],
            [-0.5, 1.0, 0.0, -0.5],
            [-0.5, 0.0, 1.0, -0.5],
            [0.0, -0.5, -0.5, -1.0],
        ],
        dtype=complex,
    )
    actual = tfim_hamiltonian(2, J=1.0, g=0.5)
    assert np.allclose(actual, expected)


def test_two_qubit_spectrum():
    eigenvalues = np.linalg.eigvalsh(tfim_hamiltonian(2, J=1.0, g=0.5))
    expected = np.array([-np.sqrt(2), -1.0, 1.0, np.sqrt(2)])
    assert np.allclose(eigenvalues, expected)


def test_plus_state_is_normalized():
    state = plus_state(5)
    assert state.shape == (32,)
    assert np.isclose(np.linalg.norm(state), 1.0)


def test_shifted_contraction_has_ground_at_zero():
    instance = dense_qsvt_instance(3, J=1.0, g=0.5)
    lambdas = np.asarray(instance["lambdas"])
    assert np.isclose(lambdas[0], 0.0, atol=1e-12)
    assert np.max(np.abs(lambdas)) < 1.0


def test_tfim_pauli_term_count_and_ordering():
    terms = tfim_pauli_terms(3, J=1.0, g=0.5, energy_shift=-1.2)
    assert terms == {
        "IIX": -0.5,
        "IXI": -0.5,
        "XII": -0.5,
        "IZZ": -1.0,
        "ZZI": -1.0,
        "III": 1.2,
    }
