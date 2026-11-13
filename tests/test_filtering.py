import numpy as np

from qsvt_tfim.filtering import apply_filter, ground_state_fidelity
from qsvt_tfim.hamiltonian import dense_qsvt_instance, plus_state


def test_zero_phase_degree_two_filter_improves_two_qubit_ground_overlap():
    instance = dense_qsvt_instance(2, J=1.0, g=0.5)
    state = plus_state(2)
    ground_state = np.asarray(instance["eigenvectors"])[:, 0]

    initial = ground_state_fidelity(state, ground_state)
    filtered, success = apply_filter(
        np.asarray(instance["A"]),
        np.zeros(3),
        state,
    )
    final = ground_state_fidelity(filtered, ground_state)

    assert 0.0 < success <= 1.0 + 1e-12
    assert final > initial


def test_postselected_state_is_normalized():
    instance = dense_qsvt_instance(2, J=1.0, g=0.5)
    filtered, _ = apply_filter(
        np.asarray(instance["A"]),
        np.zeros(3),
        plus_state(2),
    )
    assert np.isclose(np.linalg.norm(filtered), 1.0)
