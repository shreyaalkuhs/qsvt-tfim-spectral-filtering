from qsvt_tfim.resources import lcu_resource_summary


def test_shifted_tfim_resource_counts():
    resources = lcu_resource_summary(8, degree=6, include_energy_shift=True)
    assert resources["pauli_terms"] == 16
    assert resources["label_ancillas"] == 4
    assert resources["block_encoding_queries"] == 6
    assert resources["select_calls"] == 6
    assert resources["prepare_calls"] == 12
    assert resources["phase_rotations"] == 7
    assert resources["select_branches"] == 96


def test_unshifted_tfim_has_two_n_minus_one_terms():
    resources = lcu_resource_summary(5, degree=4, include_energy_shift=False)
    assert resources["pauli_terms"] == 9
