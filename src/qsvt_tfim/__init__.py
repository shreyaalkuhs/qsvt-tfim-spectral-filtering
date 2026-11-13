"""QSVT spectral filtering for small transverse-field Ising models."""

from .filtering import (
    apply_filter,
    filter_loss,
    ground_state_fidelity,
    optimize_filter,
    run_filter_experiment,
)
from .hamiltonian import (
    dense_qsvt_instance,
    local_operator,
    plus_state,
    tfim_hamiltonian,
    tfim_pauli_terms,
)
from .qsp import (
    dense_signal_block_encoding,
    matrix_phase_rotation,
    phase_rotation,
    qsp_scalar,
    qsvt_matrix,
    signal_unitary,
    spectral_qsp_matrix,
)
from .resources import lcu_resource_summary

__all__ = [
    "apply_filter",
    "dense_qsvt_instance",
    "dense_signal_block_encoding",
    "filter_loss",
    "ground_state_fidelity",
    "lcu_resource_summary",
    "local_operator",
    "matrix_phase_rotation",
    "optimize_filter",
    "phase_rotation",
    "plus_state",
    "qsp_scalar",
    "qsvt_matrix",
    "run_filter_experiment",
    "signal_unitary",
    "spectral_qsp_matrix",
    "tfim_hamiltonian",
    "tfim_pauli_terms",
]
