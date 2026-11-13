from __future__ import annotations

from math import ceil, log2


def lcu_resource_summary(
    n_qubits: int,
    degree: int,
    include_energy_shift: bool = True,
) -> dict[str, int]:
    """Return logical resource counts for the TFIM Pauli/LCU representation."""
    if not isinstance(n_qubits, int) or isinstance(n_qubits, bool) or n_qubits < 1:
        raise ValueError("n_qubits must be a positive integer.")
    if not isinstance(degree, int) or isinstance(degree, bool) or degree < 1:
        raise ValueError("degree must be a positive integer.")

    term_count = 2 * n_qubits if include_energy_shift else 2 * n_qubits - 1
    label_ancillas = ceil(log2(term_count)) if term_count > 1 else 0

    return {
        "n_qubits": n_qubits,
        "degree": degree,
        "pauli_terms": term_count,
        "label_ancillas": label_ancillas,
        "block_encoding_queries": degree,
        "select_calls": degree,
        "prepare_calls": 2 * degree,
        "phase_rotations": degree + 1,
        "select_branches": degree * term_count,
    }
