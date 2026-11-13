from __future__ import annotations

import numpy as np
from scipy.optimize import OptimizeResult, minimize

from .hamiltonian import dense_qsvt_instance, plus_state
from .qsp import dense_signal_block_encoding, qsp_scalar, qsvt_matrix


def filter_loss(phases: np.ndarray, excited_lambdas: np.ndarray) -> float:
    """Return summed squared QSP response on the excited spectrum."""
    excited_lambdas = np.asarray(excited_lambdas, dtype=float)
    return float(
        sum(abs(qsp_scalar(x, phases)[0, 0]) ** 2 for x in excited_lambdas)
    )


def optimize_filter(
    degree: int,
    excited_lambdas: np.ndarray,
    n_restarts: int = 20,
    seed: int = 42,
) -> OptimizeResult:
    """Optimize an even-degree ground-state filter by random-restart BFGS."""
    if not isinstance(degree, int) or isinstance(degree, bool):
        raise ValueError("degree must be an even positive integer.")
    if degree < 2 or degree % 2 != 0:
        raise ValueError("degree must be an even positive integer.")
    if n_restarts < 1:
        raise ValueError("n_restarts must be positive.")

    excited_lambdas = np.asarray(excited_lambdas, dtype=float)
    rng = np.random.default_rng(seed)
    best_result: OptimizeResult | None = None

    for _ in range(n_restarts):
        initial_phases = rng.uniform(-np.pi, np.pi, degree + 1)
        result = minimize(
            filter_loss,
            initial_phases,
            args=(excited_lambdas,),
            method="BFGS",
        )
        if best_result is None or result.fun < best_result.fun:
            best_result = result

    assert best_result is not None
    return best_result


def apply_filter(
    A: np.ndarray,
    phases: np.ndarray,
    state: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Apply the selected QSVT block and return normalized state and success."""
    A = np.asarray(A, dtype=complex)
    state = np.asarray(state, dtype=complex)
    system_dim = A.shape[0]

    if A.shape != (system_dim, system_dim):
        raise ValueError("A must be square.")
    if state.shape != (system_dim,):
        raise ValueError("state has incompatible dimension.")

    W = dense_signal_block_encoding(A)
    qsvt = qsvt_matrix(W, phases, system_dim)
    selected_block = qsvt[:system_dim, :system_dim]
    unnormalized = selected_block @ state
    success_probability = float(np.linalg.norm(unnormalized) ** 2)

    if success_probability <= 0.0:
        raise RuntimeError("postselection probability vanished.")

    return unnormalized / np.sqrt(success_probability), success_probability


def ground_state_fidelity(state: np.ndarray, ground_state: np.ndarray) -> float:
    """Return |<ground_state|state>|^2 for normalized vectors."""
    return float(abs(np.vdot(ground_state, state)) ** 2)


def run_filter_experiment(
    n_qubits: int,
    degrees: list[int] | tuple[int, ...],
    J: float = 1.0,
    g: float = 0.5,
    n_restarts: int = 10,
    seed: int = 42,
) -> dict[str, object]:
    """Run the dense small-system TFIM spectral-filtering experiment."""
    instance = dense_qsvt_instance(n_qubits, J=J, g=g)
    state = plus_state(n_qubits)
    ground_state = np.asarray(instance["eigenvectors"])[:, 0]

    results: dict[str, object] = {
        "instance": instance,
        "degrees": list(degrees),
        "initial_fidelity": ground_state_fidelity(state, ground_state),
        "fidelities": [],
        "success_probabilities": [],
        "losses": [],
        "phases": {},
    }

    lambdas = np.asarray(instance["lambdas"], dtype=float)

    for degree in degrees:
        optimization = optimize_filter(
            degree,
            lambdas[1:],
            n_restarts=n_restarts,
            seed=seed,
        )
        filtered_state, success = apply_filter(
            np.asarray(instance["A"]),
            optimization.x,
            state,
        )

        results["fidelities"].append(
            ground_state_fidelity(filtered_state, ground_state)
        )
        results["success_probabilities"].append(success)
        results["losses"].append(float(optimization.fun))
        results["phases"][degree] = optimization.x

    return results
