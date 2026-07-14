# -*- coding: utf-8 -*-
"""Shared layout for the permanent joint latent types.

Public type IDs are one-based to preserve the model's existing ``em_type``
arguments and ``_em{type_id}`` output filenames. Component indicators are
zero/one because they multiply the type shifts estimated by the auxiliary EM.
"""

import numpy as np


TYPE_NAMES = np.array(
    [
        "S0G0T0",
        "S0G0T1",
        "S0G1T0",
        "S0G1T1",
        "S1G0T0",
        "S1G0T1",
        "S1G1T0",
        "S1G1T1",
    ]
)

TYPE_COMPONENTS = np.array(
    [
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1],
    ],
    dtype=np.int64,
)

TYPE_SCHOOL = TYPE_COMPONENTS[:, 0]
TYPE_GRANT = TYPE_COMPONENTS[:, 1]
TYPE_TRANSFER = TYPE_COMPONENTS[:, 2]
N_TYPES = len(TYPE_COMPONENTS)
TYPE_IDS = tuple(range(1, N_TYPES + 1))


def type_index(type_id):
    """Convert a public one-based type ID to a zero-based array index."""
    if isinstance(type_id, (bool, np.bool_)) or not isinstance(
        type_id, (int, np.integer)
    ):
        raise TypeError("type_id must be an integer.")
    type_id = int(type_id)
    if type_id < 1 or type_id > N_TYPES:
        raise ValueError(f"type_id must be between 1 and {N_TYPES}; received {type_id}.")
    return type_id - 1


def type_components(type_id):
    """Return ``(school_type, grant_type, transfer_type)`` for ``type_id``."""
    school, grant, transfer = TYPE_COMPONENTS[type_index(type_id)]
    return int(school), int(grant), int(transfer)


def validate_q(q, n_individuals=None, atol=1.0e-8):
    """Validate and return an individual-by-joint-type posterior matrix."""
    q = np.asarray(q, dtype=float)
    if q.ndim != 2:
        raise ValueError(f"Posterior q must be two-dimensional; received {q.shape}.")
    if q.shape[1] != N_TYPES:
        raise ValueError(
            f"Posterior q must have {N_TYPES} type columns; received {q.shape[1]}."
        )
    if n_individuals is not None and q.shape[0] != int(n_individuals):
        raise ValueError(
            f"Posterior q has {q.shape[0]} rows; expected {int(n_individuals)}."
        )
    if not np.all(np.isfinite(q)):
        raise ValueError("Posterior q contains non-finite values.")
    if np.any(q < 0.0):
        raise ValueError("Posterior q contains negative probabilities.")
    if not np.allclose(q.sum(axis=1), 1.0, atol=atol, rtol=0.0):
        raise ValueError("Every posterior row must sum to one.")
    return q


def validate_saved_layout(type_names, type_school, type_grant, type_transfer):
    """Ensure a saved EM artifact uses the structural model's type ordering."""
    saved = (
        np.asarray(type_names).astype(str),
        np.asarray(type_school, dtype=np.int64),
        np.asarray(type_grant, dtype=np.int64),
        np.asarray(type_transfer, dtype=np.int64),
    )
    expected = (TYPE_NAMES.astype(str), TYPE_SCHOOL, TYPE_GRANT, TYPE_TRANSFER)
    labels = ("type_names", "type_school", "type_grant", "type_transfer")
    for label, observed, target in zip(labels, saved, expected):
        if not np.array_equal(observed, target):
            raise ValueError(
                f"Saved EM {label} does not match the shared latent-type layout."
            )
    return True
