# -*- coding: utf-8 -*-
"""Shared layout for the permanent joint latent types.

Public type IDs are one-based to preserve the model's existing ``em_type``
arguments and ``_em{type_id}`` output filenames. Component indicators are
zero/one because they multiply the type shifts estimated by the auxiliary EM.
"""

import numpy as np


TYPE_COMPONENTS = np.asarray(
    [
        (school, grant, transfer, loan)
        for school in (0, 1)
        for grant in (0, 1)
        for transfer in (0, 1)
        for loan in (0, 1)
    ],
    dtype=np.int64,
)
TYPE_NAMES = np.asarray(
    [
        f"S{school}G{grant}T{transfer}L{loan}"
        for school, grant, transfer, loan in TYPE_COMPONENTS
    ]
)

TYPE_SCHOOL = TYPE_COMPONENTS[:, 0]
TYPE_GRANT = TYPE_COMPONENTS[:, 1]
TYPE_TRANSFER = TYPE_COMPONENTS[:, 2]
TYPE_LOAN = TYPE_COMPONENTS[:, 3]
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
    """Return ``(school, grant, transfer, loan)`` for ``type_id``."""
    school, grant, transfer, loan = TYPE_COMPONENTS[type_index(type_id)]
    return int(school), int(grant), int(transfer), int(loan)


def sgt_index(type_id):
    """Return the zero-based S x G x T cell after collapsing loan type."""
    school, grant, transfer, _ = type_components(type_id)
    return 4 * school + 2 * grant + transfer


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


def validate_saved_layout(
    type_names, type_school, type_grant, type_transfer, type_loan
):
    """Ensure a saved EM artifact uses the structural model's type ordering."""
    saved = (
        np.asarray(type_names).astype(str),
        np.asarray(type_school, dtype=np.int64),
        np.asarray(type_grant, dtype=np.int64),
        np.asarray(type_transfer, dtype=np.int64),
        np.asarray(type_loan, dtype=np.int64),
    )
    expected = (
        TYPE_NAMES.astype(str),
        TYPE_SCHOOL,
        TYPE_GRANT,
        TYPE_TRANSFER,
        TYPE_LOAN,
    )
    labels = (
        "type_names",
        "type_school",
        "type_grant",
        "type_transfer",
        "type_loan",
    )
    for label, observed, target in zip(labels, saved, expected):
        if not np.array_equal(observed, target):
            raise ValueError(
                f"Saved EM {label} does not match the shared latent-type layout."
            )
    return True
