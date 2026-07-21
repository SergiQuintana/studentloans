"""Debug-only validation and failure snapshots for structural estimation.

This module is imported only by the explicit ``*_debug`` entry points.  The
production Bellman and initial-CCP functions do not import it, branch on it, or
pay any runtime cost for these checks.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


class DebugValidationError(RuntimeError):
    """Raised after a fatal diagnostic has been safely written to disk."""


@dataclass(frozen=True)
class DebugConfig:
    output_dir: str
    fail_fast: bool = True
    max_failures: int = 20
    trace_draws: bool = True
    verify_saved: bool = True


def _json_value(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def array_summary(array: Any) -> dict[str, Any]:
    array = np.asarray(array)
    result = {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "size": int(array.size),
    }
    if not np.issubdtype(array.dtype, np.number):
        return result
    finite = np.isfinite(array)
    result.update(
        finite=int(np.count_nonzero(finite)),
        nan=int(np.count_nonzero(np.isnan(array))),
        positive_inf=int(np.count_nonzero(np.isposinf(array))),
        negative_inf=int(np.count_nonzero(np.isneginf(array))),
        finite_min=float(np.min(array[finite])) if np.any(finite) else None,
        finite_max=float(np.max(array[finite])) if np.any(finite) else None,
    )
    return result


def first_bad_index(mask: Any):
    mask = np.asarray(mask)
    if mask.ndim == 0:
        return () if bool(mask) else None
    coordinates = np.argwhere(mask)
    return tuple(int(value) for value in coordinates[0]) if coordinates.size else None


def vjt_problems(array: Any) -> list[dict[str, Any]]:
    """Classify fatal VJT cells while allowing intentional ``-inf`` choices."""
    array = np.asarray(array)
    problems = []
    for reason, mask in (
        ("vjt_nan", np.isnan(array)),
        ("vjt_positive_inf", np.isposinf(array)),
    ):
        index = first_bad_index(mask)
        if index is not None:
            problems.append({"reason": reason, "index": index})
    if array.ndim >= 2:
        no_finite_choice = ~np.any(np.isfinite(array), axis=-1)
        index = first_bad_index(no_finite_choice)
        if index is not None:
            problems.append({"reason": "no_finite_choice", "index": index})
    return problems


def finite_problems(array: Any, name: str) -> list[dict[str, Any]]:
    array = np.asarray(array)
    index = first_bad_index(~np.isfinite(array))
    return [] if index is None else [{"reason": f"{name}_nonfinite", "index": index}]


def ccp_problems(array: Any) -> list[dict[str, Any]]:
    array = np.asarray(array)
    checks = (
        ("ccp_nonfinite", ~np.isfinite(array)),
        ("ccp_not_strictly_positive", array <= 0.0),
        ("ccp_above_one", array > 1.0),
    )
    problems = []
    for reason, mask in checks:
        index = first_bad_index(mask)
        if index is not None:
            problems.append({"reason": reason, "index": index})
    return problems


class DebugRecorder:
    """One writer per (stage, type, invariant-state) multiprocessing task."""

    def __init__(self, config: DebugConfig, stage: str, type_id: int, x1_index: int):
        self.config = config
        self.stage = str(stage)
        self.type_id = int(type_id)
        self.x1_index = int(x1_index)
        self.failures: list[dict[str, Any]] = []
        self.checks = 0
        self.started = time.time()
        self.directory = (
            Path(config.output_dir)
            / self.stage
            / f"type_{self.type_id:02d}"
            / f"x1_{self.x1_index:03d}"
        )
        self.directory.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        reason: str,
        metadata: dict[str, Any],
        arrays: dict[str, Any] | None = None,
        fatal: bool = True,
    ) -> None:
        if len(self.failures) >= self.config.max_failures:
            if fatal and self.config.fail_fast:
                raise DebugValidationError(
                    f"{reason}; diagnostic limit already reached in {self.directory}"
                )
            return
        number = len(self.failures) + 1
        stem = f"failure_{number:04d}_{reason}"
        entry = {
            "reason": reason,
            "fatal": bool(fatal),
            "stage": self.stage,
            "type_id": self.type_id,
            "x1_index": self.x1_index,
            **metadata,
        }
        if arrays:
            entry["arrays"] = {
                name: array_summary(value) for name, value in arrays.items()
            }
            np.savez_compressed(
                self.directory / f"{stem}.npz",
                **{name: np.asarray(value) for name, value in arrays.items()},
            )
        (self.directory / f"{stem}.json").write_text(
            json.dumps(_json_value(entry), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        self.failures.append(entry)
        self._write_summary()
        if fatal and self.config.fail_fast:
            location = ", ".join(
                f"{key}={metadata[key]}"
                for key in ("period", "x2_index", "choice_index", "debt_index")
                if key in metadata
            )
            raise DebugValidationError(
                f"{reason} ({location}); snapshot: {self.directory / stem}"
            )

    def check(
        self,
        name: str,
        array: Any,
        problems: list[dict[str, Any]],
        metadata: dict[str, Any],
        arrays: dict[str, Any] | None = None,
    ) -> bool:
        self.checks += 1
        if not problems:
            return True
        first = problems[0]
        index = tuple(first.get("index", ()))
        full_metadata = {**metadata, "array": name, "bad_index": index}
        if index:
            try:
                full_metadata["bad_value"] = np.asarray(array)[index]
            except IndexError:
                pass
        self.record(first["reason"], full_metadata, arrays or {name: array})
        return False

    def note(self, reason: str, metadata: dict[str, Any], arrays=None) -> None:
        self.record(reason, metadata, arrays=arrays, fatal=False)

    def _write_summary(self):
        summary = {
            "pid": os.getpid(),
            "stage": self.stage,
            "type_id": self.type_id,
            "x1_index": self.x1_index,
            "checks": self.checks,
            "failures": len(self.failures),
            "elapsed_seconds": time.time() - self.started,
            "failure_reasons": [item["reason"] for item in self.failures],
        }
        (self.directory / "summary.json").write_text(
            json.dumps(_json_value(summary), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def finalize(self) -> dict[str, Any]:
        self._write_summary()
        return {
            "stage": self.stage,
            "type_id": self.type_id,
            "x1_index": self.x1_index,
            "checks": self.checks,
            "failures": len(self.failures),
            "output_dir": str(self.directory),
        }
