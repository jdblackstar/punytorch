"""Numerical comparison records shared by all verification classes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _plain_array(value: np.ndarray) -> Any:
    def safe(item):
        if isinstance(item, list):
            return [safe(value) for value in item]
        if isinstance(item, float) and not np.isfinite(item):
            if np.isnan(item):
                return "NaN"
            return "Infinity" if item > 0 else "-Infinity"
        return item

    return safe(np.asarray(value).tolist())


def _finite_or_none(value: float) -> float | None:
    return value if np.isfinite(value) else None


@dataclass(frozen=True)
class Comparison:
    kind: str
    target: str
    passed: bool
    expected: Any
    observed: Any
    expected_shape: tuple[int, ...]
    observed_shape: tuple[int, ...]
    max_absolute_error: float
    max_relative_error: float
    atol: float
    rtol: float
    first_failing_index: tuple[int, ...] | None
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "target": self.target,
            "passed": self.passed,
            "expected": self.expected,
            "observed": self.observed,
            "expected_shape": list(self.expected_shape),
            "observed_shape": list(self.observed_shape),
            "max_absolute_error": _finite_or_none(self.max_absolute_error),
            "max_relative_error": _finite_or_none(self.max_relative_error),
            "atol": self.atol,
            "rtol": self.rtol,
            "first_failing_index": (list(self.first_failing_index) if self.first_failing_index is not None else None),
            "detail": self.detail,
        }


def compare_arrays(
    *,
    kind: str,
    target: str,
    expected,
    observed,
    atol: float,
    rtol: float,
    detail: str = "",
) -> Comparison:
    expected_array = np.asarray(expected, dtype=np.float64)
    observed_array = np.asarray(observed, dtype=np.float64)
    if expected_array.shape != observed_array.shape:
        return Comparison(
            kind=kind,
            target=target,
            passed=False,
            expected=_plain_array(expected_array),
            observed=_plain_array(observed_array),
            expected_shape=expected_array.shape,
            observed_shape=observed_array.shape,
            max_absolute_error=float("inf"),
            max_relative_error=float("inf"),
            atol=atol,
            rtol=rtol,
            first_failing_index=None,
            detail=detail or "shape mismatch",
        )

    absolute = np.abs(observed_array - expected_array)
    relative = absolute / np.maximum(np.abs(expected_array), max(atol, np.finfo(np.float64).eps))
    close = np.isclose(observed_array, expected_array, atol=atol, rtol=rtol, equal_nan=False)
    failing = np.argwhere(~close)
    first_failing_index = tuple(int(value) for value in failing[0]) if failing.size else None
    return Comparison(
        kind=kind,
        target=target,
        passed=bool(np.all(close)),
        expected=_plain_array(expected_array),
        observed=_plain_array(observed_array),
        expected_shape=expected_array.shape,
        observed_shape=observed_array.shape,
        max_absolute_error=float(np.max(absolute)) if absolute.size else 0.0,
        max_relative_error=float(np.max(relative)) if relative.size else 0.0,
        atol=atol,
        rtol=rtol,
        first_failing_index=first_failing_index,
        detail=detail,
    )
