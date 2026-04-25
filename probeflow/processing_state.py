"""ProcessingState — canonical representation of numerical processing choices.

This module separates *data-transforming* operations (bad-line removal, row
alignment, background subtraction, smoothing, FFT, edge detection) from
display-only settings (colormap, clip percentiles, vmin/vmax, grain overlay).

Typical call order
------------------
    state = ProcessingState(steps=[
        ProcessingStep("remove_bad_lines"),
        ProcessingStep("align_rows", {"method": "median"}),
        ProcessingStep("plane_bg", {"order": 1}),
    ])
    processed_arr = apply_processing_state(raw_arr, state)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ── Supported operations (must match probeflow.processing function names) ─────

_SUPPORTED_OPS: frozenset[str] = frozenset({
    "remove_bad_lines",
    "align_rows",
    "plane_bg",
    "facet_level",
    "smooth",
    "edge_detect",
    "fourier_filter",
    "fft_soft_border",
    "linear_undistort",
    "set_zero_point",
})


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ProcessingStep:
    """One numerical processing operation applied to scan data."""

    op: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessingState:
    """Ordered list of numerical processing steps.

    Represents operations that change the numerical image data.
    Does not include display-only settings such as colormap, vmin/vmax,
    percentile clipping, histogram state, or overlays.
    """

    steps: list[ProcessingStep] = field(default_factory=list)

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Serialise to a JSON-compatible dict.

        Example output::

            {
              "steps": [
                {"op": "align_rows", "params": {"method": "median"}},
                {"op": "plane_bg",   "params": {"order": 1}}
              ]
            }
        """
        return {
            "steps": [
                {"op": step.op, "params": dict(step.params)}
                for step in self.steps
            ]
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ProcessingState":
        """Deserialise from the dict produced by :meth:`to_dict`."""
        steps = []
        for item in data.get("steps", []):
            steps.append(ProcessingStep(
                op=str(item["op"]),
                params=dict(item.get("params", {})),
            ))
        return cls(steps=steps)


# ── Canonical apply function ──────────────────────────────────────────────────

def apply_processing_state(arr: np.ndarray, state: ProcessingState) -> np.ndarray:
    """Apply *state* steps in order to *arr*.

    Parameters
    ----------
    arr:
        Input 2-D numeric array (will not be mutated).
    state:
        Processing steps to apply.

    Returns
    -------
    np.ndarray of float64, same shape as *arr*.

    Raises
    ------
    ValueError
        If a step contains an unrecognised operation name.
    """
    # Always return a fresh float64 copy so raw Scan planes are never mutated.
    a = arr.astype(np.float64, copy=True)

    if not state.steps:
        return a

    import probeflow.processing as _proc

    for step in state.steps:
        p = step.params
        if step.op == "remove_bad_lines":
            a = _proc.remove_bad_lines(a)
        elif step.op == "align_rows":
            a = _proc.align_rows(a, method=p.get("method", "median"))
        elif step.op == "plane_bg":
            a = _proc.subtract_background(
                a,
                order=int(p.get("order", 1)),
                step_tolerance=bool(p.get("step_tolerance", False)),
            )
        elif step.op == "facet_level":
            a = _proc.facet_level(a)
        elif step.op == "smooth":
            a = _proc.gaussian_smooth(a, sigma_px=float(p.get("sigma_px", 1.0)))
        elif step.op == "edge_detect":
            a = _proc.edge_detect(
                a,
                method=p.get("method", "laplacian"),
                sigma=float(p.get("sigma", 1.0)),
                sigma2=float(p.get("sigma2", 2.0)),
            )
        elif step.op == "fourier_filter":
            a = _proc.fourier_filter(
                a,
                mode=p.get("mode", "low_pass"),
                cutoff=float(p.get("cutoff", 0.10)),
                window=str(p.get("window", "hanning")),
            )
        elif step.op == "fft_soft_border":
            a = _proc.fft_soft_border(
                a,
                mode=str(p.get("mode", "low_pass")),
                cutoff=float(p.get("cutoff", 0.10)),
                border_frac=float(p.get("border_frac", 0.12)),
            )
        elif step.op == "linear_undistort":
            a = _proc.linear_undistort(
                a,
                shear_x=float(p.get("shear_x", 0.0)),
                scale_y=float(p.get("scale_y", 1.0)),
            )
        elif step.op == "set_zero_point":
            a = _proc.set_zero_point(
                a,
                int(p.get("y_px", 0)),
                int(p.get("x_px", 0)),
                patch=int(p.get("patch", 1)),
            )
        else:
            raise ValueError(
                f"Unknown processing operation {step.op!r}. "
                f"Supported: {sorted(_SUPPORTED_OPS)}"
            )

    return a
