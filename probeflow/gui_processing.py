"""Apply GUI processing state to a Scan before export.

No Qt imports — this module can be tested without a running Qt event loop.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from probeflow.scan import Scan

# Keys in the GUI processing state dict that correspond to numeric data
# transforms (as opposed to display-only settings like grain overlays,
# colourmap, or clip percentiles).
NUMERIC_PROC_KEYS: tuple[str, ...] = (
    "remove_bad_lines",
    "align_rows",
    "bg_order",
    "facet_level",
    "smooth_sigma",
    "edge_method",
    "fft_mode",
)


def apply_processing_state_to_scan(
    scan: "Scan",
    proc_state: dict,
    *,
    plane_idx: int = 0,
) -> "Scan":
    """Apply GUI processing state to a Scan before export.

    Mutates scan.planes[plane_idx] in place and appends processing_history
    entries for each operation actually applied.  Display-only settings
    (grain detection overlay, colour map, clip percentiles) are ignored.

    Returns scan for convenience.
    """
    from probeflow import processing as _proc

    if plane_idx < 0 or plane_idx >= len(scan.planes):
        raise ValueError(
            f"plane_idx={plane_idx} out of range for Scan with "
            f"{len(scan.planes)} plane(s)"
        )

    a = scan.planes[plane_idx]

    def _record(op_name: str, params: dict) -> None:
        scan.processing_history.append({
            "op": op_name,
            "params": params,
            "timestamp": datetime.now().isoformat(),
        })

    if proc_state.get("remove_bad_lines"):
        a = _proc.remove_bad_lines(a)
        _record("remove_bad_lines", {"threshold_mad": 5.0})

    align = proc_state.get("align_rows")
    if align:
        a = _proc.align_rows(a, method=align)
        _record("align_rows", {"method": align})

    bg_order = proc_state.get("bg_order")
    if bg_order is not None:
        a = _proc.subtract_background(a, order=int(bg_order))
        _record("plane_bg", {"order": int(bg_order)})

    if proc_state.get("facet_level"):
        a = _proc.facet_level(a)
        _record("facet_level", {"threshold_deg": 3.0})

    smooth_sigma = proc_state.get("smooth_sigma")
    if smooth_sigma:
        a = _proc.gaussian_smooth(a, sigma_px=float(smooth_sigma))
        _record("smooth", {"sigma_px": float(smooth_sigma)})

    edge_method = proc_state.get("edge_method")
    if edge_method:
        sigma = float(proc_state.get("edge_sigma", 1.0))
        sigma2 = float(proc_state.get("edge_sigma2", 2.0))
        a = _proc.edge_detect(a, method=edge_method, sigma=sigma, sigma2=sigma2)
        _record("edge_detect", {"method": edge_method, "sigma": sigma, "sigma2": sigma2})

    fft_mode = proc_state.get("fft_mode")
    if fft_mode is not None:
        cutoff = float(proc_state.get("fft_cutoff", 0.10))
        window = str(proc_state.get("fft_window", "hanning"))
        a = _proc.fourier_filter(a, mode=fft_mode, cutoff=cutoff, window=window)
        _record("fourier_filter", {"mode": fft_mode, "cutoff": cutoff, "window": window})

    scan.planes[plane_idx] = a
    return scan
