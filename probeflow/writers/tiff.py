"""Write a :class:`probeflow.scan.Scan` plane to a TIFF file.

Two output modes:

* ``mode='float'``  — 32-bit float TIFF preserving full physical-unit values.
  Best for further scientific analysis (ImageJ, Gwyddion).
* ``mode='uint16'`` — 16-bit grayscale TIFF linearly clipped to
  ``[clip_low, clip_high]`` percentiles.  Best for presentations / display.

Resolution metadata (``XResolution`` / ``YResolution``) is populated in
pixels-per-centimetre when the scan range is known, so downstream tools can
reconstruct the physical size.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def write_tiff(
    scan,
    out_path,
    plane_idx: int = 0,
    *,
    mode: str = "float",
    clip_low: float = 1.0,
    clip_high: float = 99.0,
) -> None:
    if plane_idx < 0 or plane_idx >= scan.n_planes:
        raise ValueError(
            f"plane_idx={plane_idx} out of range for Scan with "
            f"{scan.n_planes} plane(s)"
        )

    arr = scan.planes[plane_idx]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    w_m, h_m = scan.scan_range_m
    Ny, Nx = arr.shape

    # pixels-per-centimetre
    xres_ppcm = (Nx / (w_m * 100.0)) if w_m > 0 else None
    yres_ppcm = (Ny / (h_m * 100.0)) if h_m > 0 else None

    kwargs = {"format": "TIFF"}
    if xres_ppcm is not None and yres_ppcm is not None:
        # resolution is (xres, yres), unit=3 means dots per centimetre
        kwargs["dpi"] = (xres_ppcm * 2.54, yres_ppcm * 2.54)  # dots per inch

    if mode == "float":
        img = Image.fromarray(np.ascontiguousarray(arr, dtype=np.float32),
                              mode="F")
    elif mode == "uint16":
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            raise ValueError("Plane contains no finite values.")
        vmin = float(np.percentile(finite, clip_low))
        vmax = float(np.percentile(finite, clip_high))
        if vmax <= vmin:
            vmin, vmax = float(finite.min()), float(finite.max())
        if vmax <= vmin:
            vmax = vmin + 1.0
        safe = np.where(np.isfinite(arr), arr, vmin).astype(np.float64)
        u16 = np.clip((safe - vmin) / (vmax - vmin) * 65535.0, 0, 65535).astype(np.uint16)
        # Pillow 13 dropped the ``mode`` kwarg for uint16 — use dtype-driven
        # conversion instead: fromarray(uint16) is auto-interpreted as "I;16".
        img = Image.fromarray(u16)
    else:
        raise ValueError(f"Unknown TIFF mode {mode!r} (expected 'float' or 'uint16')")

    img.save(out_path, **kwargs)
