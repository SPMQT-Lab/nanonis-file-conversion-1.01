"""Reader for Omicron / Scienta Omicron Matrix ``.Z_mtrx`` topography files.

Matrix data is split across multiple files:
  * One ``.mtrx`` per experiment (the parameter/index file)
  * Many ``_mtrx`` payload files (e.g. ``default_0001.Z_mtrx``) — one per
    acquired image + channel

This reader opens a payload file directly via the third-party
``access2thematrix`` library.  Install it with::

    pip install probeflow[omicron]

The wrapper supports topography images (Z fwd / Z bwd / I fwd / I bwd when
present).  Spectroscopy cubes are not handled here — use an Omicron-specific
workflow.

Notes
-----
* The library's ``MtrxData.volume_scan_from_file`` returns a dict of
  ``(2D ndarray, TraceData)`` tuples keyed by direction name
  (``'trace up'``, ``'retrace up'``, ``'trace down'``, ``'retrace down'``).
* Heights are already in metres and currents in amperes; no DAC scaling.
* Orientation: Omicron stores rows bottom-to-top for "up" scans, so we flip
  those vertically to give a canonical top-left origin.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np

from probeflow.scan import Scan


_INSTALL_HINT = (
    "Reading Omicron Matrix files requires the 'access2thematrix' package.\n"
    "Install it via:  pip install probeflow[omicron]"
)


def read_mtrx(path) -> Scan:
    """Load an Omicron Matrix payload file (``.*_mtrx``) into a Scan."""
    try:
        import access2thematrix  # type: ignore
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise ImportError(_INSTALL_HINT) from exc

    path = Path(path)
    mtrx = access2thematrix.MtrxData()
    traces, message = mtrx.open(str(path))
    if not traces:
        raise ValueError(f"{path.name}: no traces found ({message!r})")

    # Preferred order matching ProbeFlow's canonical layout.
    preferred = ["trace up", "retrace up", "trace down", "retrace down"]
    ordered_keys = [k for k in preferred if k in traces]
    # Append anything else so we don't silently lose channels.
    for k in traces:
        if k not in ordered_keys:
            ordered_keys.append(k)

    planes: List[np.ndarray] = []
    names: List[str] = []
    units: List[str] = []
    synthetic = [False] * len(ordered_keys)

    width_m = 0.0
    height_m = 0.0
    header: dict = {}

    for idx, key in enumerate(ordered_keys):
        trace, tdata = traces[key]
        arr = np.asarray(trace, dtype=np.float64)
        # Orient "up" scans top-left-origin.
        if "up" in key:
            arr = np.flipud(arr)
        # Backward / retrace planes need to be mirrored so they share the
        # forward scan direction.
        if "retrace" in key:
            arr = np.fliplr(arr)
        planes.append(arr)
        names.append(f"Z ({key})")
        units.append("m")

        # Pull geometry + metadata from the first trace only; Matrix traces
        # in the same file share these values.
        if idx == 0:
            width_m = float(getattr(tdata, "width", 0.0))
            height_m = float(getattr(tdata, "height", 0.0))
            for attr in ("angle", "x_offset", "y_offset", "width", "height",
                          "x_points", "y_points"):
                val = getattr(tdata, attr, None)
                if val is not None:
                    header[attr] = val

    return Scan(
        planes=planes,
        plane_names=names,
        plane_units=units,
        plane_synthetic=synthetic,
        header=header,
        scan_range_m=(width_m, height_m),
        source_path=path,
        source_format="mtrx",
    )
