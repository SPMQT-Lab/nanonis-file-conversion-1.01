"""Reader and writer helpers for Gwyddion ``.gwy`` files (via the ``gwyfile`` lib).

Gwyddion's native format is open and well-documented.  The ``gwyfile`` Python
package exposes it as nested dicts.  Install the optional dependency::

    pip install probeflow[gwyddion]

A ``.gwy`` file holds any number of data channels (fields).  Each channel is
a 2-D array of doubles with physical units metadata attached.  This reader
keeps all channels it can find and exposes them as Scan planes.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np

from probeflow.scan import Scan


_INSTALL_HINT = (
    "Reading / writing Gwyddion .gwy files requires the 'gwyfile' package.\n"
    "Install it via:  pip install probeflow[gwyddion]"
)


def _require_gwyfile():
    try:
        import gwyfile  # type: ignore
        return gwyfile
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise ImportError(_INSTALL_HINT) from exc


def read_gwy(path) -> Scan:
    """Load a Gwyddion ``.gwy`` file into a Scan."""
    gwyfile = _require_gwyfile()
    path = Path(path)

    container = gwyfile.load(str(path))
    objs = gwyfile.util.get_datafields(container)
    if not objs:
        raise ValueError(f"{path}: no data channels found")

    planes: List[np.ndarray] = []
    names: List[str] = []
    units: List[str] = []

    width_m = 0.0
    height_m = 0.0
    header: dict = {}

    for key, datafield in objs.items():
        arr = np.asarray(datafield.data, dtype=np.float64)
        planes.append(arr)
        names.append(str(key))
        units.append(str(datafield.get("si_unit_z", {}).get("unitstr", "") or ""))
        if width_m == 0.0 and height_m == 0.0:
            width_m = float(datafield.get("xreal", 0.0) or 0.0)
            height_m = float(datafield.get("yreal", 0.0) or 0.0)
            # Stash dims/offsets in the header for downstream reference.
            for k in ("xres", "yres", "xoff", "yoff", "xreal", "yreal"):
                if k in datafield:
                    header[k] = datafield[k]

    synthetic = [False] * len(planes)

    return Scan(
        planes=planes,
        plane_names=names,
        plane_units=units,
        plane_synthetic=synthetic,
        header=header,
        scan_range_m=(width_m, height_m),
        source_path=path,
        source_format="gwy",
    )
