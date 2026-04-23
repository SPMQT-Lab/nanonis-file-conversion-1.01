"""Reader for RHK ``.sm4`` topography files via the ``spym`` library.

The ``.sm4`` container can hold images, line cuts, and 1-D spectra in the
same file.  This reader keeps only the 2-D topography / current images and
builds a :class:`probeflow.scan.Scan`.  Spectroscopy objects inside the same
file are ignored — use an RHK-specific workflow for those.

Install the optional dependency::

    pip install probeflow[rhk]
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np

from probeflow.scan import Scan


_INSTALL_HINT = (
    "Reading RHK .sm4 files requires the 'spym' package.\n"
    "Install it via:  pip install probeflow[rhk]"
)


def read_sm4(path) -> Scan:
    """Load an RHK ``.sm4`` file into a Scan.

    The ``spym`` API returns an ``xarray.Dataset`` with one DataArray per
    channel.  We keep 2-D arrays that look like topography / current and
    discard 1-D spectra.
    """
    try:
        import spym  # type: ignore
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise ImportError(_INSTALL_HINT) from exc

    path = Path(path)
    ds = spym.load(str(path))
    if ds is None:
        raise ValueError(f"{path}: spym.load returned no data")

    planes: List[np.ndarray] = []
    names: List[str] = []
    units: List[str] = []

    width_m = 0.0
    height_m = 0.0
    header: dict = {}

    for var_name in ds.data_vars:
        da = ds[var_name]
        if da.ndim != 2:
            continue
        arr = np.asarray(da.values, dtype=np.float64)
        planes.append(arr)

        # Friendly label: RHK names are terse — "Topography", "Current", …
        unit = str(da.attrs.get("units", "") or "")
        direction = str(da.attrs.get("direction", "") or "").lower()
        label = var_name if not direction else f"{var_name} ({direction})"
        names.append(label)
        units.append(unit)

        # First 2D variable is authoritative for scan geometry.
        if width_m == 0.0 or height_m == 0.0:
            # xarray stores axis coords with physical units on the dim values.
            for dim in da.dims:
                coord = da.coords.get(dim)
                if coord is None:
                    continue
                span = float(coord.max() - coord.min())
                if "x" in dim.lower() and width_m == 0.0:
                    width_m = span
                elif "y" in dim.lower() and height_m == 0.0:
                    height_m = span
            # Copy any instrument metadata the file carried.
            for k, v in da.attrs.items():
                if k not in header:
                    header[k] = v

    if not planes:
        raise ValueError(f"{path}: no 2-D image channels found in .sm4")

    synthetic = [False] * len(planes)

    return Scan(
        planes=planes,
        plane_names=names,
        plane_units=units,
        plane_synthetic=synthetic,
        header=header,
        scan_range_m=(width_m, height_m),
        source_path=path,
        source_format="sm4",
    )
