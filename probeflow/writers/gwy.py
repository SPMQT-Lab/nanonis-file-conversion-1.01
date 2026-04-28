"""Write a :class:`probeflow.scan.Scan` to a Gwyddion ``.gwy`` file.

This uses the optional ``gwyfile`` package to serialise a top-level
``GwyContainer`` with one ``GwyDataField`` for the requested scan plane.
"""

from __future__ import annotations

from pathlib import Path
import json

import numpy as np

from probeflow.common import check_overwrite
from probeflow.export_provenance import processing_state_from_history, processing_state_hash
from probeflow.scan_model import Scan


def _import_gwyfile():
    """Import the optional Gwyddion serialization helpers."""
    try:
        from gwyfile.objects import GwyContainer, GwyDataField
    except ImportError as exc:
        raise ImportError(
            "Writing .gwy files requires the optional 'gwyfile' package. "
            "Install it with `pip install gwyfile`."
        ) from exc
    return GwyContainer, GwyDataField


def _plane_meta(
    GwyContainer,
    scan: Scan,
    plane_idx: int,
    plane_name: str,
    plane_unit: str,
    visible: bool,
):
    """Build a small metadata container for the exported plane."""
    meta = GwyContainer()
    meta["ProbeFlow source path"] = (
        str(scan.source_path) if scan.source_path is not None else ""
    )
    meta["ProbeFlow source format"] = str(scan.source_format)
    meta["ProbeFlow plane index"] = int(plane_idx)
    meta["ProbeFlow plane name"] = str(plane_name)
    meta["ProbeFlow plane unit"] = str(plane_unit)
    meta["ProbeFlow visible on open"] = bool(visible)
    meta["ProbeFlow scan width (m)"] = float(scan.scan_range_m[0])
    meta["ProbeFlow scan height (m)"] = float(scan.scan_range_m[1])
    meta["ProbeFlow num planes"] = int(scan.n_planes)
    ps = processing_state_from_history(scan.processing_history)
    meta["ProbeFlow processing state"] = json.dumps(ps, sort_keys=True)
    meta["ProbeFlow processing state hash"] = processing_state_hash(ps)
    meta["ProbeFlow processing steps"] = int(len(ps.get("steps", [])))
    return meta


def write_gwy(
    scan: Scan,
    out_path,
    plane_idx: int = 0,
    *,
    include_meta: bool = True,
) -> None:
    """Write one plane of *scan* to a Gwyddion ``.gwy`` file.

    Parameters
    ----------
    scan:
        The ProbeFlow scan object to export.
    out_path:
        Destination path.
    plane_idx:
        Plane to mark visible when Gwyddion opens the file.
    include_meta:
        If true, add a small metadata container under ``/0/meta``.
    """
    out_path = Path(out_path)
    if scan.source_path is not None:
        check_overwrite(scan.source_path, out_path)
    if plane_idx < 0 or plane_idx >= scan.n_planes:
        raise ValueError(
            f"plane_idx={plane_idx} out of range for Scan with "
            f"{scan.n_planes} plane(s)"
        )

    GwyContainer, GwyDataField = _import_gwyfile()

    out_path.parent.mkdir(parents=True, exist_ok=True)

    container = GwyContainer()
    container.filename = str(out_path)

    width_m, height_m = scan.scan_range_m
    width_m = float(width_m)
    height_m = float(height_m)

    plane_name = (
        scan.plane_names[plane_idx]
        if plane_idx < len(scan.plane_names)
        else f"Plane {plane_idx}"
    )
    plane_unit = (
        scan.plane_units[plane_idx]
        if plane_idx < len(scan.plane_units)
        else ""
    )
    data = np.asarray(scan.planes[plane_idx], dtype=np.float64)
    field = GwyDataField(
        data,
        xreal=width_m,
        yreal=height_m,
        xoff=0.0,
        yoff=0.0,
        si_unit_xy="m",
        si_unit_z=plane_unit or None,
    )
    container["/0/data"] = field
    container["/0/data/title"] = str(plane_name)
    container["/0/data/visible"] = True
    if include_meta:
        container["/0/meta"] = _plane_meta(
            GwyContainer,
            scan,
            plane_idx,
            plane_name,
            plane_unit,
            True,
        )

    container.tofile(str(out_path))
