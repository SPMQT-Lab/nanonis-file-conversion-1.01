"""Write a :class:`probeflow.scan.Scan` to a Gwyddion ``.gwy`` file.

Uses the optional ``gwyfile`` library.  Install with::

    pip install probeflow[gwyddion]

Every plane in the Scan becomes a separate Gwyddion data channel, with
physical dimensions and value units taken from ``Scan.scan_range_m`` and
``Scan.plane_units``.
"""

from __future__ import annotations

from pathlib import Path


_INSTALL_HINT = (
    "Writing Gwyddion .gwy files requires the 'gwyfile' package.\n"
    "Install it via:  pip install probeflow[gwyddion]"
)


def write_gwy(scan, out_path) -> None:
    try:
        import gwyfile  # type: ignore
        from gwyfile.objects import GwyContainer, GwyDataField  # type: ignore
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise ImportError(_INSTALL_HINT) from exc

    w_m, h_m = scan.scan_range_m
    if w_m <= 0:
        w_m = 1.0
    if h_m <= 0:
        h_m = 1.0

    container = GwyContainer()
    for idx, plane in enumerate(scan.planes):
        unit = scan.plane_units[idx] if idx < len(scan.plane_units) else ""
        name = scan.plane_names[idx] if idx < len(scan.plane_names) else f"plane {idx}"

        df = GwyDataField(
            plane.astype("<f8"),
            xreal=float(w_m),
            yreal=float(h_m),
            si_unit_xy="m",
            si_unit_z=unit or "",
        )
        container[f"/{idx}/data"] = df
        container[f"/{idx}/data/title"] = name

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gwyfile.save(container, str(out_path))
