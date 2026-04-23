"""
Format writers that consume :class:`probeflow.scan.Scan` objects.

Phase 1 writers (always available): ``.sxm``, ``.png``
Phase 2 writers:
  * ``.pdf``  (matplotlib)
  * ``.tiff`` / ``.tif`` (Pillow)
  * ``.gwy``  (gwyfile — install ``probeflow[gwyddion]``)
  * ``.csv``  (numpy)

Use :func:`save_scan` for a suffix-driven "write anything" dispatcher.
"""

from pathlib import Path
from typing import Tuple

from probeflow.writers.sxm import write_sxm
from probeflow.writers.png import write_png
from probeflow.writers.pdf import write_pdf
from probeflow.writers.tiff import write_tiff
from probeflow.writers.gwy import write_gwy
from probeflow.writers.csv import write_csv

__all__ = [
    "write_sxm", "write_png", "write_pdf", "write_tiff", "write_gwy", "write_csv",
    "save_scan", "SUPPORTED_OUTPUT_SUFFIXES",
]


SUPPORTED_OUTPUT_SUFFIXES: Tuple[str, ...] = (
    ".sxm", ".png", ".pdf", ".tif", ".tiff", ".gwy", ".csv",
)


def save_scan(scan, out_path, plane_idx: int = 0, **kwargs) -> None:
    """Write ``scan`` to ``out_path``, dispatching on the output suffix.

    Supported suffixes are listed in :data:`SUPPORTED_OUTPUT_SUFFIXES`.
    Extra keyword arguments are forwarded to the per-format writer — see
    e.g. :func:`write_png` or :func:`write_pdf` for their options.
    """
    out_path = Path(out_path)
    suffix = out_path.suffix.lower()

    if suffix == ".sxm":
        write_sxm(scan, out_path)
    elif suffix == ".png":
        write_png(scan, out_path, plane_idx=plane_idx, **kwargs)
    elif suffix == ".pdf":
        write_pdf(scan, out_path, plane_idx=plane_idx, **kwargs)
    elif suffix in (".tif", ".tiff"):
        write_tiff(scan, out_path, plane_idx=plane_idx, **kwargs)
    elif suffix == ".gwy":
        write_gwy(scan, out_path)
    elif suffix == ".csv":
        write_csv(scan, out_path, plane_idx=plane_idx, **kwargs)
    else:
        raise ValueError(
            f"Unsupported output format {suffix!r}. "
            f"Supported: {', '.join(SUPPORTED_OUTPUT_SUFFIXES)}"
        )
