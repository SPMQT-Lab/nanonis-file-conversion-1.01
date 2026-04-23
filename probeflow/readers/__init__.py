"""
Vendor-specific readers that produce :class:`probeflow.scan.Scan` objects.

Each submodule exposes a single ``read_<format>(path) -> Scan`` entry point.
The unified dispatcher is :func:`probeflow.scan.load_scan`.

Phase 1 readers (always available): ``.sxm``, ``.dat``
Phase 2 readers (optional extras):   ``.gwy``, ``.sm4``, ``.mtrx``

Importing the Phase 2 submodules directly is safe — each raises a clear
``ImportError`` on call if its third-party dependency isn't installed.
"""

from probeflow.readers.sxm import read_sxm
from probeflow.readers.dat import read_dat
from probeflow.readers.gwy import read_gwy
from probeflow.readers.sm4 import read_sm4
from probeflow.readers.mtrx import read_mtrx

__all__ = ["read_sxm", "read_dat", "read_gwy", "read_sm4", "read_mtrx"]
