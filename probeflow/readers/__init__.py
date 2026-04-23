"""
Vendor-specific readers that produce :class:`probeflow.scan.Scan` objects.

Each submodule exposes a single ``read_<format>(path) -> Scan`` entry point.
The unified dispatcher is :func:`probeflow.scan.load_scan`.
"""

from probeflow.readers.sxm import read_sxm
from probeflow.readers.dat import read_dat

__all__ = ["read_sxm", "read_dat"]
