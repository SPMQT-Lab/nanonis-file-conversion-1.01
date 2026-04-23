"""
Format writers that consume :class:`probeflow.scan.Scan` objects.

Each submodule exposes a single ``write_<format>(scan, out_path, ...)`` entry
point.  Phase 1 ships with ``write_sxm`` and ``write_png`` — the list grows as
new output formats are added in later phases.
"""

from probeflow.writers.sxm import write_sxm
from probeflow.writers.png import write_png

__all__ = ["write_sxm", "write_png"]
