"""Plugin registry foundation for future ProbeFlow extensions.

Architectural role
------------------
Plugins should declare parser, transformation, measurement, and writer
operations with enough metadata for CLI, GUI, and provenance wrappers to use the
same registry. This supports the intended Session -> Probe -> Scan/Spectrum ->
ScanGraph architecture by letting new operations be discovered once instead of
being hand-wired into several scripts.

Boundary rules
--------------
Keep plugin metadata, registration, manifests, and adapters here. Do not move
current processing kernels into plugins during cleanup, and do not define graph
nodes here. GUI and CLI code should eventually discover operations from this
registry rather than editing multiple command and panel files.
"""

from probeflow.plugins.api import PluginOperation, PluginSpec
from probeflow.plugins.registry import PluginRegistry

__all__ = ["PluginOperation", "PluginSpec", "PluginRegistry"]
