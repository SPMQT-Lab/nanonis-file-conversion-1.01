"""GUI-free image transformation operations.

Architectural role
------------------
``processing`` contains array-in/array-out numerical transformations for scan
images: flattening, row alignment, smoothing, FFT filters, edge detection, and
similar kernels. In the intended provenance architecture, these functions are
called by graph-aware adapters that record transformation operations from input
``ImageNode`` IDs to output ``ImageNode`` recipes.

Boundary rules
--------------
Keep this package focused on operation functions, state adapters, and thin
wrappers around existing kernels. Do not define ``ImageNode``,
``MeasurementNode``, ``OperationNode``, ``ArtifactNode``, or ``ScanGraph`` here;
those belong in ``probeflow.provenance``. Do not add GUI widgets, vendor
parsers, or writer implementations here.
"""

from probeflow.processing import image as _impl

globals().update({
    name: value
    for name, value in vars(_impl).items()
    if not (name.startswith("__") and name.endswith("__"))
})

__all__ = [
    name for name in globals()
    if not name.startswith("_")
]
