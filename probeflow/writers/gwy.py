"""Compatibility wrapper for :mod:`probeflow.io.writers.gwy`."""

from probeflow.io.writers import gwy as _impl

globals().update({
    name: value
    for name, value in vars(_impl).items()
    if not (name.startswith("__") and name.endswith("__"))
})
