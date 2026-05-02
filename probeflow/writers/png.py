"""Compatibility wrapper for :mod:`probeflow.io.writers.png`."""

from probeflow.io.writers import png as _impl

globals().update({
    name: value
    for name, value in vars(_impl).items()
    if not (name.startswith("__") and name.endswith("__"))
})
