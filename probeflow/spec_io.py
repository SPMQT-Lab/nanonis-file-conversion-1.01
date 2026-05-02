"""Compatibility wrapper for :mod:`probeflow.io.spectroscopy`."""

from probeflow.io import spectroscopy as _impl

globals().update({
    name: value
    for name, value in vars(_impl).items()
    if not (name.startswith("__") and name.endswith("__"))
})
