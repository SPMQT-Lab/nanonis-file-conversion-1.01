"""Compatibility wrapper for :mod:`probeflow.gui.browse`."""

from probeflow.gui import browse as _impl

globals().update({
    name: value
    for name, value in vars(_impl).items()
    if not (name.startswith("__") and name.endswith("__"))
})
