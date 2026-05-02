"""Compatibility wrapper for :mod:`probeflow.gui.models`."""

from probeflow.gui import models as _impl

globals().update({
    name: value
    for name, value in vars(_impl).items()
    if not (name.startswith("__") and name.endswith("__"))
})
