"""Compatibility wrapper for :mod:`probeflow.gui.features.tv`."""

from probeflow.gui.features import tv as _impl

globals().update({
    name: value
    for name, value in vars(_impl).items()
    if not (name.startswith("__") and name.endswith("__"))
})
