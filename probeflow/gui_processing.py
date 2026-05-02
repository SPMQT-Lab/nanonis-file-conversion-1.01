"""Compatibility wrapper for :mod:`probeflow.processing.gui_adapter`."""

from probeflow.processing import gui_adapter as _impl

globals().update({
    name: value
    for name, value in vars(_impl).items()
    if not (name.startswith("__") and name.endswith("__"))
})
