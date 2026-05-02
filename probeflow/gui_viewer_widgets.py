"""Compatibility wrapper for :mod:`probeflow.gui.viewer.widgets`."""

from probeflow.gui.viewer import widgets as _impl

globals().update({
    name: value
    for name, value in vars(_impl).items()
    if not (name.startswith("__") and name.endswith("__"))
})
