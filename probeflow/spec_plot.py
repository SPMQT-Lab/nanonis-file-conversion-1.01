"""Compatibility wrapper for :mod:`probeflow.analysis.spec_plot`."""

from probeflow.analysis import spec_plot as _impl

globals().update({
    name: value
    for name, value in vars(_impl).items()
    if not (name.startswith("__") and name.endswith("__"))
})
