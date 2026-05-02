"""Compatibility wrapper for :mod:`probeflow.io.file_type`."""

from probeflow.io import file_type as _impl

globals().update({
    name: value
    for name, value in vars(_impl).items()
    if not (name.startswith("__") and name.endswith("__"))
})
