"""Compatibility wrapper for :mod:`probeflow.io.writers.csv`."""

from probeflow.io.writers import csv as _impl

globals().update({
    name: value
    for name, value in vars(_impl).items()
    if not (name.startswith("__") and name.endswith("__"))
})
