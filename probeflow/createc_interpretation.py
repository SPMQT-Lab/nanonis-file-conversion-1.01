"""Compatibility wrapper for :mod:`probeflow.io.createc_interpretation`."""

from probeflow.io import createc_interpretation as _impl

globals().update({
    name: value
    for name, value in vars(_impl).items()
    if not (name.startswith("__") and name.endswith("__"))
})
