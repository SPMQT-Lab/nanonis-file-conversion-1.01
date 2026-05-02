"""Compatibility wrapper for :mod:`probeflow.io.converters.createc_dat_to_png`."""

from probeflow.io.converters import createc_dat_to_png as _impl

globals().update({
    name: value
    for name, value in vars(_impl).items()
    if not (name.startswith("__") and name.endswith("__"))
})
