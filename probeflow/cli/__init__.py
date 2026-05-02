"""Command-line orchestration package for ProbeFlow.

Architectural role
------------------
The CLI is an orchestration layer over the intended Session -> Probe ->
Scan/Spectrum -> ScanGraph architecture. Commands should call parser,
transformation, measurement, writer, provenance, and future plugin-registry
services without owning those concepts themselves.

Future cleanup
--------------
The current command implementation is parked in :mod:`probeflow.cli._legacy`
for compatibility with existing private imports. Continue moving command
runners into ``commands/`` and shared processing wrappers into
``processing_ops.py``. New commands should call canonical package APIs or the
future plugin registry rather than reaching into GUI modules.

Do not add ``Scan``/``Spectrum`` model definitions, graph node dataclasses,
numerical kernels, vendor parser logic, or GUI widgets here.
"""

from probeflow.cli import _legacy as _impl

globals().update({
    name: value
    for name, value in vars(_impl).items()
    if not (name.startswith("__") and name.endswith("__"))
})


def _sync_legacy_overrides() -> None:
    for name in vars(_impl):
        if name == "main":
            continue
        if name.startswith("__") and name.endswith("__"):
            continue
        if name in globals() and globals()[name] is not getattr(_impl, name):
            setattr(_impl, name, globals()[name])


def main(argv=None) -> int:
    """Run the CLI while preserving monkeypatch compatibility on this package."""

    _sync_legacy_overrides()
    return _impl.main(argv)


__all__ = [name for name in globals() if not name.startswith("__")]
