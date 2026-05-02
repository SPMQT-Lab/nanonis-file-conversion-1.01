"""Minimal plugin-facing types.

Architectural role
------------------
Plugin operations map directly to the intended operation classes: parser,
transformation, measurement, and writer. Future provenance-aware adapters can
use these records to connect plugin callables to ``ScanGraph`` operation nodes
without hard-coding each operation in both CLI and GUI layers.

Boundary rules
--------------
This cleanup pass intentionally keeps the API small and does not migrate
existing processing functions into plugins. Do not define provenance node
dataclasses or GUI/CLI behavior here; this package should describe operations
and callables, not own their presentation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal

OperationKind = Literal["parser", "transformation", "measurement", "writer"]


@dataclass(frozen=True)
class PluginOperation:
    """Description of one operation supplied by a plugin."""

    name: str
    kind: OperationKind
    version: str
    function: Callable[..., Any]
    input_types: tuple[str, ...] = ()
    output_types: tuple[str, ...] = ()
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PluginSpec:
    """Description of a ProbeFlow plugin package."""

    name: str
    version: str
    operations: tuple[PluginOperation, ...] = ()
