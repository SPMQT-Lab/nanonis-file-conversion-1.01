"""In-process plugin registry foundation.

The registry is the future single discovery point for parser, transformation,
measurement, and writer operations. GUI panels, CLI commands, and provenance
wrappers should eventually ask the registry what is available instead of
requiring a new plugin to be added to several unrelated scripts.

Keep this module about registration and lookup only. Graph nodes belong in
``probeflow.provenance``; operation implementations belong in ``io``,
``processing``, ``analysis``, or plugin packages.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from probeflow.plugins.api import PluginOperation, PluginSpec


@dataclass
class PluginRegistry:
    """Small registry for plugin specs and operations."""

    specs: dict[str, PluginSpec] = field(default_factory=dict)

    def register(self, spec: PluginSpec) -> None:
        self.specs[spec.name] = spec

    def operations(self, *, kind: str | None = None) -> list[PluginOperation]:
        ops: list[PluginOperation] = []
        for spec in self.specs.values():
            for op in spec.operations:
                if kind is None or op.kind == kind:
                    ops.append(op)
        return ops
