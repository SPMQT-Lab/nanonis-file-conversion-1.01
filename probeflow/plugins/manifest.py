"""Helpers for future plugin manifests.

Manifests should describe operation metadata for parser, transformation,
measurement, and writer plugins so the CLI, GUI, and provenance layer can share
one source of truth. Keep manifest serialization here; do not add operation
implementations, graph node dataclasses, or UI behavior.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from probeflow.plugins.api import PluginSpec


def manifest_from_spec(spec: PluginSpec) -> dict[str, Any]:
    """Return a JSON-compatible manifest dictionary for a plugin spec."""

    return asdict(spec)
