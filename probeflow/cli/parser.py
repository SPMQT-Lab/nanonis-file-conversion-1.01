"""Compatibility location for the ProbeFlow CLI parser builder."""

from probeflow.cli._legacy import _build_parser, main

__all__ = ["_build_parser", "main"]
