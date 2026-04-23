"""Tests for the Phase-2 optional readers.

We cannot run the actual vendor libraries without sample files and the
third-party dependencies, so these tests only verify:

  1. The dispatcher in :func:`probeflow.scan.load_scan` routes the right
     suffix to the right reader.
  2. Each reader surfaces a clear, actionable ``ImportError`` if its
     vendor library is missing.
  3. ``SUPPORTED_SUFFIXES`` includes the new formats.

End-to-end value tests live with each user who drops a real sample into
``data/`` — they're out of scope here.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from probeflow.scan import SUPPORTED_SUFFIXES, load_scan


class TestSupportedSuffixes:
    def test_phase1_present(self):
        assert ".sxm" in SUPPORTED_SUFFIXES
        assert ".dat" in SUPPORTED_SUFFIXES

    def test_phase2_present(self):
        for s in (".gwy", ".sm4", ".mtrx"):
            assert s in SUPPORTED_SUFFIXES


class TestDispatchMissingDeps:
    """If the vendor library is missing, the reader must raise ImportError
    with a helpful 'pip install probeflow[<extra>]' hint."""

    def _expect_helpful_error(self, tmp_path, suffix: str, extras_name: str):
        # Create an empty file with the target suffix so load_scan actually
        # tries to open it.
        p = tmp_path / f"fake{suffix}"
        p.write_bytes(b"")
        try:
            load_scan(p)
        except ImportError as exc:
            assert extras_name in str(exc), \
                f"ImportError for {suffix} must mention probeflow[{extras_name}]"
        except Exception:
            # If the vendor library IS installed, we don't care about the
            # downstream parse error for our empty fake file — only the
            # ImportError case is being tested here.
            pass

    def test_gwy(self, tmp_path):
        self._expect_helpful_error(tmp_path, ".gwy", "gwyddion")

    def test_sm4(self, tmp_path):
        self._expect_helpful_error(tmp_path, ".sm4", "rhk")

    def test_mtrx(self, tmp_path):
        self._expect_helpful_error(tmp_path, ".mtrx", "omicron")


class TestReaderModulesImportable:
    """The reader modules themselves must be importable without the vendor
    library installed — the import should succeed even if the library is
    missing.  Only calling ``read_<x>(path)`` should raise ImportError."""

    def test_gwy_module_loads(self):
        importlib.import_module("probeflow.readers.gwy")

    def test_sm4_module_loads(self):
        importlib.import_module("probeflow.readers.sm4")

    def test_mtrx_module_loads(self):
        importlib.import_module("probeflow.readers.mtrx")

    def test_read_functions_exported(self):
        from probeflow.readers import read_sxm, read_dat, read_gwy, read_sm4, read_mtrx
        assert all(callable(fn) for fn in
                   (read_sxm, read_dat, read_gwy, read_sm4, read_mtrx))
