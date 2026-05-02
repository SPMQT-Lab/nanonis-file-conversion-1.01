"""Import compatibility checks for the package-layout cleanup."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest


def test_backend_imports_do_not_require_qt():
    import probeflow
    from probeflow import processing
    from probeflow.processing import align_rows
    from probeflow.processing_state import ProcessingState
    from probeflow.scan import Scan, load_scan

    assert probeflow.Scan is Scan
    assert callable(load_scan)
    assert callable(align_rows)
    assert ProcessingState.__name__ == "ProcessingState"
    assert processing.align_rows is align_rows
    assert "PySide6" not in sys.modules


def test_legacy_io_and_analysis_imports_remain_available():
    from probeflow.dat_png import dat_to_hdr_imgs
    from probeflow.dat_sxm import convert_dat_to_sxm
    from probeflow.features import segment_particles
    from probeflow.lattice import LatticeParams
    from probeflow.readers.dat import read_dat
    from probeflow.readers.sxm import read_sxm
    from probeflow.writers.json import write_json

    assert callable(dat_to_hdr_imgs)
    assert callable(convert_dat_to_sxm)
    assert callable(segment_particles)
    assert LatticeParams.__name__ == "LatticeParams"
    assert callable(read_dat)
    assert callable(read_sxm)
    assert callable(write_json)


def test_pure_gui_helpers_import_without_qt():
    import probeflow.gui_models as gui_models
    import probeflow.gui_processing as gui_processing
    import probeflow.gui_rendering as gui_rendering

    assert callable(gui_processing.processing_state_from_gui)
    assert gui_models.SxmFile.__name__ == "SxmFile"
    assert callable(gui_rendering.resolve_thumbnail_plane_index)
    assert "PySide6" not in sys.modules


def test_gui_compat_import_when_qt_available():
    pytest.importorskip("PySide6")

    from probeflow.gui import ImageViewerDialog, SxmFile, main

    assert ImageViewerDialog.__name__ == "ImageViewerDialog"
    assert SxmFile.__name__ == "SxmFile"
    assert callable(main)


def test_cli_import_path_remains_available():
    from probeflow.cli import main
    from probeflow.cli.processing_ops import _op_plane_bg

    assert callable(main)
    assert _op_plane_bg(1).name == "plane_bg"


def test_plugin_foundation_imports():
    from probeflow.plugins import PluginRegistry

    registry = PluginRegistry()
    assert registry.operations() == []


def test_spec_plot_private_compatibility_imports_remain_available():
    from probeflow.spec_plot import _parse_sxm_offset, spec_position_to_pixel

    assert callable(_parse_sxm_offset)
    assert callable(spec_position_to_pixel)


def test_top_level_modules_are_compatibility_shims():
    root = Path(__file__).resolve().parents[1] / "probeflow"
    allowed_long_public_shims = {"__init__.py", "dat_sxm.py"}
    too_large = []
    for path in root.glob("*.py"):
        if path.name in allowed_long_public_shims:
            continue
        line_count = len(path.read_text(encoding="utf-8").splitlines())
        if line_count > 20:
            too_large.append((path.name, line_count))

    assert too_large == []


def test_implementation_imports_use_canonical_package_paths():
    root = Path(__file__).resolve().parents[1] / "probeflow"
    root_shims = {
        "createc_interpretation.py",
        "gui_browse.py",
        "gui_features.py",
        "gui_models.py",
        "gui_processing.py",
        "gui_rendering.py",
        "gui_tv.py",
        "gui_viewer_widgets.py",
        "gui_workers.py",
    }
    forbidden = (
        "probeflow.createc_interpretation",
        "probeflow.gui_browse",
        "probeflow.gui_features",
        "probeflow.gui_models",
        "probeflow.gui_processing",
        "probeflow.gui_rendering",
        "probeflow.gui_tv",
        "probeflow.gui_viewer_widgets",
        "probeflow.gui_workers",
    )
    offenders = []
    for path in root.rglob("*.py"):
        if path.parent == root and path.name in root_shims:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in forbidden:
            if f"from {pattern}" in text or f"import {pattern}" in text:
                offenders.append((str(path.relative_to(root)), pattern))

    assert offenders == []


def test_graph_node_types_are_reserved_for_provenance():
    root = Path(__file__).resolve().parents[1] / "probeflow"
    forbidden_defs = (
        "class ImageNode",
        "class MeasurementNode",
        "class OperationNode",
        "class ArtifactNode",
        "class ScanGraph",
    )
    offenders = []
    for package in ("processing", "analysis"):
        for path in (root / package).rglob("*.py"):
            text = path.read_text(encoding="utf-8", errors="ignore")
            for marker in forbidden_defs:
                if marker in text:
                    offenders.append((str(path.relative_to(root)), marker))

    assert offenders == []
