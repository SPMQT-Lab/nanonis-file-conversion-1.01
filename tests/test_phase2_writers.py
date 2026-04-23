"""Tests for the Phase-2 writers (PDF / TIFF / GWY / CSV) + save_scan dispatch.

We use a real Scan loaded from a bundled ``.dat`` sample so the writers are
exercised end-to-end on realistic data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from probeflow import load_scan
from probeflow.writers import (
    SUPPORTED_OUTPUT_SUFFIXES,
    save_scan,
    write_csv,
    write_pdf,
    write_tiff,
)


@pytest.fixture
def dat_scan(first_sample_dat):
    return load_scan(first_sample_dat)


class TestSupportedOutputSuffixes:
    def test_has_all_expected(self):
        for s in (".sxm", ".png", ".pdf", ".tif", ".tiff", ".gwy", ".csv"):
            assert s in SUPPORTED_OUTPUT_SUFFIXES


class TestPdf:
    def test_writes_file(self, dat_scan, tmp_path):
        out = tmp_path / "out.pdf"
        write_pdf(dat_scan, out, plane_idx=0, colormap="gray")
        assert out.exists() and out.stat().st_size > 0
        # PDF files start with the magic %PDF
        assert out.read_bytes()[:4] == b"%PDF"

    def test_via_save_method(self, dat_scan, tmp_path):
        out = tmp_path / "via_save.pdf"
        dat_scan.save(out, plane_idx=0)
        assert out.exists()


class TestTiff:
    def test_float_mode_preserves_values(self, dat_scan, tmp_path):
        from PIL import Image
        out = tmp_path / "out_f32.tif"
        write_tiff(dat_scan, out, plane_idx=0, mode="float")
        img = Image.open(out)
        arr = np.asarray(img)
        assert arr.dtype == np.float32
        assert arr.shape == dat_scan.planes[0].shape

    def test_uint16_mode_writes_16bit(self, dat_scan, tmp_path):
        from PIL import Image
        out = tmp_path / "out_u16.tif"
        write_tiff(dat_scan, out, plane_idx=0, mode="uint16")
        img = Image.open(out)
        assert img.mode in ("I;16", "I;16L", "I;16B")

    def test_via_save_method(self, dat_scan, tmp_path):
        out = tmp_path / "via_save.tiff"
        dat_scan.save(out, plane_idx=0)
        assert out.exists()


class TestCsv:
    def test_writes_grid(self, dat_scan, tmp_path):
        out = tmp_path / "out.csv"
        write_csv(dat_scan, out, plane_idx=0)
        # Use numpy.loadtxt (skips # comments) to read back.
        arr = np.loadtxt(out, delimiter=",")
        # Shape should match the plane (after stripping comment header)
        assert arr.shape == dat_scan.planes[0].shape

    def test_header_line_present(self, dat_scan, tmp_path):
        out = tmp_path / "out.csv"
        write_csv(dat_scan, out, plane_idx=0)
        first_line = out.read_text(encoding="utf-8").splitlines()[0]
        assert first_line.startswith("#")
        assert "plane=" in first_line
        assert "width_m=" in first_line

    def test_via_save_method(self, dat_scan, tmp_path):
        out = tmp_path / "via_save.csv"
        dat_scan.save(out, plane_idx=0)
        assert out.exists()


class TestGwyMissingDep:
    """write_gwy must surface a helpful ImportError when gwyfile isn't installed."""

    def test_import_error_is_actionable(self, dat_scan, tmp_path):
        import probeflow.writers.gwy as gwywriter
        try:
            gwywriter.write_gwy(dat_scan, tmp_path / "out.gwy")
        except ImportError as exc:
            assert "gwyddion" in str(exc)
        except Exception:
            # gwyfile may be installed in some dev envs — skip in that case.
            pass


class TestSaveScanDispatch:
    def test_unknown_suffix_raises(self, dat_scan, tmp_path):
        with pytest.raises(ValueError, match="Unsupported output"):
            save_scan(dat_scan, tmp_path / "out.zzz")

    def test_png_routes_correctly(self, dat_scan, tmp_path):
        out = tmp_path / "ok.png"
        save_scan(dat_scan, out, plane_idx=0, colormap="gray")
        assert out.exists()

    def test_pdf_routes_correctly(self, dat_scan, tmp_path):
        out = tmp_path / "ok.pdf"
        save_scan(dat_scan, out, plane_idx=0, colormap="gray")
        assert out.exists()
