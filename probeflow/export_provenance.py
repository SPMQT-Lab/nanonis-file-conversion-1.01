"""ExportProvenance — records where exported files came from and how.

The provenance chain is:
    source file -> Scan -> channel -> ProcessingState -> DisplayRangeState -> export file

All fields serialise to plain JSON via :meth:`ExportProvenance.to_dict`.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from probeflow.display_state import DisplayRangeState
    from probeflow.processing_state import ProcessingState
    from probeflow.scan import Scan


def _get_version() -> str | None:
    """Return the ProbeFlow package version string, or None if unavailable."""
    try:
        from probeflow import __version__
        if __version__:
            return str(__version__)
    except Exception:
        pass
    try:
        from importlib.metadata import version
        return version("probeflow")
    except Exception:
        return None


def _utc_now() -> str:
    """Return the current UTC time as an ISO-8601 string ending in Z."""
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True)
class ExportProvenance:
    """Immutable provenance record for one exported file.

    All values must be JSON-serialisable (strings, numbers, lists, dicts, None).
    Nested objects (processing_state, display_state) are stored pre-serialised
    so ``to_dict()`` never needs to know their internal structure.
    """

    source_file:        str | None
    source_format:      str | None
    item_type:          str                         # "scan" | "spectrum" | …
    channel_name:       str | None
    channel_index:      int | None
    array_shape:        tuple[int, int] | None       # (Ny, Nx)
    scan_range_m:       tuple[float, float] | None   # (width_m, height_m)
    units:              str | None
    processing_state:   dict[str, Any]               # ProcessingState.to_dict()
    display_state:      dict[str, Any]               # DisplayRangeState.to_dict()
    probeflow_version:  str | None
    export_timestamp:   str                          # ISO-8601 UTC

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Return a JSON-compatible dict of all provenance fields."""
        return {
            "source_file":       self.source_file,
            "source_format":     self.source_format,
            "item_type":         self.item_type,
            "channel_name":      self.channel_name,
            "channel_index":     self.channel_index,
            "array_shape":       list(self.array_shape) if self.array_shape else None,
            "scan_range_m":      list(self.scan_range_m) if self.scan_range_m else None,
            "units":             self.units,
            "processing_state":  self.processing_state,
            "display_state":     self.display_state,
            "probeflow_version": self.probeflow_version,
            "export_timestamp":  self.export_timestamp,
        }

    # ── Convenience constructor ───────────────────────────────────────────────

    @classmethod
    def from_scan_export(
        cls,
        scan: "Scan",
        *,
        channel_index: int | None = None,
        channel_name:  str | None = None,
        processing_state: "ProcessingState | None" = None,
        display_state:    "DisplayRangeState | None" = None,
        item_type: str = "scan",
    ) -> "ExportProvenance":
        """Construct provenance from a loaded Scan and current settings.

        Defensive: missing or unavailable fields default to None without raising.
        """
        # Source identity
        source_file   = str(scan.source_path) if getattr(scan, "source_path", None) else None
        source_format = getattr(scan, "source_format", None)

        # Array shape for the selected channel
        array_shape: tuple[int, int] | None = None
        try:
            planes = scan.planes
            idx = channel_index if channel_index is not None else 0
            if 0 <= idx < len(planes):
                ny, nx = planes[idx].shape
                array_shape = (int(ny), int(nx))
            elif planes:
                ny, nx = planes[0].shape
                array_shape = (int(ny), int(nx))
        except Exception:
            pass

        # Scan range
        scan_range_m: tuple[float, float] | None = None
        try:
            w, h = scan.scan_range_m
            scan_range_m = (float(w), float(h))
        except Exception:
            pass

        # Units for the selected channel
        units: str | None = None
        try:
            plane_units = scan.plane_units
            idx = channel_index if channel_index is not None else 0
            if 0 <= idx < len(plane_units):
                units = str(plane_units[idx])
        except Exception:
            pass

        # Channel name (prefer explicit argument, fall back to scan plane_names)
        if channel_name is None:
            try:
                plane_names = scan.plane_names
                idx = channel_index if channel_index is not None else 0
                if 0 <= idx < len(plane_names):
                    channel_name = str(plane_names[idx])
            except Exception:
                pass

        # Serialise state objects
        ps_dict = processing_state.to_dict() if processing_state is not None else {"steps": []}
        ds_dict = display_state.to_dict()    if display_state    is not None else {}

        return cls(
            source_file=source_file,
            source_format=source_format,
            item_type=item_type,
            channel_name=channel_name,
            channel_index=channel_index,
            array_shape=array_shape,
            scan_range_m=scan_range_m,
            units=units,
            processing_state=ps_dict,
            display_state=ds_dict,
            probeflow_version=_get_version(),
            export_timestamp=_utc_now(),
        )
