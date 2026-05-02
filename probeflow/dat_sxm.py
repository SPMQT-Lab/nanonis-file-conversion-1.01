"""Compatibility wrapper for :mod:`probeflow.io.converters.createc_dat_to_sxm`."""

from probeflow.io.converters import createc_dat_to_sxm as _impl

globals().update({
    name: value
    for name, value in vars(_impl).items()
    if not (name.startswith("__") and name.endswith("__"))
})


def convert_dat_to_sxm(
    dat: Path,
    out_dir: Path,
    cushion_dir: Path,
    clip_low: float = 1.0,
    clip_high: float = 99.0,
) -> None:
    """Compatibility wrapper preserving monkeypatching of this module."""

    dat = Path(dat)
    out_dir = Path(out_dir)
    out_path = out_dir / (dat.stem + ".sxm")

    scan = load_scan(dat)
    write_sxm(
        scan,
        out_path,
        cushion_dir=cushion_dir,
        clip_low=clip_low,
        clip_high=clip_high,
    )
    log.info("[OK] %s -> %s", dat.name, out_path.name)
