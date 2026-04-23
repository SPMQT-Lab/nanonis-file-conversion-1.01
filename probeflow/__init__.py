"""ProbeFlow — STM scan browser, processor, and Createc↔Nanonis toolkit.

This package provides:
  * A Qt desktop GUI (``probeflow.gui``)
  * Createc ``.dat`` → Nanonis ``.sxm`` and PNG conversion pipelines
    (``probeflow.dat_sxm``, ``probeflow.dat_png``)
  * A GUI-free image processing library for STM data
    (``probeflow.processing``)
  * A unified command-line interface (``probeflow.cli``)

The library is importable without PySide6:

    from probeflow import processing
    from probeflow.dat_sxm import process_dat, convert_dat_to_sxm
    from probeflow.dat_png import dat_to_hdr_imgs

Launch the GUI via ``probeflow gui`` (see ``pyproject.toml`` for the
console-script wiring) or programmatically via ``probeflow.gui.main()``.
"""

__version__ = "1.3.0"
