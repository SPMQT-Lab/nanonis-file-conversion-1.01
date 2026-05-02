"""Parser and writer boundaries for external ProbeFlow data.

Architectural role
------------------
``io`` owns operations that cross the filesystem boundary. Parsers turn vendor
files into core ``Scan`` / future ``Spectrum`` objects and will later create the
root image nodes in a probe's ``ScanGraph``. Writers turn ProbeFlow objects into
external artifacts and will later be recorded as writer operations that produce
``ArtifactNode`` entries.

Boundary rules
--------------
Keep vendor sniffing, readers, writers, and converters here. Do not add GUI
widgets, CLI command routing, numerical processing kernels, measurement
algorithms, or graph node dataclasses to this package.
"""
