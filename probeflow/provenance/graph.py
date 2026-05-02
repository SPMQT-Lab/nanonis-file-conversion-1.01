"""Future scan-owned provenance graph model.

This module is reserved for the graph dataclasses in the architectural vision:
``ImageNode``, ``MeasurementNode``, ``OperationNode``, ``ArtifactNode``, and
``ScanGraph``. A future ``Scan``/``Probe`` should own one graph; raw planes stay
immutable, root image nodes reference existing raw plane indices, and derived
images are virtual recipes unless explicitly materialised.

Do not put graph node definitions in ``probeflow.processing`` or
``probeflow.analysis``. Those packages should keep existing array and
measurement functions callable without provenance, while graph-aware wrappers
record operation identity, parameters, versions, warnings, units, and metadata
here.
"""
