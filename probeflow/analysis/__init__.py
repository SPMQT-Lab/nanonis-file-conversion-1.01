"""Measurement algorithms and analysis helpers.

Architectural role
------------------
``analysis`` contains operations that calculate information from scan images or
spectroscopy data: particles, lattice parameters, profiles, unit-cell tools,
classification helpers, and analysis export helpers. In the intended graph
architecture, these algorithms produce values that provenance-aware adapters can
wrap into ``MeasurementNode`` records.

Boundary rules
--------------
Keep measurement algorithms here, not graph models. Do not define
``MeasurementNode`` or other provenance node dataclasses in this package. Do not
place parser/writer boundaries, CLI routing, GUI widgets, or raw ``Scan`` model
ownership here.
"""
