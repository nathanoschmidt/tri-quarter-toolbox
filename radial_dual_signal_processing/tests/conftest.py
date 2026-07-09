"""pytest configuration for the radial_dual_signal_processing test suite.

Adds the subproject's ``src/`` directory to ``sys.path`` so the test modules can
``import tqf_hex_signal`` / ``import tqf_exact_predicate`` / ``import tqf_admissibility``
(and the simulation helpers) directly, mirroring how the simulation scripts import
their sibling modules when run from ``src/``.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
