"""TASTE variant 2 — semantic-heavy engine.

Permutation:
A2 rich docs · B2 hybrid RRF · C2 Qwen3-E-0.6B · D2 LLM-judge · E4 lazy ·
F2 critique-revise · G2 LanceDB · H5 confidence math · I2 exemplars · J2 dynamic
"""

from .api import create_app
from .core import StandardsEngine
from .store import Store

__all__ = ["create_app", "StandardsEngine", "Store"]
