"""Model Governance package.

Provides deterministic comparison and production readiness checks for baseline
and retrained model configurations.
"""

from backend.app.model_governance.schemas import Winner, ChampionDecision
from backend.app.model_governance.comparator import (
    ChampionComparator,
    ChampionComparatorError,
)

__all__ = [
    "Winner",
    "ChampionDecision",
    "ChampionComparator",
    "ChampionComparatorError",
]
