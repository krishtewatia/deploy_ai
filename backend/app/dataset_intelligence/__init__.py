"""Dataset Intelligence package.

Provides Pydantic v2 schemas representing compact, machine-readable
dataset intelligence for automated ML planning and execution, and a
deterministic builder to convert analysis outputs into those schemas.
"""

from backend.app.dataset_intelligence.context_builder import (
    DatasetContextBuilder,
    DatasetContextBuilderError,
)
from backend.app.dataset_intelligence.schemas import (
    ColumnContext,
    ColumnStatistics,
    DatasetBasicInfo,
    DatasetContext,
    DuplicateSummary,
    MissingDataSummary,
    TargetCandidateSummary,
)

__all__ = [
    "ColumnContext",
    "ColumnStatistics",
    "DatasetBasicInfo",
    "DatasetContext",
    "DatasetContextBuilder",
    "DatasetContextBuilderError",
    "DuplicateSummary",
    "MissingDataSummary",
    "TargetCandidateSummary",
]
