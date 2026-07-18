"""Problem Definition package.

Provides Pydantic v2 schemas and enums representing resolved machine-learning
problem definitions for automated planning and execution.
"""

from backend.app.problem_definition.resolver import (
    ProblemResolver,
    ProblemResolverError,
)
from backend.app.problem_definition.schemas import (
    ConfirmationItem,
    ProblemDefinition,
    ProblemType,
    ProblemWarning,
    ResolutionStatus,
    TargetSource,
)

__all__ = [
    "ConfirmationItem",
    "ProblemDefinition",
    "ProblemResolver",
    "ProblemResolverError",
    "ProblemType",
    "ProblemWarning",
    "ResolutionStatus",
    "TargetSource",
]

