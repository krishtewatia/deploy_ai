"""Compute Capabilities package.

Provides Pydantic v2 schemas and enums representing interpreted machine resource
capabilities and compute constraints for AutoML planner allocations.
"""

from backend.app.compute_capabilities.analyzer import (
    HardwareCapabilityAnalysisError,
    HardwareCapabilityAnalyzer,
)
from backend.app.compute_capabilities.schemas import (
    AcceleratorType,
    ComputeCapabilities,
    ComputeTier,
    MemoryConstraintLevel,
    ResourceWarning,
)

__all__ = [
    "AcceleratorType",
    "ComputeCapabilities",
    "ComputeTier",
    "HardwareCapabilityAnalysisError",
    "HardwareCapabilityAnalyzer",
    "MemoryConstraintLevel",
    "ResourceWarning",
]
