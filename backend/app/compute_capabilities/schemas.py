"""Pydantic v2 schemas and enums representing system compute capabilities.

These models serve as data contracts specifying dynamic compute constraints
interpreted from local system hardware (GPU acceleration, memory pressure,
parallel workers limits) used directly by AutoML planning.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class MemoryConstraintLevel(str, Enum):
    """Memory pressure category based on available system memory."""

    SEVERE = "severe"
    CONSTRAINED = "constrained"
    MODERATE = "moderate"
    COMFORTABLE = "comfortable"


class ComputeTier(str, Enum):
    """General compute resource capability category."""

    MINIMAL = "minimal"
    STANDARD = "standard"
    HIGH = "high"


class AcceleratorType(str, Enum):
    """Hardware acceleration type available on the system."""

    NONE = "none"
    CUDA = "cuda"


# ---------------------------------------------------------------------------
# ResourceWarning Schema
# ---------------------------------------------------------------------------


class ResourceWarning(BaseModel):
    """Structured resource warning detailing hardware or capacity limitations."""

    code: str = Field(
        ...,
        description="Unique limitation classification code.",
    )
    message: str = Field(
        ...,
        description="Human-readable warning explaining the constraint.",
    )

    @field_validator("code", "message", mode="before")
    @classmethod
    def _strip_and_validate_strings(cls, v: Any) -> str:
        if not isinstance(v, str):
            raise ValueError("Field must be a string")
        stripped = v.strip()
        if not stripped:
            raise ValueError("Field cannot be empty or whitespace-only")
        return stripped


# ---------------------------------------------------------------------------
# ComputeCapabilities Schema
# ---------------------------------------------------------------------------


class ComputeCapabilities(BaseModel):
    """Interpreted system compute constraints and planner execution capability facts."""

    capability_id: str = Field(
        ...,
        description="Unique identifier for this capability analysis snapshot.",
    )
    hardware_profile_id: str = Field(
        ...,
        description="Identifier of the hardware profile used to resolve capabilities.",
    )
    compute_tier: ComputeTier = Field(
        ...,
        description="Compute resource category mapping.",
    )
    memory_constraint: MemoryConstraintLevel = Field(
        ...,
        description="Memory pressure constraint category.",
    )
    cpu_training_available: bool = Field(
        ...,
        description="Whether CPU-based model training can be run.",
    )
    gpu_acceleration_available: bool = Field(
        ...,
        description="Whether GPU hardware acceleration is supported/available.",
    )
    accelerator_type: AcceleratorType = Field(
        ...,
        description="Type of GPU accelerator available.",
    )
    safe_parallel_workers: int = Field(
        ...,
        ge=1,
        description="Safe, resource-conservative worker count for parallel execution.",
    )
    max_parallel_workers: int = Field(
        ...,
        ge=1,
        description="Maximum parallel worker processes allowed by available hardware.",
    )
    available_ram_mb_snapshot: int = Field(
        ...,
        ge=0,
        description="Snapshot of available RAM in MB recorded during hardware discovery.",
    )
    total_ram_mb: int = Field(
        ...,
        ge=1,
        description="Total system memory in MB.",
    )
    warnings: list[ResourceWarning] = Field(
        default_factory=list,
        description="Collection of non-blocking capacity/lacking resource warnings.",
    )

    # --- Validators ---

    @field_validator("capability_id", "hardware_profile_id", mode="before")
    @classmethod
    def _strip_and_validate_ids(cls, v: Any) -> str:
        if not isinstance(v, str):
            raise ValueError("Field must be a string")
        stripped = v.strip()
        if not stripped:
            raise ValueError("Field cannot be empty or whitespace-only")
        return stripped

    @model_validator(mode="after")
    def _validate_compute_consistency(self) -> ComputeCapabilities:
        # 1. safe_parallel_workers <= max_parallel_workers
        if self.safe_parallel_workers > self.max_parallel_workers:
            raise ValueError(
                f"safe_parallel_workers ({self.safe_parallel_workers}) "
                f"cannot exceed max_parallel_workers ({self.max_parallel_workers})"
            )

        # 2. available_ram_mb_snapshot <= total_ram_mb
        if self.available_ram_mb_snapshot > self.total_ram_mb:
            raise ValueError(
                f"available_ram_mb_snapshot ({self.available_ram_mb_snapshot} MB) "
                f"cannot exceed total_ram_mb ({self.total_ram_mb} MB)"
            )

        # 3. GPU/Accelerator consistency checks
        if not self.gpu_acceleration_available:
            if self.accelerator_type != AcceleratorType.NONE:
                raise ValueError(
                    f"accelerator_type must be '{AcceleratorType.NONE}' "
                    f"when gpu_acceleration_available is False"
                )
        else:
            if self.accelerator_type == AcceleratorType.NONE:
                raise ValueError(
                    f"accelerator_type cannot be '{AcceleratorType.NONE}' "
                    f"when gpu_acceleration_available is True"
                )

        return self
