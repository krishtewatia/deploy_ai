"""Pydantic v2 schemas and enums representing local hardware profiles.

These models serve as data contracts specifying system metrics (OS, CPU,
memory, GPUs, Python runtime) used for resource allocation and model selection.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class OperatingSystem(str, Enum):
    """Supported operating systems."""

    WINDOWS = "windows"
    LINUX = "linux"
    MACOS = "macos"
    OTHER = "other"


class GPUVendor(str, Enum):
    """Detected GPU vendors."""

    NVIDIA = "nvidia"
    AMD = "amd"
    INTEL = "intel"
    APPLE = "apple"
    OTHER = "other"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Component Schemas
# ---------------------------------------------------------------------------


class GPUDevice(BaseModel):
    """Details of a single detected GPU device."""

    name: str = Field(
        ...,
        description="Human-readable GPU name.",
    )
    vendor: GPUVendor = Field(
        default=GPUVendor.UNKNOWN,
        description="The GPU manufacturer.",
    )
    memory_total_mb: Optional[int] = Field(
        default=None,
        ge=0,
        description="Total detected GPU memory in MB.",
    )
    memory_available_mb: Optional[int] = Field(
        default=None,
        ge=0,
        description="Currently available detected GPU memory in MB.",
    )
    cuda_available: bool = Field(
        default=False,
        description="Whether CUDA is supported by this GPU/environment.",
    )
    cuda_version: Optional[str] = Field(
        default=None,
        description="Detected CUDA version string.",
    )

    @field_validator("name", mode="before")
    @classmethod
    def _strip_and_validate_name(cls, v: Any) -> str:
        if not isinstance(v, str):
            raise ValueError("GPU name must be a string")
        stripped = v.strip()
        if not stripped:
            raise ValueError("GPU name cannot be empty or whitespace-only")
        return stripped

    @field_validator("cuda_version", mode="before")
    @classmethod
    def _strip_and_validate_cuda_version(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        if not isinstance(v, str):
            raise ValueError("CUDA version must be a string")
        stripped = v.strip()
        if not stripped:
            raise ValueError("CUDA version cannot be empty or whitespace-only")
        return stripped

    @model_validator(mode="after")
    def _validate_gpu_consistency(self) -> GPUDevice:
        if (
            self.memory_available_mb is not None
            and self.memory_total_mb is not None
        ):
            if self.memory_available_mb > self.memory_total_mb:
                raise ValueError(
                    f"Available GPU memory ({self.memory_available_mb} MB) "
                    f"cannot exceed total memory ({self.memory_total_mb} MB)"
                )
        if not self.cuda_available and self.cuda_version is not None:
            raise ValueError("CUDA version must be None if CUDA is not available")
        return self


class CPUProfile(BaseModel):
    """Details of the processor profile."""

    processor_name: str = Field(
        ...,
        description="Human-readable processor/CPU name.",
    )
    architecture: str = Field(
        ...,
        description="CPU hardware architecture.",
    )
    physical_cores: Optional[int] = Field(
        default=None,
        ge=1,
        description="Number of physical cores.",
    )
    logical_cores: int = Field(
        ...,
        ge=1,
        description="Number of logical cores.",
    )

    @field_validator("processor_name", "architecture", mode="before")
    @classmethod
    def _strip_and_validate_strings(cls, v: Any) -> str:
        if not isinstance(v, str):
            raise ValueError("Field must be a string")
        stripped = v.strip()
        if not stripped:
            raise ValueError("Field cannot be empty or whitespace-only")
        return stripped

    @model_validator(mode="after")
    def _validate_cpu_cores(self) -> CPUProfile:
        if self.physical_cores is not None:
            if self.physical_cores > self.logical_cores:
                raise ValueError(
                    f"Physical cores ({self.physical_cores}) "
                    f"cannot exceed logical cores ({self.logical_cores})"
                )
        return self


class MemoryProfile(BaseModel):
    """Details of system memory (RAM)."""

    total_ram_mb: int = Field(
        ...,
        ge=1,
        description="Total system RAM in MB.",
    )
    available_ram_mb: int = Field(
        ...,
        ge=0,
        description="Currently available system RAM in MB.",
    )

    @model_validator(mode="after")
    def _validate_memory(self) -> MemoryProfile:
        if self.available_ram_mb > self.total_ram_mb:
            raise ValueError(
                f"Available RAM ({self.available_ram_mb} MB) "
                f"cannot exceed total RAM ({self.total_ram_mb} MB)"
            )
        return self


class RuntimeProfile(BaseModel):
    """Details of the executing Python runtime environment."""

    python_version: str = Field(
        ...,
        description="Detected Python version (e.g. '3.14.0').",
    )
    python_implementation: str = Field(
        ...,
        description="Python implementation name (e.g. 'CPython').",
    )
    machine_architecture: str = Field(
        ...,
        description="Machine platform architecture (e.g. 'AMD64').",
    )

    @field_validator(
        "python_version",
        "python_implementation",
        "machine_architecture",
        mode="before",
    )
    @classmethod
    def _strip_and_validate_strings(cls, v: Any) -> str:
        if not isinstance(v, str):
            raise ValueError("Field must be a string")
        stripped = v.strip()
        if not stripped:
            raise ValueError("Field cannot be empty or whitespace-only")
        return stripped


# ---------------------------------------------------------------------------
# Main HardwareProfile Schema
# ---------------------------------------------------------------------------


class HardwareProfile(BaseModel):
    """Aggregated snapshot representing system hardware facts."""

    profile_id: str = Field(
        ...,
        description="Unique identifier for this hardware profile snapshot.",
    )
    operating_system: OperatingSystem = Field(
        ...,
        description="Operating system family.",
    )
    os_version: str = Field(
        ...,
        description="Human-readable operating system version.",
    )
    cpu: CPUProfile = Field(
        ...,
        description="System CPU profile details.",
    )
    memory: MemoryProfile = Field(
        ...,
        description="System RAM profile details.",
    )
    gpus: list[GPUDevice] = Field(
        default_factory=list,
        description="List of detected GPUs on the local system.",
    )
    runtime: RuntimeProfile = Field(
        ...,
        description="Runtime environment details.",
    )

    @field_validator("profile_id", "os_version", mode="before")
    @classmethod
    def _strip_and_validate_strings(cls, v: Any) -> str:
        if not isinstance(v, str):
            raise ValueError("Field must be a string")
        stripped = v.strip()
        if not stripped:
            raise ValueError("Field cannot be empty or whitespace-only")
        return stripped
