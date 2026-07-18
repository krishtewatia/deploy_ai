"""Hardware Profile package.

Provides Pydantic v2 schemas and enums representing local system hardware
specifications to assist with downstream AutoML resource allocation and execution.
"""

from backend.app.hardware.detector import (
    HardwareDetectionError,
    HardwareDetector,
)
from backend.app.hardware.schemas import (
    CPUProfile,
    GPUDevice,
    GPUVendor,
    HardwareProfile,
    MemoryProfile,
    OperatingSystem,
    RuntimeProfile,
)

__all__ = [
    "CPUProfile",
    "GPUDevice",
    "GPUVendor",
    "HardwareDetectionError",
    "HardwareDetector",
    "HardwareProfile",
    "MemoryProfile",
    "OperatingSystem",
    "RuntimeProfile",
]

