"""Hardware capability analyzer for mapping raw hardware profiles to AutoML constraints.

Transforms raw system metrics (CPU, Memory, GPU) into a compute capability level
for planning workloads locally.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from backend.app.compute_capabilities.schemas import (
    AcceleratorType,
    ComputeCapabilities,
    ComputeTier,
    MemoryConstraintLevel,
    ResourceWarning,
)
from backend.app.hardware.schemas import HardwareProfile

logger = logging.getLogger(__name__)


class HardwareCapabilityAnalysisError(Exception):
    """Raised when hardware capabilities cannot be analyzed or validation fails."""


class HardwareCapabilityAnalyzer:
    """Analyzes a HardwareProfile to derive planning compute constraints and levels."""

    def analyze(self, hardware_profile: HardwareProfile) -> ComputeCapabilities:
        """Transforms a HardwareProfile into ComputeCapabilities.

        Args:
            hardware_profile: The input validated system hardware snapshot.

        Returns:
            A resolved ComputeCapabilities object representing ML resource constraints.

        Raises:
            HardwareCapabilityAnalysisError: If the input is invalid or not a HardwareProfile.
        """
        try:
            # Validate input type
            if not isinstance(hardware_profile, HardwareProfile):
                raise HardwareCapabilityAnalysisError(
                    f"Expected HardwareProfile instance, got {type(hardware_profile).__name__}"
                )
            # 1. Capability ID
            capability_id = f"capability_{uuid.uuid4().hex}"
            hardware_profile_id = hardware_profile.profile_id

            # Extract basic facts
            total_ram_mb = hardware_profile.memory.total_ram_mb
            available_ram_mb = hardware_profile.memory.available_ram_mb
            logical_cores = hardware_profile.cpu.logical_cores

            # 2. CPU Training Availability
            cpu_training_available = logical_cores >= 1

            # 3. Max Parallel Workers
            max_parallel_workers = logical_cores

            # 4. Memory Constraint Classification
            available_ratio = available_ram_mb / total_ram_mb

            if available_ram_mb < 1024 or available_ratio < 0.10:
                memory_constraint = MemoryConstraintLevel.SEVERE
            elif available_ram_mb < 4096 or available_ratio < 0.25:
                memory_constraint = MemoryConstraintLevel.CONSTRAINED
            elif available_ram_mb < 8192 or available_ratio < 0.50:
                memory_constraint = MemoryConstraintLevel.MODERATE
            else:
                memory_constraint = MemoryConstraintLevel.COMFORTABLE

            # 5. GPU Acceleration
            gpu_acceleration_available = any(gpu.cuda_available for gpu in hardware_profile.gpus)
            if gpu_acceleration_available:
                accelerator_type = AcceleratorType.CUDA
            else:
                accelerator_type = AcceleratorType.NONE

            # 6. Compute Tier
            if (total_ram_mb >= 32768 and logical_cores >= 12) or (
                total_ram_mb >= 16384 and logical_cores >= 8 and gpu_acceleration_available
            ):
                compute_tier = ComputeTier.HIGH
            elif total_ram_mb >= 8192 and logical_cores >= 4:
                compute_tier = ComputeTier.STANDARD
            else:
                compute_tier = ComputeTier.MINIMAL

            # 7. Safe Parallel Workers
            cpu_worker_limit = max(1, logical_cores // 2)

            if memory_constraint == MemoryConstraintLevel.SEVERE:
                memory_worker_limit = 1
            elif memory_constraint == MemoryConstraintLevel.CONSTRAINED:
                memory_worker_limit = 2
            elif memory_constraint == MemoryConstraintLevel.MODERATE:
                memory_worker_limit = 4
            else:
                memory_worker_limit = 8

            safe_parallel_workers = min(
                cpu_worker_limit,
                memory_worker_limit,
                max_parallel_workers,
            )
            safe_parallel_workers = max(1, safe_parallel_workers)

            # 8 & 9. Resource Warnings & Deterministic Order
            warnings: list[ResourceWarning] = []

            # A/B. Memory warning
            if memory_constraint == MemoryConstraintLevel.SEVERE:
                warnings.append(
                    ResourceWarning(
                        code="SEVERE_MEMORY_PRESSURE",
                        message="Available system memory is critically limited for local ML workloads.",
                    )
                )
            elif memory_constraint == MemoryConstraintLevel.CONSTRAINED:
                warnings.append(
                    ResourceWarning(
                        code="LOW_AVAILABLE_MEMORY",
                        message="Available system memory may limit parallel or memory-intensive ML operations.",
                    )
                )

            # C. GPU Warning
            if not gpu_acceleration_available:
                warnings.append(
                    ResourceWarning(
                        code="NO_GPU_ACCELERATOR",
                        message="No supported GPU accelerator was detected; training will use CPU resources.",
                    )
                )

            # D. CPU Warning
            if logical_cores <= 2:
                warnings.append(
                    ResourceWarning(
                        code="LIMITED_CPU_PARALLELISM",
                        message="Limited logical CPU cores may restrict parallel ML operations.",
                    )
                )

            # 10. Construct capabilities (with snapshot RAM details)
            capabilities = ComputeCapabilities(
                capability_id=capability_id,
                hardware_profile_id=hardware_profile_id,
                compute_tier=compute_tier,
                memory_constraint=memory_constraint,
                cpu_training_available=cpu_training_available,
                gpu_acceleration_available=gpu_acceleration_available,
                accelerator_type=accelerator_type,
                safe_parallel_workers=safe_parallel_workers,
                max_parallel_workers=max_parallel_workers,
                available_ram_mb_snapshot=available_ram_mb,
                total_ram_mb=total_ram_mb,
                warnings=warnings,
            )
            return capabilities

        except Exception as e:
            if not isinstance(e, HardwareCapabilityAnalysisError):
                logger.exception("Unexpected error during hardware capability analysis")
                raise HardwareCapabilityAnalysisError(
                    f"Capabilities construction failed: {e}"
                ) from e
            raise
