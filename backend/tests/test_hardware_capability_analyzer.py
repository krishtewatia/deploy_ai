"""Unit tests for HardwareCapabilityAnalyzer.

Covers all 56 specified test scenarios for mapping a HardwareProfile
into interpreted ComputeCapabilities.
"""

import copy
import json
from unittest.mock import patch

import pytest
from pydantic import ValidationError

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
from backend.app.hardware.schemas import (
    CPUProfile,
    GPUDevice,
    GPUVendor,
    HardwareProfile,
    MemoryProfile,
    OperatingSystem,
    RuntimeProfile,
)


# ── Helper Constructors ────────────────────────────────────────────────


def _make_hardware_profile(
    logical_cores: int = 4,
    total_ram_mb: int = 16384,
    available_ram_mb: int = None,
    gpus: list[GPUDevice] = None,
) -> HardwareProfile:
    """Create a valid HardwareProfile instance with customizable resource bounds."""
    if gpus is None:
        gpus = []

    if available_ram_mb is None:
        available_ram_mb = total_ram_mb // 2

    cpu = CPUProfile(
        processor_name="Intel Core i7",
        architecture="AMD64",
        physical_cores=max(1, logical_cores // 2),
        logical_cores=logical_cores,
    )
    memory = MemoryProfile(
        total_ram_mb=total_ram_mb,
        available_ram_mb=available_ram_mb,
    )
    runtime = RuntimeProfile(
        python_version="3.14.0",
        python_implementation="CPython",
        machine_architecture="AMD64",
    )
    return HardwareProfile(
        profile_id="hw_test_id_123",
        operating_system=OperatingSystem.WINDOWS,
        os_version="10.0.19045",
        cpu=cpu,
        memory=memory,
        gpus=gpus,
        runtime=runtime,
    )


# ── Tests ──────────────────────────────────────────────────────────────


class TestHardwareCapabilityAnalyzer:
    """Comprehensive unit tests covering all hardware analyzer mapping logic."""

    # === INPUT VALIDATION ===

    # 1. Valid HardwareProfile accepted
    def test_valid_hardware_profile_accepted(self):
        analyzer = HardwareCapabilityAnalyzer()
        profile = _make_hardware_profile()
        caps = analyzer.analyze(profile)
        assert isinstance(caps, ComputeCapabilities)

    # 2. Dictionary input rejected
    def test_dictionary_input_rejected(self):
        analyzer = HardwareCapabilityAnalyzer()
        with pytest.raises(HardwareCapabilityAnalysisError, match="Expected HardwareProfile instance"):
            analyzer.analyze({"profile_id": "hw-123"})

    # 3. None rejected
    def test_none_input_rejected(self):
        analyzer = HardwareCapabilityAnalyzer()
        with pytest.raises(HardwareCapabilityAnalysisError, match="Expected HardwareProfile instance"):
            analyzer.analyze(None)

    # 4. Input HardwareProfile is not mutated
    def test_input_hardware_profile_not_mutated(self):
        analyzer = HardwareCapabilityAnalyzer()
        profile = _make_hardware_profile()
        original_profile = copy.deepcopy(profile)
        analyzer.analyze(profile)
        assert profile == original_profile

    # === CAPABILITY IDENTIFIERS ===

    # 5. capability_id starts with "capability_"
    def test_capability_id_prefix(self):
        analyzer = HardwareCapabilityAnalyzer()
        caps = analyzer.analyze(_make_hardware_profile())
        assert caps.capability_id.startswith("capability_")

    # 6. Two analyses produce different capability IDs
    def test_different_capability_ids_for_each_run(self):
        analyzer = HardwareCapabilityAnalyzer()
        profile = _make_hardware_profile()
        c1 = analyzer.analyze(profile)
        c2 = analyzer.analyze(profile)
        assert c1.capability_id != c2.capability_id

    # 7. hardware_profile_id matches input profile ID
    def test_hardware_profile_id_matches(self):
        analyzer = HardwareCapabilityAnalyzer()
        profile = _make_hardware_profile()
        caps = analyzer.analyze(profile)
        assert caps.hardware_profile_id == profile.profile_id

    # === CPU ===

    # 8. cpu_training_available is True for valid hardware
    def test_cpu_training_available_true(self):
        analyzer = HardwareCapabilityAnalyzer()
        profile = _make_hardware_profile(logical_cores=1)
        caps = analyzer.analyze(profile)
        assert caps.cpu_training_available is True

    # 9. max_parallel_workers equals logical cores
    def test_max_parallel_workers_equals_logical_cores(self):
        analyzer = HardwareCapabilityAnalyzer()
        profile = _make_hardware_profile(logical_cores=6)
        caps = analyzer.analyze(profile)
        assert caps.max_parallel_workers == 6

    # === MEMORY — SEVERE ===

    # 10. available RAM below 1024 MB -> severe
    def test_memory_severe_absolute(self):
        analyzer = HardwareCapabilityAnalyzer()
        profile = _make_hardware_profile(total_ram_mb=16384, available_ram_mb=1023)
        caps = analyzer.analyze(profile)
        assert caps.memory_constraint == MemoryConstraintLevel.SEVERE

    # 11. available ratio below 10% -> severe
    def test_memory_severe_ratio(self):
        analyzer = HardwareCapabilityAnalyzer()
        profile = _make_hardware_profile(total_ram_mb=20000, available_ram_mb=1900)  # 9.5%
        caps = analyzer.analyze(profile)
        assert caps.memory_constraint == MemoryConstraintLevel.SEVERE

    # 12. 1024 MB with ratio exactly 10% is not severe
    def test_memory_severe_boundary_exactly_10_percent(self):
        analyzer = HardwareCapabilityAnalyzer()
        profile = _make_hardware_profile(total_ram_mb=10240, available_ram_mb=1024)  # exactly 10% and 1024 MB
        caps = analyzer.analyze(profile)
        assert caps.memory_constraint != MemoryConstraintLevel.SEVERE

    # === MEMORY — CONSTRAINED ===

    # 13. available RAM below 4096 MB -> constrained
    def test_memory_constrained_absolute(self):
        analyzer = HardwareCapabilityAnalyzer()
        profile = _make_hardware_profile(total_ram_mb=16384, available_ram_mb=4095)
        caps = analyzer.analyze(profile)
        assert caps.memory_constraint == MemoryConstraintLevel.CONSTRAINED

    # 14. available ratio below 25% -> constrained
    def test_memory_constrained_ratio(self):
        analyzer = HardwareCapabilityAnalyzer()
        profile = _make_hardware_profile(total_ram_mb=20000, available_ram_mb=4800)  # 24%
        caps = analyzer.analyze(profile)
        assert caps.memory_constraint == MemoryConstraintLevel.CONSTRAINED

    # 15. 4096 MB with ratio exactly 25% is not constrained
    def test_memory_constrained_boundary_exactly_25_percent(self):
        analyzer = HardwareCapabilityAnalyzer()
        profile = _make_hardware_profile(total_ram_mb=16384, available_ram_mb=4096)  # exactly 25% and 4096 MB
        caps = analyzer.analyze(profile)
        assert caps.memory_constraint != MemoryConstraintLevel.SEVERE
        assert caps.memory_constraint != MemoryConstraintLevel.CONSTRAINED

    # === MEMORY — MODERATE ===

    # 16. available RAM below 8192 MB -> moderate
    def test_memory_moderate_absolute(self):
        analyzer = HardwareCapabilityAnalyzer()
        profile = _make_hardware_profile(total_ram_mb=20000, available_ram_mb=8191)
        caps = analyzer.analyze(profile)
        assert caps.memory_constraint == MemoryConstraintLevel.MODERATE

    # 17. available ratio below 50% -> moderate
    def test_memory_moderate_ratio(self):
        analyzer = HardwareCapabilityAnalyzer()
        profile = _make_hardware_profile(total_ram_mb=20000, available_ram_mb=9800)  # 49%
        caps = analyzer.analyze(profile)
        assert caps.memory_constraint == MemoryConstraintLevel.MODERATE

    # 18. 8192 MB with ratio exactly 50% -> comfortable
    def test_memory_moderate_boundary_exactly_50_percent(self):
        analyzer = HardwareCapabilityAnalyzer()
        profile = _make_hardware_profile(total_ram_mb=16384, available_ram_mb=8192)  # exactly 50% and 8192 MB
        caps = analyzer.analyze(profile)
        assert caps.memory_constraint == MemoryConstraintLevel.COMFORTABLE

    # === MEMORY — COMFORTABLE ===

    # 19. sufficient absolute and ratio memory -> comfortable
    def test_memory_comfortable(self):
        analyzer = HardwareCapabilityAnalyzer()
        profile = _make_hardware_profile(total_ram_mb=16384, available_ram_mb=10000)
        caps = analyzer.analyze(profile)
        assert caps.memory_constraint == MemoryConstraintLevel.COMFORTABLE

    # === GPU ACCELERATION ===

    # 20. Empty GPU list -> no acceleration
    def test_gpu_empty_list_no_acceleration(self):
        analyzer = HardwareCapabilityAnalyzer()
        profile = _make_hardware_profile(gpus=[])
        caps = analyzer.analyze(profile)
        assert caps.gpu_acceleration_available is False
        assert caps.accelerator_type == AcceleratorType.NONE

    # 21. Non-CUDA GPU -> no acceleration
    def test_gpu_non_cuda_vendor_no_acceleration(self):
        analyzer = HardwareCapabilityAnalyzer()
        gpus = [
            GPUDevice(
                name="Intel Iris Xe",
                vendor=GPUVendor.INTEL,
                cuda_available=False,
            )
        ]
        profile = _make_hardware_profile(gpus=gpus)
        caps = analyzer.analyze(profile)
        assert caps.gpu_acceleration_available is False
        assert caps.accelerator_type == AcceleratorType.NONE

    # 22. NVIDIA GPU with cuda_available=False -> no acceleration
    def test_gpu_nvidia_no_cuda_no_acceleration(self):
        analyzer = HardwareCapabilityAnalyzer()
        gpus = [
            GPUDevice(
                name="GeForce RTX 4070",
                vendor=GPUVendor.NVIDIA,
                cuda_available=False,
            )
        ]
        profile = _make_hardware_profile(gpus=gpus)
        caps = analyzer.analyze(profile)
        assert caps.gpu_acceleration_available is False
        assert caps.accelerator_type == AcceleratorType.NONE

    # 23. CUDA GPU -> CUDA acceleration
    def test_gpu_cuda_available_gives_cuda_acceleration(self):
        analyzer = HardwareCapabilityAnalyzer()
        gpus = [
            GPUDevice(
                name="GeForce RTX 3080",
                vendor=GPUVendor.NVIDIA,
                cuda_available=True,
                cuda_version="12.1",
            )
        ]
        profile = _make_hardware_profile(gpus=gpus)
        caps = analyzer.analyze(profile)
        assert caps.gpu_acceleration_available is True
        assert caps.accelerator_type == AcceleratorType.CUDA

    # 24. Multiple GPUs with one CUDA GPU -> CUDA acceleration
    def test_gpu_multiple_gpus_one_cuda_gives_cuda(self):
        analyzer = HardwareCapabilityAnalyzer()
        gpus = [
            GPUDevice(name="Intel Iris", vendor=GPUVendor.INTEL, cuda_available=False),
            GPUDevice(name="GeForce RTX 3080", vendor=GPUVendor.NVIDIA, cuda_available=True, cuda_version="11.8"),
        ]
        profile = _make_hardware_profile(gpus=gpus)
        caps = analyzer.analyze(profile)
        assert caps.gpu_acceleration_available is True
        assert caps.accelerator_type == AcceleratorType.CUDA

    # 25. GPU vendor alone does not imply CUDA
    def test_gpu_vendor_nvidia_alone_does_not_imply_cuda(self):
        analyzer = HardwareCapabilityAnalyzer()
        gpus = [
            GPUDevice(
                name="Unknown GPU",
                vendor=GPUVendor.NVIDIA,
                cuda_available=False,
            )
        ]
        profile = _make_hardware_profile(gpus=gpus)
        caps = analyzer.analyze(profile)
        assert caps.gpu_acceleration_available is False
        assert caps.accelerator_type == AcceleratorType.NONE

    # === COMPUTE TIER — MINIMAL ===

    # 26. Low RAM and low CPU -> minimal
    def test_compute_tier_low_ram_low_cpu_is_minimal(self):
        analyzer = HardwareCapabilityAnalyzer()
        profile = _make_hardware_profile(logical_cores=2, total_ram_mb=4096)
        caps = analyzer.analyze(profile)
        assert caps.compute_tier == ComputeTier.MINIMAL

    # 27. High CPU alone without sufficient RAM does not become high
    def test_compute_tier_high_cpu_low_ram_is_not_high(self):
        analyzer = HardwareCapabilityAnalyzer()
        profile = _make_hardware_profile(logical_cores=16, total_ram_mb=8192)
        caps = analyzer.analyze(profile)
        assert caps.compute_tier != ComputeTier.HIGH

    # 28. CUDA alone does not become high
    def test_compute_tier_cuda_alone_does_not_become_high(self):
        analyzer = HardwareCapabilityAnalyzer()
        gpus = [
            GPUDevice(
                name="GeForce RTX 3080",
                vendor=GPUVendor.NVIDIA,
                cuda_available=True,
            )
        ]
        profile = _make_hardware_profile(logical_cores=2, total_ram_mb=4096, gpus=gpus)
        caps = analyzer.analyze(profile)
        assert caps.compute_tier != ComputeTier.HIGH

    # === COMPUTE TIER — STANDARD ===

    # 29. 8192 MB RAM + 4 logical cores -> standard
    def test_compute_tier_standard_exact_bounds(self):
        analyzer = HardwareCapabilityAnalyzer()
        profile = _make_hardware_profile(logical_cores=4, total_ram_mb=8192)
        caps = analyzer.analyze(profile)
        assert caps.compute_tier == ComputeTier.STANDARD

    # 30. More than minimum standard resources -> standard when high rules are not met
    def test_compute_tier_standard_high_bounds_not_met(self):
        analyzer = HardwareCapabilityAnalyzer()
        profile = _make_hardware_profile(logical_cores=8, total_ram_mb=16384, gpus=[])
        # High CPU and High RAM but NO CUDA -> Standard (since High requires 12 cores OR 8 cores + CUDA)
        caps = analyzer.analyze(profile)
        assert caps.compute_tier == ComputeTier.STANDARD

    # === COMPUTE TIER — HIGH ===

    # 31. 32768 MB RAM + 12 logical cores -> high without CUDA
    def test_compute_tier_high_no_cuda(self):
        analyzer = HardwareCapabilityAnalyzer()
        profile = _make_hardware_profile(logical_cores=12, total_ram_mb=32768, gpus=[])
        caps = analyzer.analyze(profile)
        assert caps.compute_tier == ComputeTier.HIGH

    # 32. 16384 MB RAM + 8 logical cores + CUDA -> high
    def test_compute_tier_high_with_cuda(self):
        analyzer = HardwareCapabilityAnalyzer()
        gpus = [
            GPUDevice(
                name="GeForce RTX 3080",
                vendor=GPUVendor.NVIDIA,
                cuda_available=True,
            )
        ]
        profile = _make_hardware_profile(logical_cores=8, total_ram_mb=16384, gpus=gpus)
        caps = analyzer.analyze(profile)
        assert caps.compute_tier == ComputeTier.HIGH

    # 33. 16384 MB RAM + 8 logical cores without CUDA is standard
    def test_compute_tier_high_requires_cuda_or_more_cores(self):
        analyzer = HardwareCapabilityAnalyzer()
        profile = _make_hardware_profile(logical_cores=8, total_ram_mb=16384, gpus=[])
        caps = analyzer.analyze(profile)
        assert caps.compute_tier == ComputeTier.STANDARD

    # === SAFE PARALLEL WORKERS ===

    # 34. Severe memory -> maximum safe memory worker limit of 1
    def test_safe_workers_severe_memory(self):
        analyzer = HardwareCapabilityAnalyzer()
        profile = _make_hardware_profile(logical_cores=12, total_ram_mb=16384, available_ram_mb=500)
        caps = analyzer.analyze(profile)
        assert caps.safe_parallel_workers == 1

    # 35. Constrained memory -> maximum safe memory worker limit of 2
    def test_safe_workers_constrained_memory(self):
        analyzer = HardwareCapabilityAnalyzer()
        profile = _make_hardware_profile(logical_cores=12, total_ram_mb=16384, available_ram_mb=2048)
        caps = analyzer.analyze(profile)
        assert caps.safe_parallel_workers == 2

    # 36. Moderate memory -> maximum safe memory worker limit of 4
    def test_safe_workers_moderate_memory(self):
        analyzer = HardwareCapabilityAnalyzer()
        profile = _make_hardware_profile(logical_cores=12, total_ram_mb=16384, available_ram_mb=6144)
        caps = analyzer.analyze(profile)
        assert caps.safe_parallel_workers == 4

    # 37. Comfortable memory -> maximum safe memory worker limit of 8
    def test_safe_workers_comfortable_memory(self):
        analyzer = HardwareCapabilityAnalyzer()
        profile = _make_hardware_profile(logical_cores=12, total_ram_mb=16384, available_ram_mb=12000)
        caps = analyzer.analyze(profile)
        # min(12//2 = 6, memory_limit = 8, max_parallel = 12) -> 6
        assert caps.safe_parallel_workers == 6

    # 38. CPU half-core limit is respected
    def test_safe_workers_cpu_half_core_limit(self):
        analyzer = HardwareCapabilityAnalyzer()
        profile = _make_hardware_profile(logical_cores=4, total_ram_mb=16384, available_ram_mb=12000)
        caps = analyzer.analyze(profile)
        # min(4//2 = 2, memory_limit = 8, max_parallel = 4) -> 2
        assert caps.safe_parallel_workers == 2

    # 39. One logical core produces one safe worker
    def test_safe_workers_one_core_produces_one_worker(self):
        analyzer = HardwareCapabilityAnalyzer()
        profile = _make_hardware_profile(logical_cores=1, total_ram_mb=16384, available_ram_mb=12000)
        caps = analyzer.analyze(profile)
        # cpu_limit = max(1, 1//2) = 1
        assert caps.safe_parallel_workers == 1

    # 40. safe_parallel_workers never exceeds max_parallel_workers
    def test_safe_workers_not_exceed_max_workers(self):
        analyzer = HardwareCapabilityAnalyzer()
        profile = _make_hardware_profile(logical_cores=2, total_ram_mb=16384, available_ram_mb=12000)
        caps = analyzer.analyze(profile)
        # cpu_limit = 1, memory_limit = 8, max_parallel = 2 -> min(1, 8, 2) = 1
        assert caps.safe_parallel_workers <= caps.max_parallel_workers

    # === WARNINGS ===

    # 41. Severe memory produces SEVERE_MEMORY_PRESSURE
    def test_warning_severe_memory_pressure(self):
        analyzer = HardwareCapabilityAnalyzer()
        profile = _make_hardware_profile(available_ram_mb=500)
        caps = analyzer.analyze(profile)
        warnings = [w.code for w in caps.warnings]
        assert "SEVERE_MEMORY_PRESSURE" in warnings

    # 42. Severe memory does not also produce LOW_AVAILABLE_MEMORY
    def test_warning_severe_memory_excludes_constrained_warning(self):
        analyzer = HardwareCapabilityAnalyzer()
        profile = _make_hardware_profile(available_ram_mb=500)
        caps = analyzer.analyze(profile)
        warnings = [w.code for w in caps.warnings]
        assert "SEVERE_MEMORY_PRESSURE" in warnings
        assert "LOW_AVAILABLE_MEMORY" not in warnings

    # 43. Constrained memory produces LOW_AVAILABLE_MEMORY
    def test_warning_low_available_memory(self):
        analyzer = HardwareCapabilityAnalyzer()
        profile = _make_hardware_profile(available_ram_mb=2048)
        caps = analyzer.analyze(profile)
        warnings = [w.code for w in caps.warnings]
        assert "LOW_AVAILABLE_MEMORY" in warnings

    # 44. Moderate memory produces no memory warning
    def test_warning_moderate_memory_no_warning(self):
        analyzer = HardwareCapabilityAnalyzer()
        profile = _make_hardware_profile(available_ram_mb=6144)
        caps = analyzer.analyze(profile)
        warnings = [w.code for w in caps.warnings]
        assert "SEVERE_MEMORY_PRESSURE" not in warnings
        assert "LOW_AVAILABLE_MEMORY" not in warnings

    # 45. Comfortable memory produces no memory warning
    def test_warning_comfortable_memory_no_warning(self):
        analyzer = HardwareCapabilityAnalyzer()
        profile = _make_hardware_profile(available_ram_mb=12000)
        caps = analyzer.analyze(profile)
        warnings = [w.code for w in caps.warnings]
        assert "SEVERE_MEMORY_PRESSURE" not in warnings
        assert "LOW_AVAILABLE_MEMORY" not in warnings

    # 46. No CUDA accelerator produces NO_GPU_ACCELERATOR
    def test_warning_no_gpu_accelerator(self):
        analyzer = HardwareCapabilityAnalyzer()
        profile = _make_hardware_profile(gpus=[])
        caps = analyzer.analyze(profile)
        warnings = [w.code for w in caps.warnings]
        assert "NO_GPU_ACCELERATOR" in warnings

    # 47. CUDA accelerator suppresses NO_GPU_ACCELERATOR
    def test_warning_cuda_suppresses_no_gpu_accelerator(self):
        analyzer = HardwareCapabilityAnalyzer()
        gpus = [GPUDevice(name="RTX", vendor=GPUVendor.NVIDIA, cuda_available=True)]
        profile = _make_hardware_profile(gpus=gpus)
        caps = analyzer.analyze(profile)
        warnings = [w.code for w in caps.warnings]
        assert "NO_GPU_ACCELERATOR" not in warnings

    # 48. <= 2 logical cores produces LIMITED_CPU_PARALLELISM
    def test_warning_limited_cpu_parallelism(self):
        analyzer = HardwareCapabilityAnalyzer()
        profile = _make_hardware_profile(logical_cores=2)
        caps = analyzer.analyze(profile)
        warnings = [w.code for w in caps.warnings]
        assert "LIMITED_CPU_PARALLELISM" in warnings

    # 49. > 2 logical cores does not produce LIMITED_CPU_PARALLELISM
    def test_warning_greater_cpu_no_limited_parallelism(self):
        analyzer = HardwareCapabilityAnalyzer()
        profile = _make_hardware_profile(logical_cores=4)
        caps = analyzer.analyze(profile)
        warnings = [w.code for w in caps.warnings]
        assert "LIMITED_CPU_PARALLELISM" not in warnings

    # 50. Warning order is deterministic
    def test_warning_order_deterministic(self):
        analyzer = HardwareCapabilityAnalyzer()
        # Trigger all warning categories: severe memory, no gpu, <= 2 cores
        profile = _make_hardware_profile(logical_cores=2, available_ram_mb=500, gpus=[])
        caps = analyzer.analyze(profile)
        assert len(caps.warnings) == 3
        assert caps.warnings[0].code == "SEVERE_MEMORY_PRESSURE"
        assert caps.warnings[1].code == "NO_GPU_ACCELERATOR"
        assert caps.warnings[2].code == "LIMITED_CPU_PARALLELISM"

    # === OUTPUT ===

    # 51. Output is a ComputeCapabilities instance
    def test_output_is_compute_capabilities(self):
        analyzer = HardwareCapabilityAnalyzer()
        caps = analyzer.analyze(_make_hardware_profile())
        assert isinstance(caps, ComputeCapabilities)

    # 52. available_ram_mb_snapshot matches input
    def test_available_ram_snapshot_matches(self):
        analyzer = HardwareCapabilityAnalyzer()
        profile = _make_hardware_profile(available_ram_mb=1234)
        caps = analyzer.analyze(profile)
        assert caps.available_ram_mb_snapshot == 1234

    # 53. total_ram_mb matches input
    def test_total_ram_mb_matches(self):
        analyzer = HardwareCapabilityAnalyzer()
        profile = _make_hardware_profile(total_ram_mb=5678)
        caps = analyzer.analyze(profile)
        assert caps.total_ram_mb == 5678

    # 54. Output supports model_dump()
    def test_output_model_dump(self):
        analyzer = HardwareCapabilityAnalyzer()
        caps = analyzer.analyze(_make_hardware_profile())
        assert isinstance(caps.model_dump(), dict)

    # 55. Output supports model_dump_json()
    def test_output_model_dump_json(self):
        analyzer = HardwareCapabilityAnalyzer()
        caps = analyzer.analyze(_make_hardware_profile())
        json_str = caps.model_dump_json()
        assert isinstance(json.loads(json_str), dict)

    # 56. Repeated analysis of identical hardware produces equivalent capability values except capability_id
    def test_repeated_analysis_produces_equivalent_values(self):
        analyzer = HardwareCapabilityAnalyzer()
        profile = _make_hardware_profile()
        c1 = analyzer.analyze(profile)
        c2 = analyzer.analyze(profile)
        d1 = c1.model_dump()
        d2 = c2.model_dump()
        # Pop IDs to compare equality
        d1.pop("capability_id")
        d2.pop("capability_id")
        assert d1 == d2

    # Extra edge-case error wrapping coverage test
    def test_analyzer_exception_wrapping(self):
        analyzer = HardwareCapabilityAnalyzer()
        profile = _make_hardware_profile()
        with patch.object(profile.memory, "total_ram_mb", 0):
            with pytest.raises(HardwareCapabilityAnalysisError, match="Capabilities construction failed"):
                analyzer.analyze(profile)
