"""Tests for backend.app.compute_capabilities.schemas.

Covers all 41 specified test conditions for ComputeCapabilities and ResourceWarning.
"""

import json
import pytest
from pydantic import ValidationError

from backend.app.compute_capabilities.schemas import (
    AcceleratorType,
    ComputeCapabilities,
    ComputeTier,
    MemoryConstraintLevel,
    ResourceWarning,
)


# ── Helper Constructors ────────────────────────────────────────────────


def _make_valid_capabilities_dict(**overrides) -> dict:
    """Return a minimal valid CPU-only capabilities dictionary."""
    base = {
        "capability_id": "cap-123",
        "hardware_profile_id": "hw-123",
        "compute_tier": ComputeTier.STANDARD,
        "memory_constraint": MemoryConstraintLevel.MODERATE,
        "cpu_training_available": True,
        "gpu_acceleration_available": False,
        "accelerator_type": AcceleratorType.NONE,
        "safe_parallel_workers": 2,
        "max_parallel_workers": 4,
        "available_ram_mb_snapshot": 8192,
        "total_ram_mb": 16384,
    }
    base.update(overrides)
    return base


# ── Tests ──────────────────────────────────────────────────────────────


class TestComputeCapabilitiesEnums:
    """Tests covering enum values and compatibility."""

    # 1. All MemoryConstraintLevel values work
    @pytest.mark.parametrize(
        "val", ["severe", "constrained", "moderate", "comfortable"]
    )
    def test_memory_constraint_levels(self, val):
        assert MemoryConstraintLevel(val) == val

    # 2. All ComputeTier values work
    @pytest.mark.parametrize("val", ["minimal", "standard", "high"])
    def test_compute_tiers(self, val):
        assert ComputeTier(val) == val

    # 3. All AcceleratorType values work
    @pytest.mark.parametrize("val", ["none", "cuda"])
    def test_accelerator_types(self, val):
        assert AcceleratorType(val) == val


class TestResourceWarningSchema:
    """Tests covering the ResourceWarning sub-schema."""

    # 4. Valid warning
    def test_valid_warning(self):
        warn = ResourceWarning(code="LOW_MEM", message="System memory is low")
        assert warn.code == "LOW_MEM"
        assert warn.message == "System memory is low"

    # 5. Surrounding whitespace is stripped
    def test_warning_whitespace_stripped(self):
        warn = ResourceWarning(code="  CODE_A  \n", message="  Some warning\t")
        assert warn.code == "CODE_A"
        assert warn.message == "Some warning"

    # 6. Empty code rejected
    def test_empty_code_rejected(self):
        with pytest.raises(ValidationError, match="Field cannot be empty"):
            ResourceWarning(code="", message="msg")

    # 7. Whitespace-only code rejected
    def test_whitespace_only_code_rejected(self):
        with pytest.raises(ValidationError, match="Field cannot be empty"):
            ResourceWarning(code="   ", message="msg")

    # 8. Empty message rejected
    def test_empty_message_rejected(self):
        with pytest.raises(ValidationError, match="Field cannot be empty"):
            ResourceWarning(code="CODE", message="")

    # 9. Whitespace-only message rejected
    def test_whitespace_only_message_rejected(self):
        with pytest.raises(ValidationError, match="Field cannot be empty"):
            ResourceWarning(code="CODE", message="  \t  ")

    # 10. Non-string values rejected
    def test_non_string_values_rejected(self):
        with pytest.raises(ValidationError, match="Field must be a string"):
            ResourceWarning(code=123, message="msg")
        with pytest.raises(ValidationError, match="Field must be a string"):
            ResourceWarning(code="CODE", message=True)


class TestComputeCapabilitiesSchema:
    """Tests covering the main ComputeCapabilities Pydantic model."""

    # 11. Minimal valid CPU-only capability object
    def test_minimal_valid_cpu_only_capabilities(self):
        caps = ComputeCapabilities(**_make_valid_capabilities_dict())
        assert caps.capability_id == "cap-123"
        assert caps.gpu_acceleration_available is False
        assert caps.accelerator_type == AcceleratorType.NONE
        assert caps.warnings == []

    # 12. Valid CUDA capability object
    def test_valid_cuda_capabilities(self):
        caps = ComputeCapabilities(
            **_make_valid_capabilities_dict(
                gpu_acceleration_available=True,
                accelerator_type=AcceleratorType.CUDA,
            )
        )
        assert caps.gpu_acceleration_available is True
        assert caps.accelerator_type == AcceleratorType.CUDA

    # 13. capability_id whitespace is stripped
    def test_capability_id_whitespace_stripped(self):
        caps = ComputeCapabilities(
            **_make_valid_capabilities_dict(capability_id="  cap-strip  \n")
        )
        assert caps.capability_id == "cap-strip"

    # 14. hardware_profile_id whitespace is stripped
    def test_hardware_profile_id_whitespace_stripped(self):
        caps = ComputeCapabilities(
            **_make_valid_capabilities_dict(hardware_profile_id="\thw-strip ")
        )
        assert caps.hardware_profile_id == "hw-strip"

    # 15. Empty capability_id rejected
    def test_empty_capability_id_rejected(self):
        with pytest.raises(ValidationError, match="Field cannot be empty"):
            ComputeCapabilities(**_make_valid_capabilities_dict(capability_id=""))
        with pytest.raises(ValidationError, match="Field cannot be empty"):
            ComputeCapabilities(**_make_valid_capabilities_dict(capability_id="  "))

    # 16. Empty hardware_profile_id rejected
    def test_empty_hardware_profile_id_rejected(self):
        with pytest.raises(ValidationError, match="Field cannot be empty"):
            ComputeCapabilities(**_make_valid_capabilities_dict(hardware_profile_id=""))
        with pytest.raises(ValidationError, match="Field cannot be empty"):
            ComputeCapabilities(**_make_valid_capabilities_dict(hardware_profile_id="   "))

    # 17. safe_parallel_workers < 1 rejected
    def test_safe_workers_below_one_rejected(self):
        with pytest.raises(ValidationError, match="safe_parallel_workers"):
            ComputeCapabilities(**_make_valid_capabilities_dict(safe_parallel_workers=0))

    # 18. max_parallel_workers < 1 rejected
    def test_max_workers_below_one_rejected(self):
        with pytest.raises(ValidationError, match="max_parallel_workers"):
            ComputeCapabilities(**_make_valid_capabilities_dict(max_parallel_workers=0))

    # 19. safe_parallel_workers > max_parallel_workers rejected
    def test_safe_workers_greater_than_max_rejected(self):
        with pytest.raises(ValidationError, match="cannot exceed max_parallel_workers"):
            ComputeCapabilities(
                **_make_valid_capabilities_dict(
                    safe_parallel_workers=5, max_parallel_workers=4
                )
            )

    # 20. available_ram_mb_snapshot < 0 rejected
    def test_available_ram_below_zero_rejected(self):
        with pytest.raises(ValidationError, match="available_ram_mb_snapshot"):
            ComputeCapabilities(
                **_make_valid_capabilities_dict(available_ram_mb_snapshot=-1)
            )

    # 21. total_ram_mb < 1 rejected
    def test_total_ram_below_one_rejected(self):
        with pytest.raises(ValidationError, match="total_ram_mb"):
            ComputeCapabilities(**_make_valid_capabilities_dict(total_ram_mb=0))

    # 22. available RAM greater than total RAM rejected
    def test_available_ram_greater_than_total_rejected(self):
        with pytest.raises(ValidationError, match="cannot exceed total_ram_mb"):
            ComputeCapabilities(
                **_make_valid_capabilities_dict(
                    available_ram_mb_snapshot=1001, total_ram_mb=1000
                )
            )

    # 23. Zero available RAM accepted
    def test_zero_available_ram_accepted(self):
        caps = ComputeCapabilities(
            **_make_valid_capabilities_dict(available_ram_mb_snapshot=0)
        )
        assert caps.available_ram_mb_snapshot == 0

    # 24. gpu_acceleration_available=False + accelerator_type=NONE accepted
    def test_gpu_false_accelerator_none_accepted(self):
        caps = ComputeCapabilities(
            **_make_valid_capabilities_dict(
                gpu_acceleration_available=False,
                accelerator_type=AcceleratorType.NONE,
            )
        )
        assert caps.gpu_acceleration_available is False
        assert caps.accelerator_type == AcceleratorType.NONE

    # 25. gpu_acceleration_available=True + accelerator_type=CUDA accepted
    def test_gpu_true_accelerator_cuda_accepted(self):
        caps = ComputeCapabilities(
            **_make_valid_capabilities_dict(
                gpu_acceleration_available=True,
                accelerator_type=AcceleratorType.CUDA,
            )
        )
        assert caps.gpu_acceleration_available is True
        assert caps.accelerator_type == AcceleratorType.CUDA

    # 26. gpu_acceleration_available=False + accelerator_type=CUDA rejected
    def test_gpu_false_accelerator_cuda_rejected(self):
        with pytest.raises(ValidationError, match="must be 'AcceleratorType.NONE' when gpu_acceleration_available is False"):
            ComputeCapabilities(
                **_make_valid_capabilities_dict(
                    gpu_acceleration_available=False,
                    accelerator_type=AcceleratorType.CUDA,
                )
            )

    # 27. gpu_acceleration_available=True + accelerator_type=NONE rejected
    def test_gpu_true_accelerator_none_rejected(self):
        with pytest.raises(ValidationError, match="cannot be 'AcceleratorType.NONE' when gpu_acceleration_available is True"):
            ComputeCapabilities(
                **_make_valid_capabilities_dict(
                    gpu_acceleration_available=True,
                    accelerator_type=AcceleratorType.NONE,
                )
            )

    # 28. cpu_training_available=False is accepted
    def test_cpu_training_false_accepted(self):
        caps = ComputeCapabilities(
            **_make_valid_capabilities_dict(cpu_training_available=False)
        )
        assert caps.cpu_training_available is False

    # 29. Default warnings list is empty
    def test_default_warnings_empty(self):
        caps = ComputeCapabilities(**_make_valid_capabilities_dict())
        assert caps.warnings == []

    # 30. Two instances do not share warnings lists
    def test_warnings_list_not_shared(self):
        c1 = ComputeCapabilities(**_make_valid_capabilities_dict())
        c2 = ComputeCapabilities(**_make_valid_capabilities_dict())
        c1.warnings.append(ResourceWarning(code="W", message="M"))
        assert len(c1.warnings) == 1
        assert len(c2.warnings) == 0

    # 31. One warning serializes correctly
    def test_one_warning_serialization(self):
        warn = ResourceWarning(code="WARN", message="Warning msg")
        caps = ComputeCapabilities(
            **_make_valid_capabilities_dict(warnings=[warn])
        )
        json_str = caps.model_dump_json()
        parsed = json.loads(json_str)
        assert len(parsed["warnings"]) == 1
        assert parsed["warnings"][0]["code"] == "WARN"

    # 32. Multiple warnings serialize correctly
    def test_multiple_warnings_serialization(self):
        warnings = [
            ResourceWarning(code="W1", message="M1"),
            ResourceWarning(code="W2", message="M2"),
        ]
        caps = ComputeCapabilities(
            **_make_valid_capabilities_dict(warnings=warnings)
        )
        json_str = caps.model_dump_json()
        parsed = json.loads(json_str)
        assert len(parsed["warnings"]) == 2
        assert parsed["warnings"][0]["code"] == "W1"
        assert parsed["warnings"][1]["code"] == "W2"

    # 33. model_dump() works
    def test_model_dump(self):
        caps = ComputeCapabilities(**_make_valid_capabilities_dict())
        dumped = caps.model_dump()
        assert isinstance(dumped, dict)
        assert dumped["capability_id"] == "cap-123"

    # 34. model_dump_json() produces valid JSON
    def test_model_dump_json(self):
        caps = ComputeCapabilities(**_make_valid_capabilities_dict())
        json_str = caps.model_dump_json()
        parsed = json.loads(json_str)
        assert parsed["capability_id"] == "cap-123"

    # 35. Enum values serialize as strings in JSON
    def test_enum_serialization_as_strings(self):
        caps = ComputeCapabilities(
            **_make_valid_capabilities_dict(
                compute_tier=ComputeTier.HIGH,
                memory_constraint=MemoryConstraintLevel.SEVERE,
            )
        )
        json_str = caps.model_dump_json()
        parsed = json.loads(json_str)
        assert parsed["compute_tier"] == "high"
        assert parsed["memory_constraint"] == "severe"

    # 36. hardware_profile_id is preserved
    def test_hardware_profile_id_preserved(self):
        caps = ComputeCapabilities(**_make_valid_capabilities_dict(hardware_profile_id="hw-999"))
        assert caps.hardware_profile_id == "hw-999"

    # 37. available_ram_mb_snapshot is preserved
    def test_available_ram_preserved(self):
        caps = ComputeCapabilities(**_make_valid_capabilities_dict(available_ram_mb_snapshot=4096))
        assert caps.available_ram_mb_snapshot == 4096

    # 38. total_ram_mb is preserved
    def test_total_ram_preserved(self):
        caps = ComputeCapabilities(**_make_valid_capabilities_dict(total_ram_mb=8192))
        assert caps.total_ram_mb == 8192

    # 39. safe_parallel_workers boundary equal to max_parallel_workers is accepted
    def test_safe_workers_equal_max_accepted(self):
        caps = ComputeCapabilities(
            **_make_valid_capabilities_dict(
                safe_parallel_workers=4, max_parallel_workers=4
            )
        )
        assert caps.safe_parallel_workers == 4
        assert caps.max_parallel_workers == 4

    # 40. Valid minimal-tier/severe-memory combination is accepted
    def test_minimal_severe_combination_accepted(self):
        caps = ComputeCapabilities(
            **_make_valid_capabilities_dict(
                compute_tier=ComputeTier.MINIMAL,
                memory_constraint=MemoryConstraintLevel.SEVERE,
            )
        )
        assert caps.compute_tier == ComputeTier.MINIMAL
        assert caps.memory_constraint == MemoryConstraintLevel.SEVERE

    # 41. Valid high-tier/comfortable-memory combination is accepted
    def test_high_comfortable_combination_accepted(self):
        caps = ComputeCapabilities(
            **_make_valid_capabilities_dict(
                compute_tier=ComputeTier.HIGH,
                memory_constraint=MemoryConstraintLevel.COMFORTABLE,
            )
        )
        assert caps.compute_tier == ComputeTier.HIGH
        assert caps.memory_constraint == MemoryConstraintLevel.COMFORTABLE

    # Additional coverage test for type checking validation
    def test_invalid_types_rejected(self):
        with pytest.raises(ValidationError, match="Field must be a string"):
            ComputeCapabilities(**_make_valid_capabilities_dict(capability_id=123))
