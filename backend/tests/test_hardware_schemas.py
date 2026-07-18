"""Tests for backend.app.hardware.schemas.

Covers all 44 specified test conditions for HardwareProfile, GPUDevice,
CPUProfile, MemoryProfile, and RuntimeProfile.
"""

import json
import pytest
from pydantic import ValidationError

from backend.app.hardware.schemas import (
    CPUProfile,
    GPUDevice,
    GPUVendor,
    HardwareProfile,
    MemoryProfile,
    OperatingSystem,
    RuntimeProfile,
)


# ── Helper Fixtures/Constructors ───────────────────────────────────────


def _make_gpu(**overrides) -> dict:
    """Helper to return a valid GPUDevice dict."""
    base = {
        "name": "NVIDIA RTX 4060 Laptop GPU",
        "vendor": GPUVendor.NVIDIA,
        "memory_total_mb": 8192,
        "memory_available_mb": 4096,
        "cuda_available": True,
        "cuda_version": "12.1",
    }
    base.update(overrides)
    return base


def _make_cpu(**overrides) -> dict:
    """Helper to return a valid CPUProfile dict."""
    base = {
        "processor_name": "Intel Core i7-12700H",
        "architecture": "x86_64",
        "physical_cores": 14,
        "logical_cores": 20,
    }
    base.update(overrides)
    return base


def _make_memory(**overrides) -> dict:
    """Helper to return a valid MemoryProfile dict."""
    base = {
        "total_ram_mb": 16384,
        "available_ram_mb": 8192,
    }
    base.update(overrides)
    return base


def _make_runtime(**overrides) -> dict:
    """Helper to return a valid RuntimeProfile dict."""
    base = {
        "python_version": "3.14.0",
        "python_implementation": "CPython",
        "machine_architecture": "AMD64",
    }
    base.update(overrides)
    return base


def _make_hardware_profile(**overrides) -> dict:
    """Helper to return a valid HardwareProfile dict."""
    base = {
        "profile_id": "prof-123",
        "operating_system": OperatingSystem.WINDOWS,
        "os_version": "Windows 11 Home 23H2",
        "cpu": _make_cpu(),
        "memory": _make_memory(),
        "gpus": [_make_gpu()],
        "runtime": _make_runtime(),
    }
    base.update(overrides)
    return base


# ── Tests ──────────────────────────────────────────────────────────────


class TestGPUDeviceSchemas:
    """Unit tests for GPUDevice schema."""

    # 1. Minimal valid GPU
    def test_minimal_valid_gpu(self):
        gpu = GPUDevice(name="Intel Iris Xe")
        assert gpu.name == "Intel Iris Xe"
        assert gpu.vendor == GPUVendor.UNKNOWN
        assert gpu.memory_total_mb is None
        assert gpu.memory_available_mb is None
        assert gpu.cuda_available is False
        assert gpu.cuda_version is None

    # 2. Fully populated NVIDIA CUDA GPU
    def test_fully_populated_nvidia_cuda_gpu(self):
        gpu = GPUDevice(**_make_gpu())
        assert gpu.vendor == GPUVendor.NVIDIA
        assert gpu.cuda_available is True
        assert gpu.cuda_version == "12.1"

    # 3. Non-CUDA Intel GPU
    def test_non_cuda_intel_gpu(self):
        gpu = GPUDevice(
            name="Intel Arc A770",
            vendor=GPUVendor.INTEL,
            memory_total_mb=16384,
            memory_available_mb=16000,
            cuda_available=False,
            cuda_version=None,
        )
        assert gpu.vendor == GPUVendor.INTEL
        assert gpu.cuda_available is False
        assert gpu.cuda_version is None

    # 4. All GPUVendor enum values
    @pytest.mark.parametrize(
        "vendor", ["nvidia", "amd", "intel", "apple", "other", "unknown"]
    )
    def test_all_gpu_vendor_enum_values(self, vendor):
        gpu = GPUDevice(name="GPU", vendor=vendor)
        assert gpu.vendor == vendor

    # 5. Empty GPU name rejected
    def test_empty_gpu_name_rejected(self):
        with pytest.raises(ValidationError, match="GPU name"):
            GPUDevice(name="")

    # 6. Whitespace-only GPU name rejected
    def test_whitespace_only_gpu_name_rejected(self):
        with pytest.raises(ValidationError, match="GPU name"):
            GPUDevice(name="   ")

    # 7. Negative total GPU memory rejected
    def test_negative_total_gpu_memory_rejected(self):
        with pytest.raises(ValidationError, match="memory_total_mb"):
            GPUDevice(name="GPU", memory_total_mb=-100)

    # 8. Negative available GPU memory rejected
    def test_negative_available_gpu_memory_rejected(self):
        with pytest.raises(ValidationError, match="memory_available_mb"):
            GPUDevice(name="GPU", memory_available_mb=-1)

    # 9. Available GPU memory greater than total rejected
    def test_available_memory_greater_than_total_rejected(self):
        with pytest.raises(ValidationError, match="cannot exceed total memory"):
            GPUDevice(name="GPU", memory_total_mb=1000, memory_available_mb=1001)

    # 10. cuda_available=False with cuda_version provided rejected
    def test_cuda_unavailable_with_version_rejected(self):
        with pytest.raises(ValidationError, match="CUDA version must be None"):
            GPUDevice(name="GPU", cuda_available=False, cuda_version="11.8")

    # 11. cuda_available=True without cuda_version accepted
    def test_cuda_available_without_version_accepted(self):
        gpu = GPUDevice(name="GPU", cuda_available=True, cuda_version=None)
        assert gpu.cuda_available is True
        assert gpu.cuda_version is None

    # 12. cuda_available=True with cuda_version accepted
    def test_cuda_available_with_version_accepted(self):
        gpu = GPUDevice(name="GPU", cuda_available=True, cuda_version="12.0")
        assert gpu.cuda_available is True
        assert gpu.cuda_version == "12.0"

    # Validator type coverage checks
    def test_gpu_invalid_types_rejected(self):
        with pytest.raises(ValidationError, match="GPU name must be a string"):
            GPUDevice(name=123)
        with pytest.raises(ValidationError, match="CUDA version must be a string"):
            GPUDevice(name="GPU", cuda_available=True, cuda_version=12.1)
        with pytest.raises(ValidationError, match="CUDA version cannot be empty"):
            GPUDevice(name="GPU", cuda_available=True, cuda_version="  ")


class TestCPUProfileSchemas:
    """Unit tests for CPUProfile schema."""

    # 13. Valid CPU profile
    def test_valid_cpu_profile(self):
        cpu = CPUProfile(**_make_cpu())
        assert cpu.processor_name == "Intel Core i7-12700H"
        assert cpu.architecture == "x86_64"
        assert cpu.physical_cores == 14
        assert cpu.logical_cores == 20

    # 14. CPU with physical_cores=None accepted
    def test_cpu_physical_cores_none_accepted(self):
        cpu = CPUProfile(
            processor_name="Apple M3",
            architecture="arm64",
            physical_cores=None,
            logical_cores=8,
        )
        assert cpu.physical_cores is None
        assert cpu.logical_cores == 8

    # 15. Empty processor_name rejected
    def test_empty_processor_name_rejected(self):
        with pytest.raises(ValidationError, match="Field cannot be empty"):
            CPUProfile(**_make_cpu(processor_name=""))
        with pytest.raises(ValidationError, match="Field cannot be empty"):
            CPUProfile(**_make_cpu(processor_name="   "))

    # 16. Empty architecture rejected
    def test_empty_architecture_rejected(self):
        with pytest.raises(ValidationError, match="Field cannot be empty"):
            CPUProfile(**_make_cpu(architecture=""))
        with pytest.raises(ValidationError, match="Field cannot be empty"):
            CPUProfile(**_make_cpu(architecture="   "))

    # 17. physical_cores < 1 rejected
    def test_physical_cores_below_one_rejected(self):
        with pytest.raises(ValidationError, match="physical_cores"):
            CPUProfile(**_make_cpu(physical_cores=0))

    # 18. logical_cores < 1 rejected
    def test_logical_cores_below_one_rejected(self):
        with pytest.raises(ValidationError, match="logical_cores"):
            CPUProfile(**_make_cpu(logical_cores=0))

    # 19. physical_cores greater than logical_cores rejected
    def test_physical_cores_greater_than_logical_rejected(self):
        with pytest.raises(ValidationError, match="cannot exceed logical cores"):
            CPUProfile(**_make_cpu(physical_cores=8, logical_cores=4))

    # Validator type coverage checks
    def test_cpu_invalid_types_rejected(self):
        with pytest.raises(ValidationError, match="Field must be a string"):
            CPUProfile(**_make_cpu(processor_name=123))


class TestMemoryProfileSchemas:
    """Unit tests for MemoryProfile schema."""

    # 20. Valid memory profile
    def test_valid_memory_profile(self):
        mem = MemoryProfile(**_make_memory())
        assert mem.total_ram_mb == 16384
        assert mem.available_ram_mb == 8192

    # 21. total_ram_mb < 1 rejected
    def test_total_ram_below_one_rejected(self):
        with pytest.raises(ValidationError, match="total_ram_mb"):
            MemoryProfile(total_ram_mb=0, available_ram_mb=0)

    # 22. negative available RAM rejected
    def test_negative_available_ram_rejected(self):
        with pytest.raises(ValidationError, match="available_ram_mb"):
            MemoryProfile(total_ram_mb=1000, available_ram_mb=-10)

    # 23. available RAM greater than total rejected
    def test_available_ram_greater_than_total_rejected(self):
        with pytest.raises(ValidationError, match="cannot exceed total RAM"):
            MemoryProfile(total_ram_mb=16000, available_ram_mb=16001)

    # 24. zero available RAM accepted
    def test_zero_available_ram_accepted(self):
        mem = MemoryProfile(total_ram_mb=8192, available_ram_mb=0)
        assert mem.available_ram_mb == 0


class TestRuntimeProfileSchemas:
    """Unit tests for RuntimeProfile schema."""

    # 25. Valid runtime profile
    def test_valid_runtime_profile(self):
        rt = RuntimeProfile(**_make_runtime())
        assert rt.python_version == "3.14.0"
        assert rt.python_implementation == "CPython"
        assert rt.machine_architecture == "AMD64"

    # 26. Empty python_version rejected
    def test_empty_python_version_rejected(self):
        with pytest.raises(ValidationError, match="Field cannot be empty"):
            RuntimeProfile(**_make_runtime(python_version=""))

    # 27. Empty python_implementation rejected
    def test_empty_python_implementation_rejected(self):
        with pytest.raises(ValidationError, match="Field cannot be empty"):
            RuntimeProfile(**_make_runtime(python_implementation="  "))

    # 28. Empty machine_architecture rejected
    def test_empty_machine_architecture_rejected(self):
        with pytest.raises(ValidationError, match="Field cannot be empty"):
            RuntimeProfile(**_make_runtime(machine_architecture=""))

    # 29. Surrounding whitespace normalization works
    def test_surrounding_whitespace_normalization(self):
        rt = RuntimeProfile(
            python_version="  3.14.0  \n",
            python_implementation="\tCPython ",
            machine_architecture=" AMD64\t",
        )
        assert rt.python_version == "3.14.0"
        assert rt.python_implementation == "CPython"
        assert rt.machine_architecture == "AMD64"

    # Validator type coverage checks
    def test_runtime_invalid_types_rejected(self):
        with pytest.raises(ValidationError, match="Field must be a string"):
            RuntimeProfile(**_make_runtime(python_version=3.14))


class TestHardwareProfileSchemas:
    """Unit tests for the main HardwareProfile schema."""

    # 30. Minimal valid hardware profile with no GPU
    def test_minimal_valid_hardware_profile_no_gpu(self):
        profile = HardwareProfile(**_make_hardware_profile(gpus=[]))
        assert profile.profile_id == "prof-123"
        assert profile.operating_system == OperatingSystem.WINDOWS
        assert profile.gpus == []

    # 31. Hardware profile with one GPU
    def test_hardware_profile_with_one_gpu(self):
        profile = HardwareProfile(**_make_hardware_profile(gpus=[_make_gpu()]))
        assert len(profile.gpus) == 1

    # 32. Hardware profile with multiple GPUs
    def test_hardware_profile_with_multiple_gpus(self):
        gpus = [_make_gpu(name="GPU 1"), _make_gpu(name="GPU 2")]
        profile = HardwareProfile(**_make_hardware_profile(gpus=gpus))
        assert len(profile.gpus) == 2
        assert profile.gpus[0].name == "GPU 1"
        assert profile.gpus[1].name == "GPU 2"

    # 33. Default gpus list is empty
    def test_default_gpus_list_is_empty(self):
        kwargs = _make_hardware_profile()
        del kwargs["gpus"]
        profile = HardwareProfile(**kwargs)
        assert profile.gpus == []

    # 34. Two HardwareProfile instances do not share the same GPU list
    def test_gpu_lists_are_not_shared_references(self):
        kwargs1 = _make_hardware_profile()
        kwargs2 = _make_hardware_profile()
        del kwargs1["gpus"]
        del kwargs2["gpus"]
        p1 = HardwareProfile(**kwargs1)
        p2 = HardwareProfile(**kwargs2)
        p1.gpus.append(GPUDevice(name="TempGPU"))
        assert len(p1.gpus) == 1
        assert len(p2.gpus) == 0

    # 35. Empty profile_id rejected
    def test_empty_profile_id_rejected(self):
        with pytest.raises(ValidationError, match="Field cannot be empty"):
            HardwareProfile(**_make_hardware_profile(profile_id=""))
        with pytest.raises(ValidationError, match="Field cannot be empty"):
            HardwareProfile(**_make_hardware_profile(profile_id="   "))

    # 36. Empty os_version rejected
    def test_empty_os_version_rejected(self):
        with pytest.raises(ValidationError, match="Field cannot be empty"):
            HardwareProfile(**_make_hardware_profile(os_version=""))
        with pytest.raises(ValidationError, match="Field cannot be empty"):
            HardwareProfile(**_make_hardware_profile(os_version="   "))

    # 37. All OperatingSystem enum values work
    @pytest.mark.parametrize(
        "os_name", ["windows", "linux", "macos", "other"]
    )
    def test_all_os_enum_values(self, os_name):
        profile = HardwareProfile(**_make_hardware_profile(operating_system=os_name))
        assert profile.operating_system == os_name

    # 38. model_dump() works
    def test_model_dump(self):
        profile = HardwareProfile(**_make_hardware_profile())
        dumped = profile.model_dump()
        assert isinstance(dumped, dict)
        assert dumped["profile_id"] == "prof-123"
        assert dumped["operating_system"] == "windows"

    # 39. model_dump_json() produces valid JSON
    def test_model_dump_json(self):
        profile = HardwareProfile(**_make_hardware_profile())
        json_str = profile.model_dump_json()
        parsed = json.loads(json_str)
        assert parsed["profile_id"] == "prof-123"

    # 40. Enum values serialize as strings
    def test_enums_serialize_as_strings(self):
        profile = HardwareProfile(**_make_hardware_profile())
        json_str = profile.model_dump_json()
        parsed = json.loads(json_str)
        assert parsed["operating_system"] == "windows"
        assert parsed["gpus"][0]["vendor"] == "nvidia"

    # 41. Nested CPU serializes correctly
    def test_nested_cpu_serializes(self):
        profile = HardwareProfile(**_make_hardware_profile())
        json_str = profile.model_dump_json()
        parsed = json.loads(json_str)
        assert parsed["cpu"]["processor_name"] == "Intel Core i7-12700H"
        assert parsed["cpu"]["logical_cores"] == 20

    # 42. Nested memory serializes correctly
    def test_nested_memory_serializes(self):
        profile = HardwareProfile(**_make_hardware_profile())
        json_str = profile.model_dump_json()
        parsed = json.loads(json_str)
        assert parsed["memory"]["total_ram_mb"] == 16384
        assert parsed["memory"]["available_ram_mb"] == 8192

    # 43. Nested runtime serializes correctly
    def test_nested_runtime_serializes(self):
        profile = HardwareProfile(**_make_hardware_profile())
        json_str = profile.model_dump_json()
        parsed = json.loads(json_str)
        assert parsed["runtime"]["python_version"] == "3.14.0"
        assert parsed["runtime"]["python_implementation"] == "CPython"

    # 44. Nested GPU list serializes correctly
    def test_nested_gpu_list_serializes(self):
        profile = HardwareProfile(**_make_hardware_profile())
        json_str = profile.model_dump_json()
        parsed = json.loads(json_str)
        assert isinstance(parsed["gpus"], list)
        assert len(parsed["gpus"]) == 1
        assert parsed["gpus"][0]["name"] == "NVIDIA RTX 4060 Laptop GPU"

    # Validator type coverage checks
    def test_profile_invalid_types_rejected(self):
        with pytest.raises(ValidationError, match="Field must be a string"):
            HardwareProfile(**_make_hardware_profile(profile_id=123))
