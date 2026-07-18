import json
import os
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from backend.app.hardware.detector import (
    HardwareDetectionError,
    HardwareDetector,
)
from backend.app.hardware.schemas import (
    GPUVendor,
    HardwareProfile,
    OperatingSystem,
)


class TestHardwareDetector:
    """Comprehensive unit tests for HardwareDetector."""

    @pytest.fixture
    def base_mocks(self):
        """Configure default mocks that simulate a standard Windows PC."""
        mocks = {
            "system": "Windows",
            "version": "10.0.19045",
            "processor": "Intel64 Family 6 Model 158 Stepping 10, GenuineIntel",
            "machine": "AMD64",
            "py_ver": "3.14.0",
            "py_impl": "CPython",
            "cpu_physical": 6,
            "cpu_logical": 12,
            "total_ram": 17179869184,  # 16 GB in bytes
            "available_ram": 8589934592,  # 8 GB in bytes
            "os_cpu_count": 12,
            "nvidia_smi": None,  # by default, no nvidia-smi
        }
        return mocks

    def _run_detector_with_mocks(self, mock_settings):
        """Helper to run detect() under specified patched environment values."""
        with patch("platform.system", return_value=mock_settings["system"]), \
             patch("platform.version", return_value=mock_settings["version"]), \
             patch("platform.processor", return_value=mock_settings["processor"]), \
             patch("platform.machine", return_value=mock_settings["machine"]), \
             patch("platform.python_version", return_value=mock_settings["py_ver"]), \
             patch("platform.python_implementation", return_value=mock_settings["py_impl"]):

            virtual_mem_mock = MagicMock()
            virtual_mem_mock.total = mock_settings["total_ram"]
            virtual_mem_mock.available = mock_settings["available_ram"]

            with patch("psutil.virtual_memory", return_value=virtual_mem_mock), \
                 patch("psutil.cpu_count", side_effect=lambda logical: (
                     mock_settings["cpu_logical"] if logical else mock_settings["cpu_physical"]
                 )), \
                 patch("os.cpu_count", return_value=mock_settings["os_cpu_count"]), \
                 patch("shutil.which", return_value=mock_settings["nvidia_smi"]):

                detector = HardwareDetector()
                return detector.detect()

    # 1. HardwareDetector can be instantiated
    def test_detector_instantiation(self):
        detector = HardwareDetector()
        assert detector is not None

    # 2. Successful complete detection returns HardwareProfile
    def test_successful_complete_detection(self, base_mocks):
        profile = self._run_detector_with_mocks(base_mocks)
        assert isinstance(profile, HardwareProfile)

    # 3. Every detect() call generates a non-empty profile_id
    def test_profile_id_generated_and_non_empty(self, base_mocks):
        profile = self._run_detector_with_mocks(base_mocks)
        assert profile.profile_id.startswith("hardware_")
        assert len(profile.profile_id) > len("hardware_")

    # 4. Two detect() calls generate different profile IDs
    def test_two_calls_generate_different_ids(self, base_mocks):
        p1 = self._run_detector_with_mocks(base_mocks)
        p2 = self._run_detector_with_mocks(base_mocks)
        assert p1.profile_id != p2.profile_id

    # 5-9. OS Mapping tests
    @pytest.mark.parametrize(
        "sys_val, expected_enum",
        [
            ("Windows", OperatingSystem.WINDOWS),
            ("WINDOWS", OperatingSystem.WINDOWS),
            ("Linux", OperatingSystem.LINUX),
            ("darwin", OperatingSystem.MACOS),
            ("Darwin", OperatingSystem.MACOS),
            ("FreeBSD", OperatingSystem.OTHER),
        ],
    )
    def test_os_mappings(self, base_mocks, sys_val, expected_enum):
        base_mocks["system"] = sys_val
        profile = self._run_detector_with_mocks(base_mocks)
        assert profile.operating_system == expected_enum

    # 10. Empty/invalid OS version causes core detection failure or fallback
    def test_empty_os_version_fallback(self, base_mocks):
        # If platform.version() is empty, the code falls back to platform.release()
        base_mocks["version"] = ""
        with patch("platform.release", return_value="23H2"):
            profile = self._run_detector_with_mocks(base_mocks)
            assert profile.os_version == "23H2"

        # If both version and release are empty, it falls back to platform.system()
        with patch("platform.release", return_value=""):
            profile = self._run_detector_with_mocks(base_mocks)
            assert profile.os_version == "Windows"

        # If system is also empty, it raises error
        with patch("platform.release", return_value=""), \
             patch("platform.system", return_value=""), \
             patch("platform.version", return_value=""):
            detector = HardwareDetector()
            with pytest.raises(HardwareDetectionError, match="Unable to determine operating system version"):
                detector.detect()

    # 11. CPU profile is populated correctly
    def test_cpu_profile_values(self, base_mocks):
        profile = self._run_detector_with_mocks(base_mocks)
        assert profile.cpu.processor_name == base_mocks["processor"]
        assert profile.cpu.architecture == base_mocks["machine"]
        assert profile.cpu.physical_cores == base_mocks["cpu_physical"]
        assert profile.cpu.logical_cores == base_mocks["cpu_logical"]

    # 12-13. platform.processor() fallback tests
    def test_cpu_processor_fallback_uname(self, base_mocks):
        base_mocks["processor"] = ""
        # If platform.processor() is empty, falls back to platform.uname().processor
        uname_mock = MagicMock()
        uname_mock.processor = "Uname CPU Name"
        with patch("platform.uname", return_value=uname_mock):
            profile = self._run_detector_with_mocks(base_mocks)
            assert profile.cpu.processor_name == "Uname CPU Name"

    # 14. Windows PROCESSOR_IDENTIFIER fallback works
    def test_cpu_processor_windows_env_fallback(self, base_mocks):
        base_mocks["processor"] = ""
        base_mocks["system"] = "Windows"
        uname_mock = MagicMock()
        uname_mock.processor = ""
        with patch("platform.uname", return_value=uname_mock), \
             patch.dict(os.environ, {"PROCESSOR_IDENTIFIER": "Intel64 Family 6 env"}):
            profile = self._run_detector_with_mocks(base_mocks)
            assert profile.cpu.processor_name == "Intel64 Family 6 env"

    def test_no_cpu_processor_name_raises_error(self, base_mocks):
        base_mocks["processor"] = ""
        uname_mock = MagicMock()
        uname_mock.processor = ""
        with patch("platform.uname", return_value=uname_mock), patch.dict(os.environ, {}, clear=True):
            detector = HardwareDetector()
            with pytest.raises(HardwareDetectionError, match="Unable to determine CPU processor name"):
                detector.detect()

    # 15. physical core count may be None
    def test_physical_cores_can_be_none(self, base_mocks):
        base_mocks["cpu_physical"] = None
        profile = self._run_detector_with_mocks(base_mocks)
        assert profile.cpu.physical_cores is None

    # 16-17. logical core count fallbacks
    def test_logical_cores_fallback_to_os_cpu_count(self, base_mocks):
        base_mocks["cpu_logical"] = None  # psutil returns None
        base_mocks["os_cpu_count"] = 8    # os.cpu_count returns 8
        profile = self._run_detector_with_mocks(base_mocks)
        assert profile.cpu.logical_cores == 8

    # 18. missing logical core count raises HardwareDetectionError
    def test_missing_logical_cores_raises_error(self, base_mocks):
        base_mocks["cpu_logical"] = None
        base_mocks["os_cpu_count"] = None
        detector = HardwareDetector()
        with pytest.raises(HardwareDetectionError, match="logical CPU core count"):
            self._run_detector_with_mocks(base_mocks)

    # 19. physical cores greater than logical cores are handled safely
    def test_physical_cores_greater_than_logical_handled(self, base_mocks):
        base_mocks["cpu_physical"] = 16
        base_mocks["cpu_logical"] = 8
        # physical cores capped to logical cores
        profile = self._run_detector_with_mocks(base_mocks)
        assert profile.cpu.physical_cores == 8

    # 20. empty architecture is handled safely or raises HardwareDetectionError
    def test_empty_architecture_raises_error(self, base_mocks):
        base_mocks["machine"] = ""
        detector = HardwareDetector()
        with pytest.raises(HardwareDetectionError, match="architecture"):
            self._run_detector_with_mocks(base_mocks)

    # 21-23. RAM conversion tests
    def test_ram_bytes_to_mb_conversion(self, base_mocks):
        base_mocks["total_ram"] = 8589934592  # 8 GB in bytes
        base_mocks["available_ram"] = 4294967296  # 4 GB in bytes
        profile = self._run_detector_with_mocks(base_mocks)
        assert profile.memory.total_ram_mb == 8192
        assert profile.memory.available_ram_mb == 4096

    # 24. invalid total RAM causes HardwareDetectionError
    def test_invalid_total_ram_raises_error(self, base_mocks):
        base_mocks["total_ram"] = 0
        detector = HardwareDetector()
        with pytest.raises(HardwareDetectionError, match="total system RAM.*is invalid"):
            self._run_detector_with_mocks(base_mocks)

    # 25. available RAM greater than total is handled safely as a detection failure
    def test_available_ram_greater_than_total_raises_error(self, base_mocks):
        base_mocks["total_ram"] = 1000 * 1024 * 1024
        base_mocks["available_ram"] = 1001 * 1024 * 1024
        detector = HardwareDetector()
        with pytest.raises(HardwareDetectionError, match="exceeds total system RAM"):
            self._run_detector_with_mocks(base_mocks)

    def test_negative_available_ram_raises_error(self, base_mocks):
        base_mocks["available_ram"] = -1024 * 1024 * 10
        detector = HardwareDetector()
        with pytest.raises(HardwareDetectionError, match="available RAM.*is negative"):
            self._run_detector_with_mocks(base_mocks)

    # 26-29. RuntimeProfile fields
    def test_runtime_profile_values(self, base_mocks):
        profile = self._run_detector_with_mocks(base_mocks)
        assert profile.runtime.python_version == "3.14.0"
        assert profile.runtime.python_implementation == "CPython"
        assert profile.runtime.machine_architecture == "AMD64"

    def test_invalid_runtime_values_raise_error(self, base_mocks):
        base_mocks["py_ver"] = ""
        detector = HardwareDetector()
        with pytest.raises(HardwareDetectionError, match="runtime profile"):
            self._run_detector_with_mocks(base_mocks)

    # 30. nvidia-smi unavailable returns empty GPU list
    def test_nvidia_smi_unavailable_returns_empty_gpu_list(self, base_mocks):
        base_mocks["nvidia_smi"] = None
        profile = self._run_detector_with_mocks(base_mocks)
        assert profile.gpus == []

    # 31. nvidia-smi available with one valid GPU
    def test_nvidia_smi_one_valid_gpu(self, base_mocks):
        base_mocks["nvidia_smi"] = "/usr/bin/nvidia-smi"

        # Mock standard nvidia-smi query output
        query_stdout = "NVIDIA GeForce RTX 3080, 10240, 9128\n"
        # Mock CUDA version output
        cuda_stdout = "NVIDIA-SMI 525.60.13   Driver Version: 525.60.13   CUDA Version: 12.0\n"

        def _sub_run_mock(args, **kwargs):
            mock_res = MagicMock()
            if len(args) > 1 and "--query-gpu=name,memory.total,memory.free" in args[1]:
                mock_res.stdout = query_stdout
                mock_res.returncode = 0
            else:
                mock_res.stdout = cuda_stdout
                mock_res.returncode = 0
            return mock_res

        with patch("subprocess.run", side_effect=_sub_run_mock):
            profile = self._run_detector_with_mocks(base_mocks)
            assert len(profile.gpus) == 1
            gpu = profile.gpus[0]
            assert gpu.name == "NVIDIA GeForce RTX 3080"
            assert gpu.vendor == GPUVendor.NVIDIA
            assert gpu.memory_total_mb == 10240
            assert gpu.memory_available_mb == 9128
            assert gpu.cuda_available is True
            assert gpu.cuda_version == "12.0"

    # 32. multiple NVIDIA GPU rows
    def test_nvidia_smi_multiple_gpus(self, base_mocks):
        base_mocks["nvidia_smi"] = "/usr/bin/nvidia-smi"

        query_stdout = "NVIDIA RTX 4090, 24576, 20000\nNVIDIA RTX 4090, 24576, 18000\n"
        cuda_stdout = "CUDA Version: 12.1"

        def _sub_run_mock(args, **kwargs):
            mock_res = MagicMock()
            if len(args) > 1 and "--query-gpu" in args[1]:
                mock_res.stdout = query_stdout
                mock_res.returncode = 0
            else:
                mock_res.stdout = cuda_stdout
                mock_res.returncode = 0
            return mock_res

        with patch("subprocess.run", side_effect=_sub_run_mock):
            profile = self._run_detector_with_mocks(base_mocks)
            assert len(profile.gpus) == 2
            assert profile.gpus[0].name == "NVIDIA RTX 4090"
            assert profile.gpus[0].memory_available_mb == 20000
            assert profile.gpus[1].memory_available_mb == 18000
            assert profile.gpus[0].cuda_version == "12.1"

    # 35-36. malformed rows parsing and skipping
    def test_nvidia_smi_malformed_row_skipped(self, base_mocks):
        base_mocks["nvidia_smi"] = "/usr/bin/nvidia-smi"

        query_stdout = (
            "NVIDIA RTX 3080, 10240, 9000\n"
            "MALFORMED_ROW_NO_COMMAS\n"
            "NVIDIA RTX 3080, invalid_mem, 8000\n"
            "NVIDIA RTX 3080, 10240, invalid_free\n"
            "NVIDIA RTX 3070, 8192, 7000\n"
        )
        cuda_stdout = "CUDA Version: 11.8"

        def _sub_run_mock(args, **kwargs):
            mock_res = MagicMock()
            if len(args) > 1 and "--query-gpu" in args[1]:
                mock_res.stdout = query_stdout
                mock_res.returncode = 0
            else:
                mock_res.stdout = cuda_stdout
                mock_res.returncode = 0
            return mock_res

        with patch("subprocess.run", side_effect=_sub_run_mock):
            profile = self._run_detector_with_mocks(base_mocks)
            # The two valid rows should still parse successfully
            assert len(profile.gpus) == 2
            assert profile.gpus[0].name == "NVIDIA RTX 3080"
            assert profile.gpus[0].memory_total_mb == 10240
            assert profile.gpus[1].name == "NVIDIA RTX 3070"
            assert profile.gpus[1].memory_total_mb == 8192

    # 37-40. Subprocess failure modes (FileNotFound, Timeout, non-zero code) do not crash detection
    @pytest.mark.parametrize(
        "sub_effect",
        [
            subprocess.TimeoutExpired(cmd=["nvidia-smi"], timeout=10),
            FileNotFoundError("No such file"),
            OSError("Subprocess error"),
            subprocess.SubprocessError("Failure"),
        ],
    )
    def test_nvidia_smi_exceptions_do_not_crash_detection(self, base_mocks, sub_effect):
        base_mocks["nvidia_smi"] = "/usr/bin/nvidia-smi"

        with patch("subprocess.run", side_effect=sub_effect):
            # Core hardware detection should still succeed, with empty GPU list.
            profile = self._run_detector_with_mocks(base_mocks)
            assert isinstance(profile, HardwareProfile)
            assert profile.gpus == []

    def test_nvidia_smi_non_zero_return_code(self, base_mocks):
        base_mocks["nvidia_smi"] = "/usr/bin/nvidia-smi"

        mock_res = MagicMock()
        mock_res.returncode = 1
        mock_res.stdout = "Driver error"

        with patch("subprocess.run", return_value=mock_res):
            profile = self._run_detector_with_mocks(base_mocks)
            assert profile.gpus == []

    # 41-42. CUDA availability conditions
    def test_cuda_unavailable_without_nvidia_smi(self, base_mocks):
        base_mocks["nvidia_smi"] = None
        profile = self._run_detector_with_mocks(base_mocks)
        assert profile.gpus == []

    def test_cuda_unavailable_when_version_regex_fails(self, base_mocks):
        base_mocks["nvidia_smi"] = "/usr/bin/nvidia-smi"

        query_stdout = "NVIDIA RTX 3080, 10240, 9000\n"
        cuda_stdout = "NVIDIA-SMI without standard CUDA version label..."

        def _sub_run_mock(args, **kwargs):
            mock_res = MagicMock()
            if len(args) > 1 and "--query-gpu" in args[1]:
                mock_res.stdout = query_stdout
                mock_res.returncode = 0
            else:
                mock_res.stdout = cuda_stdout
                mock_res.returncode = 0
            return mock_res

        with patch("subprocess.run", side_effect=_sub_run_mock):
            profile = self._run_detector_with_mocks(base_mocks)
            assert len(profile.gpus) == 1
            gpu = profile.gpus[0]
            assert gpu.cuda_available is False
            assert gpu.cuda_version is None

    # 43-46. Serialization checks
    def test_profile_serialization(self, base_mocks):
        profile = self._run_detector_with_mocks(base_mocks)
        dumped = profile.model_dump()
        assert isinstance(dumped, dict)
        assert dumped["operating_system"] == "windows"
        assert isinstance(dumped["cpu"], dict)

        json_str = profile.model_dump_json()
        parsed = json.loads(json_str)
        assert parsed["operating_system"] == "windows"
        assert parsed["cpu"]["processor_name"] == base_mocks["processor"]

    # 48. GPU detection returns a fresh list on each call
    def test_fresh_gpu_list_instances(self, base_mocks):
        base_mocks["nvidia_smi"] = "/usr/bin/nvidia-smi"

        query_stdout = "NVIDIA RTX 3080, 10240, 9000\n"
        cuda_stdout = "CUDA Version: 12.0"

        def _sub_run_mock(args, **kwargs):
            mock_res = MagicMock()
            if "--query-gpu" in args[1]:
                mock_res.stdout = query_stdout
                mock_res.returncode = 0
            else:
                mock_res.stdout = cuda_stdout
                mock_res.returncode = 0
            return mock_res

        with patch("subprocess.run", side_effect=_sub_run_mock):
            detector = HardwareDetector()
            p1 = self._run_detector_with_mocks(base_mocks)
            p2 = self._run_detector_with_mocks(base_mocks)
            assert p1.gpus is not p2.gpus
            assert len(p1.gpus) == 1
            assert len(p2.gpus) == 1
            assert p1.gpus[0] is not p2.gpus[0]

    def test_memory_detection_generic_failure(self):
        with patch("psutil.virtual_memory", side_effect=RuntimeError("Generic RAM Failure")):
            with patch("platform.system", return_value="Windows"), \
                 patch("platform.version", return_value="10.0"), \
                 patch("platform.processor", return_value="Intel"), \
                 patch("platform.machine", return_value="x86_64"), \
                 patch("platform.python_version", return_value="3.14"), \
                 patch("platform.python_implementation", return_value="CPython"), \
                 patch("psutil.cpu_count", return_value=8):
                detector = HardwareDetector()
                with pytest.raises(HardwareDetectionError, match="Failed to detect system RAM"):
                    detector.detect()

    def test_gpus_detection_generic_failure(self, base_mocks):
        with patch.object(HardwareDetector, "_detect_nvidia_gpus", side_effect=RuntimeError("GPU Exception")):
            profile = self._run_detector_with_mocks(base_mocks)
            assert profile.gpus == []

    def test_nvidia_smi_empty_lines_and_validation_error(self, base_mocks):
        base_mocks["nvidia_smi"] = "/usr/bin/nvidia-smi"
        query_stdout = "NVIDIA RTX 3080, 10240, 9000\n\n , 10240, 9000\n"
        cuda_stdout = "CUDA Version: 12.0"
        def _sub_run_mock(args, **kwargs):
            mock_res = MagicMock()
            if len(args) > 1 and "--query-gpu" in args[1]:
                mock_res.stdout = query_stdout
                mock_res.returncode = 0
            else:
                mock_res.stdout = cuda_stdout
                mock_res.returncode = 0
            return mock_res
        with patch("subprocess.run", side_effect=_sub_run_mock):
            profile = self._run_detector_with_mocks(base_mocks)
            assert len(profile.gpus) == 1
            assert profile.gpus[0].name == "NVIDIA RTX 3080"

    def test_query_cuda_info_exception(self, base_mocks):
        base_mocks["nvidia_smi"] = "/usr/bin/nvidia-smi"
        query_stdout = "NVIDIA RTX 3080, 10240, 9000\n"
        def _sub_run_mock(args, **kwargs):
            if len(args) == 1:
                raise RuntimeError("CUDA query crashed")
            mock_res = MagicMock()
            mock_res.stdout = query_stdout
            mock_res.returncode = 0
            return mock_res
        with patch("subprocess.run", side_effect=_sub_run_mock):
            profile = self._run_detector_with_mocks(base_mocks)
            assert len(profile.gpus) == 1
            assert profile.gpus[0].cuda_available is False
            assert profile.gpus[0].cuda_version is None

