"""HardwareDetector implementation for system facts discovery.

Detects operating system, CPU, memory, and optional GPU information.
"""

from __future__ import annotations

import logging
import os
import platform
import re
import shutil
import subprocess
import uuid
from typing import Any, List, Optional

import psutil
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

logger = logging.getLogger(__name__)


class HardwareDetectionError(Exception):
    """Raised when core hardware detection fails or violates validation constraints."""


class HardwareDetector:
    """Detects local hardware facts and runtime platform specifications."""

    def detect(self) -> HardwareProfile:
        """Execute hardware detection and return a validated HardwareProfile.

        Returns
        -------
        HardwareProfile
            Structured, validated snapshot of detected system specifications.

        Raises
        ------
        HardwareDetectionError
            If core system information (OS, CPU, memory, runtime) cannot be
            reliably detected or fails validation.
        """
        logger.info("Executing system hardware detection...")

        # Generate a unique profile ID for every call.
        profile_id = f"hardware_{uuid.uuid4().hex}"

        # Detect core components.
        operating_system, os_version = self._detect_operating_system()
        cpu = self._detect_cpu()
        memory = self._detect_memory()
        runtime = self._detect_runtime()

        # Best-effort optional GPU detection.
        gpus = self._detect_gpus()

        profile = HardwareProfile(
            profile_id=profile_id,
            operating_system=operating_system,
            os_version=os_version,
            cpu=cpu,
            memory=memory,
            gpus=gpus,
            runtime=runtime,
        )

        logger.info(
            "Hardware detection complete. profile_id=%r OS=%s CPU=%s RAM=%dMB GPUCount=%d",
            profile.profile_id,
            profile.operating_system,
            profile.cpu.processor_name,
            profile.memory.total_ram_mb,
            len(profile.gpus),
        )
        return profile

    # ── private core detection helpers ──────────────────────────────────

    def _detect_operating_system(self) -> tuple[OperatingSystem, str]:
        """Detect operating system family and version."""
        sys_name = platform.system().lower()
        if "windows" in sys_name:
            os_family = OperatingSystem.WINDOWS
        elif "linux" in sys_name:
            os_family = OperatingSystem.LINUX
        elif "darwin" in sys_name:
            os_family = OperatingSystem.MACOS
        else:
            os_family = OperatingSystem.OTHER

        version = platform.version().strip()
        if not version:
            # Fallback to platform release or system name if version is blank.
            version = platform.release().strip() or platform.system().strip()

        if not version:
            raise HardwareDetectionError("Unable to determine operating system version.")

        return os_family, version

    def _detect_cpu(self) -> CPUProfile:
        """Detect CPU name, architecture, and core counts."""
        # 1. Determine processor name
        processor_name = platform.processor().strip()
        if not processor_name:
            processor_name = platform.uname().processor.strip()

        # Windows env var fallback
        if not processor_name and platform.system().lower() == "windows":
            processor_name = os.environ.get("PROCESSOR_IDENTIFIER", "").strip()

        if not processor_name:
            raise HardwareDetectionError("Unable to determine CPU processor name.")

        # 2. Determine architecture
        architecture = platform.machine().strip()
        if not architecture:
            raise HardwareDetectionError("Unable to determine CPU hardware architecture.")

        # 3. Core counts using psutil with fallback to os.cpu_count
        logical_cores = psutil.cpu_count(logical=True)
        if logical_cores is None or logical_cores < 1:
            logical_cores = os.cpu_count()

        if logical_cores is None or logical_cores < 1:
            raise HardwareDetectionError("Unable to determine logical CPU core count.")

        physical_cores = psutil.cpu_count(logical=False)

        # Handle physical cores greater than logical cores (e.g. nested virtualization oddities)
        if physical_cores is not None and physical_cores > logical_cores:
            physical_cores = logical_cores

        return CPUProfile(
            processor_name=processor_name,
            architecture=architecture,
            physical_cores=physical_cores,
            logical_cores=logical_cores,
        )

    def _detect_memory(self) -> MemoryProfile:
        """Detect total and available RAM in megabytes."""
        try:
            vm = psutil.virtual_memory()
            total_ram_mb = int(vm.total / (1024 * 1024))
            available_ram_mb = int(vm.available / (1024 * 1024))
        except Exception as e:
            raise HardwareDetectionError(f"Failed to detect system RAM: {e}") from e

        if total_ram_mb < 1:
            raise HardwareDetectionError(
                f"Detected total system RAM ({total_ram_mb} MB) is invalid."
            )

        if available_ram_mb < 0:
            raise HardwareDetectionError(
                f"Detected available RAM ({available_ram_mb} MB) is negative."
            )

        if available_ram_mb > total_ram_mb:
            raise HardwareDetectionError(
                f"Detected available RAM ({available_ram_mb} MB) exceeds "
                f"total system RAM ({total_ram_mb} MB)."
            )

        return MemoryProfile(
            total_ram_mb=total_ram_mb,
            available_ram_mb=available_ram_mb,
        )

    def _detect_runtime(self) -> RuntimeProfile:
        """Detect executing Python runtime facts."""
        py_version = platform.python_version().strip()
        py_impl = platform.python_implementation().strip()
        mach_arch = platform.machine().strip()

        if not py_version or not py_impl or not mach_arch:
            raise HardwareDetectionError(
                "Unable to determine executing Python runtime profile."
            )

        return RuntimeProfile(
            python_version=py_version,
            python_implementation=py_impl,
            machine_architecture=mach_arch,
        )

    # ── private GPU detection helpers (best-effort) ─────────────────────

    def _detect_gpus(self) -> List[GPUDevice]:
        """Optionally detect GPU devices on the local system."""
        gpus: List[GPUDevice] = []
        try:
            # Structuring for future multi-vendor support:
            gpus.extend(self._detect_nvidia_gpus())
            # future extensions:
            # gpus.extend(self._detect_amd_gpus())
            # gpus.extend(self._detect_intel_gpus())
        except Exception as e:
            logger.warning("GPU detection encountered an unexpected error: %s", e)
        return gpus

    def _detect_nvidia_gpus(self) -> List[GPUDevice]:
        """Detect NVIDIA GPUs using nvidia-smi."""
        nvidia_smi_path = shutil.which("nvidia-smi")
        if not nvidia_smi_path:
            logger.info("nvidia-smi not found. Skipping NVIDIA GPU detection.")
            return []

        # 1. Best-effort CUDA version query.
        cuda_available, cuda_version = self._query_cuda_info(nvidia_smi_path)

        # 2. Query GPU details in structured CSV format.
        try:
            result = subprocess.run(
                [
                    nvidia_smi_path,
                    "--query-gpu=name,memory.total,memory.free",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=10,
                check=True,
            )
        except (subprocess.SubprocessError, FileNotFoundError, OSError) as e:
            logger.warning("nvidia-smi execution failed: %s", e)
            return []

        devices: List[GPUDevice] = []
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line:
                continue

            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 3:
                logger.warning("Skipping malformed nvidia-smi CSV row: %r", line)
                continue

            name, total_str, free_str = parts

            # Parse memory values. If parsing fails, skip the row.
            try:
                memory_total_mb = int(total_str)
                memory_available_mb = int(free_str)
            except ValueError as e:
                logger.warning("Skipping GPU due to invalid memory formatting: %s", e)
                continue

            try:
                device = GPUDevice(
                    name=name,
                    vendor=GPUVendor.NVIDIA,
                    memory_total_mb=memory_total_mb,
                    memory_available_mb=memory_available_mb,
                    cuda_available=cuda_available,
                    cuda_version=cuda_version,
                )
                devices.append(device)
            except ValidationError as ve:
                logger.warning("Skipping GPU device due to validation failure: %s", ve)

        return devices

    def _query_cuda_info(self, nvidia_smi_path: str) -> tuple[bool, Optional[str]]:
        """Retrieve driver-reported CUDA availability and version via nvidia-smi."""
        try:
            result = subprocess.run(
                [nvidia_smi_path],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except Exception:
            return False, None

        if result.returncode != 0:
            return False, None

        # Search for "CUDA Version: X.Y" in the standard output.
        match = re.search(r"CUDA\s+Version:\s*(\d+\.\d+)", result.stdout, re.IGNORECASE)
        if match:
            cuda_version = match.group(1)
            return True, cuda_version

        return False, None
