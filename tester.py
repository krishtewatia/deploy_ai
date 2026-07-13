import json
from pprint import pprint

from backend.app.hardware import HardwareDetector


def main():
    print("=" * 80)
    print("STAGE 5B — REAL HARDWARE DETECTION TEST")
    print("=" * 80)

    detector = HardwareDetector()
    profile = detector.detect()

    print("\n1. SERIALIZED JSON OUTPUT")
    print("=" * 80)
    print(profile.model_dump_json(indent=2))

    print("\n" + "=" * 80)
    print("2. VERIFY CORE VALUES")
    print("=" * 80)
    pprint(profile.model_dump())

    assert profile.profile_id.startswith("hardware_")
    assert profile.operating_system
    assert profile.os_version
    assert profile.cpu.processor_name
    assert profile.cpu.architecture
    assert profile.cpu.logical_cores >= 1
    assert profile.memory.total_ram_mb >= 1
    assert profile.memory.available_ram_mb >= 0
    assert profile.memory.available_ram_mb <= profile.memory.total_ram_mb
    assert profile.runtime.python_version
    assert profile.runtime.python_implementation
    assert profile.runtime.machine_architecture
    assert isinstance(profile.gpus, list)

    print("\n" + "=" * 80)
    print("ALL REAL HARDWARE CHECKS PASSED SUCCESSFULLY! [OK]")
    print("=" * 80)


if __name__ == "__main__":
    main()