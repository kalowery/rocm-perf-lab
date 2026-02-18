import pytest
from rocm_perf_lab.schema.profile import ProfileModel


def test_valid_profile_schema():
    profile = {
        "schema_version": "1.0",
        "kernel": {
            "name": "test_kernel",
            "grid": (1, 1, 1),
            "block": (1, 1, 1),
        },
        "gpu": {
            "architecture": "rdna2",
            "wave_size": 32,
            "compute_units": 16,
        },
        "runtime_ms": 1.0,
        "stability": {
            "runs": 3,
            "mean_ms": 1.0,
            "stddev_ms": 0.1,
            "cv": 0.1,
            "classification": "moderate",
        },
        "resources": {
            "vgpr_per_thread": 8,
            "sgpr_per_wave": 128,
            "lds_bytes": 0,
        },
        "occupancy": {
            "theoretical": 1.0,
            "threads_per_block": 256,
            "wave_size": 32,
        },
    }

    validated = ProfileModel(**profile)
    assert validated.schema_version == "1.0"


def test_invalid_profile_schema_missing_field():
    invalid_profile = {
        "schema_version": "1.0"
    }

    with pytest.raises(Exception):
        ProfileModel(**invalid_profile)
