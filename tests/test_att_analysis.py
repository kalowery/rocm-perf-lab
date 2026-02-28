import json
from pathlib import Path

from rocm_perf_lab.analysis.att_analysis import analyze_att


def _write_code_json(tmp_path: Path, entries: list):
    payload = {"code": entries}
    code_path = tmp_path / "code.json"
    code_path.write_text(json.dumps(payload))
    return tmp_path


def test_analyze_att_computes_mix_and_latency(tmp_path):
    entries = [
        ["v_add_f32", 0, 0, 0, 0, 0, 100, 10, 5, 5],
        ["s_or_b32", 0, 0, 0, 0, 0, 50, 2, 0, 0],
        ["global_load", 0, 0, 0, 0, 0, 25, 40, 10, 0],
    ]

    dispatch_dir = _write_code_json(tmp_path, entries)
    result = analyze_att(dispatch_dir)

    assert abs(result.instruction_mix.get("VALU", 0.0) - 100 / 175) < 1e-6
    assert abs(result.instruction_mix.get("SALU", 0.0) - 50 / 175) < 1e-6
    assert abs(result.instruction_mix.get("VMEM", 0.0) - 25 / 175) < 1e-6

    assert abs(result.stall_fraction - 15 / 70) < 1e-6
    assert abs(result.idle_fraction - 5 / 70) < 1e-6
    assert abs(result.ipc - 175 / 70) < 1e-6
    assert result.avg_memory_latency == 40.0


def test_analyze_att_handles_empty_logs(tmp_path):
    dispatch_dir = _write_code_json(tmp_path, [])
    result = analyze_att(dispatch_dir)

    assert result.instruction_mix == {}
    assert result.stall_fraction == 0.0
    assert result.idle_fraction == 0.0
    assert result.avg_memory_latency == 0.0
