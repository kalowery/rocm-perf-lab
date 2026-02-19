from pathlib import Path
from typing import Optional

from rocm_perf_lab.profiler.pipeline import build_profile
from rocm_perf_lab.analysis.critical_path import analyze_critical_path
from rocm_perf_lab.analysis.att_analysis import analyze_att
from rocm_perf_lab.analysis.bottleneck_classifier import classify_bottleneck



def build_extended_profile(
    base_profile: dict,
    rocpd_db_path: Optional[Path] = None,
    att_dispatch_dir: Optional[Path] = None,
):
    """
    Augment an already-built base_profile with:
      - Critical path analysis (if rocpd_db_path provided)
      - ATT deep analysis (if att_dispatch_dir provided)
      - Bottleneck classification
      - Headroom estimation
    """

    extended = dict(base_profile)

    critical_result = None
    att_result = None
    bottleneck = None
    headroom_fraction = None

    # ----------------------------
    # Critical Path
    # ----------------------------
    if rocpd_db_path is not None and rocpd_db_path.exists():
        critical_result = analyze_critical_path(str(rocpd_db_path))

        extended["critical_path"] = {
            "critical_path_ns": critical_result.critical_path_ns,
            "dominant_symbol": critical_result.dominant_symbol_name,
            "fraction": critical_result.dominant_symbol_fraction,
        }

    # ----------------------------
    # ATT Deep Analysis
    # ----------------------------
    if att_dispatch_dir is not None and att_dispatch_dir.exists():
        try:
            att_result = analyze_att(att_dispatch_dir)

            extended["att"] = {
                "instruction_mix": att_result.instruction_mix,
                "stall_fraction": att_result.stall_fraction,
                "idle_fraction": att_result.idle_fraction,
                "avg_memory_latency": att_result.avg_memory_latency,
                "ipc": att_result.ipc,
            }
        except Exception as e:
            print(f"[ATT WARNING] ATT analysis failed: {e}")
            extended["att"] = {}

    # ----------------------------
    # Bottleneck + Headroom
    # ----------------------------
    if att_result is not None:
        roofline_bound = None
        if base_profile.get("roofline"):
            roofline_bound = base_profile["roofline"].get("bound")

        bottleneck = classify_bottleneck(
            stall_fraction=att_result.stall_fraction,
            instruction_mix=att_result.instruction_mix,
            roofline_bound=roofline_bound,
            avg_memory_latency=att_result.avg_memory_latency,
        )

        # Conservative headroom estimate: proportion of stall cycles
        headroom_fraction = att_result.stall_fraction * 0.8

        extended["bottleneck"] = {
            "primary": bottleneck.primary,
            "confidence": bottleneck.confidence,
            "reasoning": bottleneck.reasoning,
        }

        extended["headroom_fraction"] = headroom_fraction

    return extended
