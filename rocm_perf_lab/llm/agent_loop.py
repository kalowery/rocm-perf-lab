import shutil
import subprocess
from pathlib import Path
from typing import Callable

from rocm_perf_lab.llm.prompt_builder import build_optimization_context, build_llm_prompt
from rocm_perf_lab.llm.patch_extractor import extract_cpp_patch
from rocm_perf_lab.profiler.pipeline import build_profile
from rocm_perf_lab.profiler.extended_pipeline import build_extended_profile
from rocm_perf_lab.profiler.att_runner import run_att


def replace_dominant_kernel(source_text: str, kernel_name: str, new_kernel_code: str) -> str:
    kernel_base = kernel_name.split("(")[0]
    pattern = rf"__global__\s+void\s+{kernel_base}\s*\("
    import re

    match = re.search(pattern, source_text)
    if not match:
        raise RuntimeError("Dominant kernel not found for replacement.")

    brace_start = source_text.find("{", match.end())
    brace_count = 0
    i = brace_start
    while i < len(source_text):
        if source_text[i] == "{":
            brace_count += 1
        elif source_text[i] == "}":
            brace_count -= 1
            if brace_count == 0:
                return source_text[:match.start()] + new_kernel_code + source_text[i+1:]
        i += 1

    raise RuntimeError("Failed to replace kernel body.")


def run_llm_optimization_loop(
    source_path: Path,
    binary_cmd: str,
    llm_callable: Callable[[str], str],
    max_iters: int = 3,
    min_improvement: float = 0.02,
    auto_approve: bool = False,
):
    original_source = source_path.read_text()
    best_source = original_source

    print("=== Baseline Profiling ===")

    base_profile = build_profile(
        cmd=binary_cmd,
        runs=3,
        use_rocprof=True,
        roofline=True,
        persist_rocpd=True,
    )

    # Detect rocpd DB inside .rocpd_profile
    import glob
    profile_dir = Path(".rocpd_profile")
    db_files = glob.glob(str(profile_dir / "**/*_results.db"), recursive=True)
    rocpd_db_path = Path(max(db_files, key=lambda p: Path(p).stat().st_mtime)) if db_files else None

    att_dispatch_dir = run_att(binary_cmd)

    extended = build_extended_profile(
        base_profile=base_profile,
        rocpd_db_path=rocpd_db_path,
        att_dispatch_dir=att_dispatch_dir,
    )

    best_runtime = extended.get("runtime_ms")
    cp = extended.get("critical_path", {})
    dominant_symbol = cp.get("dominant_symbol")
    fraction = cp.get("fraction", 1.0)

    if not dominant_symbol:
        raise RuntimeError("Critical-path analysis failed: dominant_symbol is None.")

    if fraction < 0.999:
        max_whole_app_speedup = 1.0 / (1.0 - fraction)
    else:
        max_whole_app_speedup = float("inf")

    print(f"Baseline runtime: {best_runtime} ms")
    print(f"[INFO] Dominant kernel: {dominant_symbol}")
    print(f"[INFO] Dominant fraction: {fraction:.3f}")
    print(f"[INFO] Theoretical whole-app ceiling: {max_whole_app_speedup:.2f}x")

    previous_patch = None

    for i in range(1, max_iters + 1):
        print(f"\n=== LLM Iteration {i} ===")

        context = build_optimization_context(
            source_path=source_path,
            extended_profile=extended,
            full_source=False,
        )

        if i == 1 or previous_patch is None:
            prompt = build_llm_prompt(context, compact=False)
        else:
            prompt = build_llm_prompt(context, compact=False) + (
                f"\n\n=== Previous Optimization Result ===\n"
                f"Previous runtime: {best_runtime} ms\n"
                f"Refine the previous optimization further."
            )

        max_attempts = 2
        attempt = 0
        compile_error_text = None

        while attempt < max_attempts:
            attempt += 1

            response = llm_callable(prompt)
            new_kernel = extract_cpp_patch(response, dominant_symbol)
            previous_patch = new_kernel

            # Enforce kernel signature stability
            import re
            def extract_signature(kernel_code: str) -> str:
                sig_pattern = r"__global__\s+void\s+\w+\s*\((.*?)\)"
                match = re.search(sig_pattern, kernel_code, re.DOTALL)
                if not match:
                    raise RuntimeError("Could not extract kernel signature.")
                return match.group(1).strip()

            orig_sig_pattern = rf"__global__\s+void\s+{dominant_symbol.split('(')[0]}\s*\((.*?)\)"
            orig_sig_match = re.search(orig_sig_pattern, best_source, re.DOTALL)
            if not orig_sig_match:
                raise RuntimeError("Could not extract original kernel signature.")
            orig_sig = orig_sig_match.group(1).strip()

            new_sig = extract_signature(new_kernel)

            if orig_sig != new_sig:
                prompt = build_llm_prompt(context, compact=False) + (
                    f"\n\nThe kernel signature MUST remain exactly:\n({orig_sig})\n"
                    "Do NOT change the parameter list. Only modify the body."
                )
                if attempt >= max_attempts:
                    raise RuntimeError("Kernel signature change is not allowed.")
                continue

            candidate_source = replace_dominant_kernel(best_source, dominant_symbol, new_kernel)

            try:
                candidate_path = Path(".optimization") / f"llm_iter_{i}" / source_path.name
                candidate_path.parent.mkdir(parents=True, exist_ok=True)
                candidate_path.write_text(candidate_source)

                candidate_binary = candidate_path.parent / "variant_binary"
                subprocess.run(["hipcc", "-O3", str(candidate_path), "-o", str(candidate_binary)], check=True, capture_output=True, text=True)
                break
            except subprocess.CalledProcessError as e:
                compile_error_text = e.stderr
                prompt = build_llm_prompt(context, compact=False) + (
                    "\n\nCompilation failed with the following error:\n"
                    f"{compile_error_text}\n"
                    "Fix the kernel while preserving the original signature."
                )
                if attempt >= max_attempts:
                    raise
                continue

        # after successful compile
        candidate_binary = Path(".optimization") / f"llm_iter_{i}" / "variant_binary"

        iter_dir = Path(".optimization") / f"llm_iter_{i}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        candidate_path = iter_dir / source_path.name
        candidate_path.write_text(candidate_source)

        candidate_binary = iter_dir / "variant_binary"

        subprocess.run(["hipcc", "-O3", str(candidate_path), "-o", str(candidate_binary)], check=True)

        base_profile_new = build_profile(
            cmd=str(candidate_binary),
            runs=3,
            use_rocprof=True,
            roofline=True,
            persist_rocpd=True,
        )

        # Detect rocpd DB inside .rocpd_profile for candidate
        import glob
        profile_dir_new = Path(".rocpd_profile")
        db_files_new = glob.glob(str(profile_dir_new / "**/*_results.db"), recursive=True)
        rocpd_db_path_new = Path(max(db_files_new, key=lambda p: Path(p).stat().st_mtime)) if db_files_new else None

        extended_new = build_extended_profile(
            base_profile=base_profile_new,
            rocpd_db_path=rocpd_db_path_new,
            att_dispatch_dir=run_att(str(candidate_binary)),
        )

        new_runtime = extended_new.get("runtime_ms")
        improvement = (best_runtime - new_runtime) / best_runtime

        new_cp = extended_new.get("critical_path", {})
        new_dominant = new_cp.get("dominant_symbol")
        new_fraction = new_cp.get("fraction", 1.0)

        if new_dominant and new_dominant != dominant_symbol:
            print("[INFO] Dominance shifted:")
            print(f"       {dominant_symbol} → {new_dominant}")
            dominant_symbol = new_dominant
            fraction = new_fraction

        print(f"New runtime: {new_runtime} ms")
        print(f"Improvement: {improvement * 100:.2f}%")

        # Regression detection
        from rocm_perf_lab.analysis.optimization_score import detect_regression

        ok_regression, regression_reasons = detect_regression(extended, extended_new)
        if not ok_regression:
            print(f"Rejected due to regression signals: {regression_reasons}")
            break

        if improvement >= min_improvement:
            print("Improvement accepted.")
            best_runtime = new_runtime
            best_source = candidate_source
            extended = extended_new

            if not auto_approve:
                resp = input("Continue optimizing? [y/n]: ").strip().lower()
                if resp != "y":
                    break
        else:
            print("Improvement below threshold. Stopping.")
            break

    print("\n=== Optimization Complete ===")
    print(f"Best runtime: {best_runtime} ms")

    final_path = source_path.parent / (source_path.stem + "_llm_opt.cu")
    final_path.write_text(best_source)
    print(f"Best source written to {final_path}")
