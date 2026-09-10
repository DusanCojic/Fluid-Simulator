#!/usr/bin/env python3

import csv
import datetime
import io
import os
import pathlib
import platform
import re
import shlex
import shutil
import subprocess
import sys


PARTICLE_COUNT = 262144
FRAMES_PER_REPETITION = 60
REPETITIONS = 5
PROFILED_FRAMES = FRAMES_PER_REPETITION * REPETITIONS
FRAME_BUDGET_MS = 1000.0 / 60.0
PROFILE_RANGE = "PBF 262144 particles"

PROJECT_DIR = pathlib.Path(__file__).resolve().parent.parent
BUILD_DIR = PROJECT_DIR / "build"
BENCHMARK = BUILD_DIR / "pbf_benchmark"
OUTPUT_DIR = PROJECT_DIR / "benchmark" / "profiles"
SYSTEMS_BASE = OUTPUT_DIR / "pbf_systems"
SYSTEMS_REPORT = OUTPUT_DIR / "pbf_systems.nsys-rep"
SYSTEMS_SQLITE = OUTPUT_DIR / "pbf_systems.sqlite"
COMPUTE_BASE = OUTPUT_DIR / "pbf_compute"
COMPUTE_REPORT = OUTPUT_DIR / "pbf_compute.ncu-rep"
REPORT_PATH = OUTPUT_DIR / "PERFORMANCE_REPORT.md"


def run(command, *, capture=False, check=True, output_file=None):
    command = [str(part) for part in command]
    print("+", shlex.join(command), flush=True)
    if output_file is not None:
        with pathlib.Path(output_file).open("w", encoding="utf-8") as output:
            result = subprocess.run(
                command, text=True, stdout=output, stderr=subprocess.STDOUT
            )
    else:
        result = subprocess.run(command, text=True, capture_output=capture)

    if check and result.returncode != 0:
        if capture:
            if result.stdout:
                print(result.stdout, end="", file=sys.stderr)
            if result.stderr:
                print(result.stderr, end="", file=sys.stderr)
        result.check_returncode()
    return result


def require_program(name):
    if shutil.which(name) is None:
        raise RuntimeError(f"Required program is not installed: {name}")


def command_output(command):
    if shutil.which(str(command[0])) is None:
        return "Unavailable"
    result = run(command, capture=True, check=False)
    if result.returncode != 0:
        return "Unavailable"
    return " ".join(result.stdout.strip().split()) or "Unavailable"


def version_output(command):
    if shutil.which(str(command[0])) is None:
        return "Unavailable"
    result = run(command, capture=True, check=False)
    if result.returncode != 0:
        return "Unavailable"
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    preferred = [line for line in lines if "version" in line.lower() or "release" in line.lower()]
    return (preferred[-1] if preferred else (lines[0] if lines else "Unavailable"))


def run_streamed(command, output_file):
    command = [str(part) for part in command]
    print("+", shlex.join(command), flush=True)
    with pathlib.Path(output_file).open("w", encoding="utf-8") as output:
        process = subprocess.Popen(
            command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            output.write(line)
        return process.wait()


def cache_value(name):
    cache = BUILD_DIR / "CMakeCache.txt"
    if not cache.is_file():
        return "Unavailable"
    match = re.search(rf"^{re.escape(name)}:[^=]*=(.*)$", cache.read_text(), re.MULTILINE)
    return match.group(1).strip() if match else "Unavailable"


def collect_metadata(test_status):
    cpu = "Unavailable"
    cpu_info = pathlib.Path("/proc/cpuinfo")
    if cpu_info.is_file():
        match = re.search(r"^model name\s*:\s*(.+)$", cpu_info.read_text(), re.MULTILINE)
        if match:
            cpu = match.group(1).strip()

    git_commit = command_output([
        "git", "-C", PROJECT_DIR, "rev-parse", "--short=12", "HEAD"
    ])
    dirty = run(
        ["git", "-C", PROJECT_DIR, "status", "--porcelain"],
        capture=True,
        check=False,
    )
    dirty_lines = [
        line for line in dirty.stdout.splitlines()
        if "benchmark/profiles/PERFORMANCE_REPORT.md" not in line
    ]
    git_state = "dirty" if dirty_lines else "clean"

    return {
        "generated": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "git": f"{git_commit} ({git_state})",
        "os": platform.platform(),
        "cpu": cpu,
        "gpu": command_output([
            "nvidia-smi", "--query-gpu=name,driver_version,compute_cap",
            "--format=csv,noheader",
        ]),
        "nvcc": version_output(["nvcc", "--version"]),
        "nsys": version_output(["nsys", "--version"]),
        "ncu": version_output(["ncu", "--version"]),
        "cmake": version_output(["cmake", "--version"]),
        "build_type": cache_value("CMAKE_BUILD_TYPE"),
        "cuda_architecture": cache_value("CMAKE_CUDA_ARCHITECTURES"),
        "binary_architecture": ", ".join(binary_architectures()),
        "tests": test_status,
    }


def binary_architectures():
    output = command_output(["cuobjdump", "--list-elf", BENCHMARK])
    return sorted(set(re.findall(r"sm_\d+", output)))


def validate_native_binary():
    result = run([
        "nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader",
    ], capture=True)
    capability = result.stdout.splitlines()[0].strip()
    expected = "sm_" + capability.replace(".", "")
    images = binary_architectures()
    if expected not in images:
        raise RuntimeError(
            f"Benchmark binary is not native for GPU compute capability {capability}; "
            f"expected {expected}, found {', '.join(images) or 'no CUDA images'}"
        )


def parse_benchmark(output):
    def number(label):
        match = re.search(rf"^{re.escape(label)}:\s+([0-9.]+)", output, re.MULTILINE)
        if not match:
            raise RuntimeError(f"Missing benchmark metric: {label}")
        return float(match.group(1))

    run_averages = [
        float(value) for value in re.findall(
            r"^Run \d+ average physics time/frame:\s+([0-9.]+)",
            output,
            re.MULTILINE,
        )
    ]
    if len(run_averages) != REPETITIONS:
        raise RuntimeError(
            f"Expected {REPETITIONS} benchmark repetitions, found {len(run_averages)}"
        )

    frame_time_groups = re.findall(
        r"^Run \d+ frame times \(ms\):(.*)$", output, re.MULTILINE
    )
    frame_times = [
        [numeric(value) for value in group.split()]
        for group in frame_time_groups
    ]
    if len(frame_times) != REPETITIONS or any(
        len(values) != FRAMES_PER_REPETITION for values in frame_times
    ):
        raise RuntimeError(
            f"Expected {REPETITIONS} groups of {FRAMES_PER_REPETITION} frame times"
        )

    return {
        "run_averages": run_averages,
        "frame_times": frame_times,
        "total_ms": number("Median total physics time"),
        "average_ms": number("Median average physics time/frame"),
        "p95_ms": number("Median-run P95 physics time/frame"),
        "minimum_ms": number("Median-run minimum physics time/frame"),
        "maximum_ms": number("Median-run maximum physics time/frame"),
        "run_stddev_ms": number("Run-average standard deviation"),
        "particle_frames": number("Particle-frames/second"),
        "particle_substeps": number("Particle-substeps/second"),
    }


def parse_csv(output, required_header):
    lines = [line for line in output.splitlines() if line.strip()]
    header = next(
        (index for index, line in enumerate(lines) if required_header in line), None
    )
    if header is None:
        raise RuntimeError(f"Profiler output has no '{required_header}' table")
    rows = list(csv.DictReader(lines[header:]))
    if not rows:
        raise RuntimeError(f"Profiler output has an empty '{required_header}' table")
    return rows


def field(row, prefix, default="0"):
    for key, value in row.items():
        if key.strip().lower().startswith(prefix.lower()):
            return value.strip()
    return default


def numeric(value):
    cleaned = value.replace("%", "").replace(",", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


def total_time(row):
    return numeric(field(row, "Total Time"))


def row_count(row):
    for prefix in ("Instances", "Num Calls", "Count", "Calls"):
        value = field(row, prefix, "")
        if value:
            return numeric(value)
    return 0.0


def nsys_stats(report, filename, *, force_export=False):
    rows = []
    for repetition in range(REPETITIONS):
        command = [
            "nsys", "stats", "--quiet", "--timeunit", "milliseconds",
            "--format", "csv", "--filter-nvtx",
            f"{PROFILE_RANGE}/{repetition}", "--report", report,
        ]
        if force_export and repetition == 0:
            command.extend(["--force-export=true", SYSTEMS_REPORT])
        else:
            command.append(SYSTEMS_SQLITE)
        rows.extend(parse_csv(run(command, capture=True).stdout, "Total Time"))

    # Nsight Systems accepts only one --filter-nvtx option per invocation.
    # Aggregate the five independently filtered tables so warm-up and reset
    # transfers remain excluded while every measured repetition is included.
    grouped = {}
    for row in rows:
        name = field(row, "Name", field(row, "Operation", "Unknown"))
        aggregate = grouped.setdefault(name, {"time": 0.0, "count": 0.0})
        aggregate["time"] += total_time(row)
        aggregate["count"] += row_count(row)

    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(["Total Time (ms)", "Instances", "Avg (ms)", "Name"])
    for name, aggregate in sorted(
        grouped.items(), key=lambda item: item[1]["time"], reverse=True
    ):
        count = aggregate["count"]
        average = aggregate["time"] / count if count else 0.0
        writer.writerow([
            f"{aggregate['time']:.9f}", f"{count:.0f}",
            f"{average:.9f}", name,
        ])
    text = output.getvalue()
    (OUTPUT_DIR / filename).write_text(text, encoding="utf-8")
    return text


def collect_systems_profile():
    run([
        "nsys", "profile", "--trace=cuda,nvtx,osrt", "--sample=none",
        "--cpuctxsw=none", "--force-overwrite=true", f"--output={SYSTEMS_BASE}",
        BENCHMARK,
    ])
    kernels = parse_csv(
        nsys_stats("cuda_gpu_kern_sum:base", "pbf_kernels.csv", force_export=True),
        "Total Time",
    )
    memory = parse_csv(
        nsys_stats("cuda_gpu_mem_time_sum", "pbf_memory_operations.csv"),
        "Total Time",
    )
    api = parse_csv(
        nsys_stats("cuda_api_sum", "pbf_cuda_api.csv"),
        "Total Time",
    )
    return {"status": "Complete", "kernels": kernels, "memory": memory, "api": api}


def parse_compute_metrics(output):
    lines = [line for line in output.splitlines() if line.strip()]
    header = next(
        (index for index, line in enumerate(lines) if "Metric Name" in line), None
    )
    if header is None:
        return []

    wanted = {
        "sm__warps_active.avg.pct_of_peak_sustained_active": "occupancy",
        "sm__throughput.avg.pct_of_peak_sustained_elapsed": "sm_throughput",
        "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed":
            "memory_throughput",
        "dram__throughput.avg.pct_of_peak_sustained_elapsed": "dram_throughput",
    }
    grouped = {}
    for row in csv.DictReader(lines[header:]):
        metric = row.get("Metric Name", "").strip()
        if metric not in wanted:
            continue
        kernel = row.get("Kernel Name", "Unknown").strip()
        value = numeric(row.get("Metric Value", "0"))
        grouped.setdefault(kernel, {}).setdefault(wanted[metric], []).append(value)

    results = []
    for kernel, metrics in grouped.items():
        result = {"kernel": kernel}
        for name, values in metrics.items():
            result[name] = sum(values) / len(values)
        results.append(result)
    return results


def collect_compute_profile():
    if os.environ.get("PBF_SKIP_COMPUTE") == "1":
        return {"status": "Skipped by PBF_SKIP_COMPUTE=1", "metrics": []}

    launch_count = os.environ.get("NCU_LAUNCH_COUNT", "200")
    command = [
        "ncu", "--target-processes", "all", "--nvtx",
        "--nvtx-include", PROFILE_RANGE + "/",
        "--metrics", ",".join([
            "sm__warps_active.avg.pct_of_peak_sustained_active",
            "sm__throughput.avg.pct_of_peak_sustained_elapsed",
            "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
            "dram__throughput.avg.pct_of_peak_sustained_elapsed",
        ]),
        "--print-summary", "per-kernel", "--force-overwrite",
        "--export", COMPUTE_BASE,
    ]
    if launch_count != "all":
        command.extend(["--launch-count", launch_count])
    command.append(BENCHMARK)
    print(f"Collecting Compute metrics for {launch_count} matching launches.")
    return_code = run_streamed(command, OUTPUT_DIR / "pbf_compute.txt")
    if return_code != 0:
        log = (OUTPUT_DIR / "pbf_compute.txt").read_text(encoding="utf-8")
        if "ERR_NVGPUCTRPERM" in log:
            status = "Unavailable: GPU performance-counter permission denied"
        else:
            status = f"Failed with exit code {return_code}"
        return {"status": status, "metrics": []}

    raw = run(
        ["ncu", "--import", COMPUTE_REPORT, "--csv", "--page", "raw"],
        capture=True,
    ).stdout
    (OUTPUT_DIR / "pbf_compute_metrics.csv").write_text(raw, encoding="utf-8")
    metrics = parse_compute_metrics(raw)
    status = (
        f"Complete ({launch_count} matching launches)"
        if metrics
        else "Complete, but expected hardware metrics were not found"
    )
    return {"status": status, "metrics": metrics}


def markdown_name(name):
    return "`" + name.replace("|", "\\|") + "`"


def write_report(benchmark, systems, compute, metadata):
    kernels = sorted(systems.get("kernels", []), key=total_time, reverse=True)
    memory = systems.get("memory", [])
    api = systems.get("api", [])
    kernel_total = sum(total_time(row) for row in kernels)
    memory_total = sum(total_time(row) for row in memory)
    gpu_operation_total = kernel_total + memory_total

    blocking_api = []
    for row in api:
        name = field(row, "Name").lower()
        if "synchronize" in name or ("memcpy" in name and "async" not in name):
            blocking_api.append(row)
    blocking_api.sort(key=total_time, reverse=True)
    blocking_total = sum(total_time(row) for row in blocking_api)

    average = benchmark["average_ms"]
    excess = average - FRAME_BUDGET_MS
    verdict = (
        f"Meets the 60 FPS physics budget with {-excess:.3f} ms remaining."
        if excess <= 0
        else f"Does not meet 60 FPS; physics uses {average / FRAME_BUDGET_MS:.2f}× "
             f"the frame budget and exceeds it by {excess:.3f} ms."
    )
    systems_complete = systems["status"] == "Complete"
    source_clean = metadata["git"].endswith("(clean)")
    if systems_complete and source_clean:
        report_status = "Publication-ready"
    elif systems_complete:
        report_status = "Complete, but source tree is dirty; commit before publishing"
    else:
        report_status = "Incomplete"

    lines = [
        "# PBF Performance Report",
        "",
        f"**Report status: {report_status}**",
        "",
        "| Stage | Status |",
        "|---|---|",
        "| Benchmark | Complete |",
        f"| Nsight Systems | {systems['status']} |",
        f"| Nsight Compute (optional) | {compute['status']} |",
        "",
        "## Executive summary",
        "",
        f"- Workload: **{PARTICLE_COUNT:,} particles**, {FRAMES_PER_REPETITION} frames "
        f"per repetition, {REPETITIONS} repetitions.",
        f"- Median average physics time: **{average:.3f} ms/frame**.",
        f"- Median-run P95: **{benchmark['p95_ms']:.3f} ms/frame**.",
        f"- Run-to-run standard deviation: **{benchmark['run_stddev_ms']:.3f} ms**.",
        f"- 60 FPS assessment: **{verdict}**",
        "",
        "## Primary benchmark metrics",
        "",
        "| Metric | Result |",
        "|---|---:|",
        f"| Median total physics time | {benchmark['total_ms']:.3f} ms |",
        f"| Median average physics time/frame | {average:.3f} ms |",
        f"| Median-run P95 physics time/frame | {benchmark['p95_ms']:.3f} ms |",
        f"| Median-run minimum | {benchmark['minimum_ms']:.3f} ms |",
        f"| Median-run maximum | {benchmark['maximum_ms']:.3f} ms |",
        f"| Run-average standard deviation | {benchmark['run_stddev_ms']:.3f} ms |",
        f"| Particle-frames/second | {benchmark['particle_frames']:,.0f} |",
        f"| Particle-substeps/second | {benchmark['particle_substeps']:,.0f} |",
        f"| 60 FPS frame budget | {FRAME_BUDGET_MS:.3f} ms |",
        "",
        "### Repetition stability",
        "",
        "| Repetition | Average physics time/frame |",
        "|---:|---:|",
    ]
    for index, value in enumerate(benchmark["run_averages"], start=1):
        lines.append(f"| {index} | {value:.3f} ms |")

    lines.extend([
        "",
        "The primary values come from an unprofiled run. Every solver step synchronizes "
        "before returning, so wall time includes launches, execution, blocking copies, and "
        "solver synchronization. P95 describes workload variation during this evolving "
        "fall-and-collision scene; it is not a stationary-frame jitter percentile.",
        "",
        "## GPU execution overview",
        "",
    ])
    if systems_complete:
        lines.extend([
            "| Metric | Total across profiled repetitions | Per frame |",
            "|---|---:|---:|",
            f"| GPU kernel time | {kernel_total:.3f} ms | "
            f"{kernel_total / PROFILED_FRAMES:.3f} ms |",
            f"| GPU memory-operation time | {memory_total:.3f} ms | "
            f"{memory_total / PROFILED_FRAMES:.3f} ms |",
            f"| Summed GPU operation time | {gpu_operation_total:.3f} ms | "
            f"{gpu_operation_total / PROFILED_FRAMES:.3f} ms |",
            f"| Blocking CUDA API time | {blocking_total:.3f} ms | "
            f"{blocking_total / PROFILED_FRAMES:.3f} ms |",
            "",
            "Summed GPU operation time adds kernel and GPU memory-operation durations; it "
            "is not a timeline duration if operations overlap. Blocking API time mostly "
            "represents the CPU waiting for GPU dependencies; it overlaps GPU execution "
            "and is not additional time that can simply be added or removed.",
            "",
            f"## All GPU kernel types launched ({len(kernels)} unique)",
            "",
            "Nsight Systems aggregates every launch across all measured repetitions. `Calls` "
            "is the number of launches represented by each row.",
            "",
            "| Kernel | Total | Per call | Calls | Kernel-time share |",
            "|---|---:|---:|---:|---:|",
        ])
        for row in kernels:
            duration = total_time(row)
            calls = int(numeric(field(row, "Instances", field(row, "Count", "0"))))
            share = 100.0 * duration / kernel_total if kernel_total else 0.0
            lines.append(
                f"| {markdown_name(field(row, 'Name'))} | {duration:.3f} ms | "
                f"{numeric(field(row, 'Avg')):.6f} ms | {calls:,} | {share:.1f}% |"
            )

        lines.extend([
            "",
            "## Blocking CUDA API breakdown",
            "",
            "| CUDA API | Total | Calls | Average call |",
            "|---|---:|---:|---:|",
        ])
        for row in blocking_api:
            calls = int(row_count(row))
            lines.append(
                f"| {markdown_name(field(row, 'Name'))} | {total_time(row):.3f} ms | "
                f"{calls:,} | {numeric(field(row, 'Avg')):.6f} ms |"
            )
    else:
        lines.extend([
            f"Nsight Systems data is unavailable: **{systems['status']}**.", ""
        ])

    lines.extend(["", "## Nsight Compute hardware counters", ""])
    if compute["metrics"]:
        lines.extend([
            "These are averages across the representative launches collected by Compute.",
            "",
            "| Kernel | Occupancy | SM throughput | Memory throughput | DRAM throughput |",
            "|---|---:|---:|---:|---:|",
        ])
        for row in compute["metrics"]:
            lines.append(
                f"| {markdown_name(row['kernel'])} | {row.get('occupancy', 0):.1f}% | "
                f"{row.get('sm_throughput', 0):.1f}% | "
                f"{row.get('memory_throughput', 0):.1f}% | "
                f"{row.get('dram_throughput', 0):.1f}% |"
            )
    else:
        lines.extend([
            f"No hardware-counter table is available: **{compute['status']}**.",
            "The timing report remains complete because Compute is an optional diagnostic; "
            "enable NVIDIA performance-counter access and rerun to add this appendix.",
        ])

    lines.extend([
        "",
        "## Reproducibility metadata",
        "",
        "| Item | Value |",
        "|---|---|",
    ])
    for label, key in [
        ("Generated", "generated"), ("Git revision", "git"),
        ("Operating system", "os"), ("CPU", "cpu"), ("GPU / driver / CC", "gpu"),
        ("CUDA compiler", "nvcc"), ("Nsight Systems", "nsys"),
        ("Nsight Compute", "ncu"), ("CMake", "cmake"),
        ("Build type", "build_type"), ("CUDA architecture", "cuda_architecture"),
        ("Binary CUDA images", "binary_architecture"),
        ("Test suite", "tests"),
    ]:
        value = metadata[key].replace("|", "\\|")
        lines.append(f"| {label} | {value} |")

    lines.extend([
        "",
        "## Methodology and scene",
        "",
        "The script configures a native-architecture Release build, runs the correctness "
        "tests, then measures five deterministic repetitions. Ten warm-up frames initialize "
        "CUDA/CUB and raise GPU clocks; particle state is restored before every repetition.",
        "",
        "The scene is an 8.2 × 4.0 × 4.2 m open-top container. A 262,144-particle "
        "water block falls under gravity onto one static AABB and one static sphere. Particle "
        "spacing is 0.06 m, radius is 0.025 m, smoothing radius is 0.12 m, and every frame "
        "uses two substeps with four PBF iterations per substep.",
        "",
        "Nsight Systems profiles all measured repetitions. Nsight Compute samples the first "
        f"{os.environ.get('NCU_LAUNCH_COUNT', '200')} matching launches for occupancy and "
        "throughput counters; profiler timings are diagnostic and never replace the "
        "unprofiled baseline.",
        "",
        "## Artifacts",
        "",
        "Raw outputs are in `benchmark/profiles/`: benchmark output, correctness-test output, "
        "Systems report and CSV tables, and optional Compute report and metrics.",
        "",
    ])
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main():
    required = ["cmake", "ctest", "git", "nvcc", "nvidia-smi", "cuobjdump", "nsys"]
    if os.environ.get("PBF_SKIP_COMPUTE") != "1":
        required.append("ncu")
    for program in required:
        require_program(program)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n=== Native Release build ===")
    run([
        "cmake", "-S", PROJECT_DIR, "-B", BUILD_DIR,
        "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_CUDA_ARCHITECTURES=native",
    ])
    run(["cmake", "--build", BUILD_DIR, "-j2"])
    validate_native_binary()

    print("\n=== Correctness tests ===")
    tests = run(
        ["ctest", "--test-dir", BUILD_DIR, "--output-on-failure"],
        check=False,
        output_file=OUTPUT_DIR / "test_output.txt",
    )
    test_status = "Passed" if tests.returncode == 0 else f"Failed ({tests.returncode})"
    if tests.returncode != 0:
        raise RuntimeError("Correctness tests failed; see benchmark/profiles/test_output.txt")

    metadata = collect_metadata(test_status)
    systems = {"status": "Pending", "kernels": [], "memory": [], "api": []}
    compute = {"status": "Pending", "metrics": []}

    print("\n=== Unprofiled benchmark ===")
    baseline = run([BENCHMARK], capture=True)
    print(baseline.stdout, end="")
    (OUTPUT_DIR / "benchmark_output.txt").write_text(baseline.stdout, encoding="utf-8")
    benchmark = parse_benchmark(baseline.stdout)
    write_report(benchmark, systems, compute, metadata)
    print(f"Initial report: {REPORT_PATH}")

    print("\n=== Nsight Systems ===")
    try:
        systems = collect_systems_profile()
    except (RuntimeError, subprocess.CalledProcessError) as error:
        systems = {"status": f"Failed: {error}", "kernels": [], "memory": [], "api": []}
    write_report(benchmark, systems, compute, metadata)
    print(f"Systems results written: {REPORT_PATH}")

    print("\n=== Nsight Compute (optional) ===")
    compute = collect_compute_profile()
    write_report(benchmark, systems, compute, metadata)
    print(f"Final report: {REPORT_PATH}")
    if systems["status"] != "Complete":
        raise RuntimeError(
            "Nsight Systems profiling is incomplete; see the generated report"
        )


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, subprocess.CalledProcessError) as error:
        print(f"Profiling failed: {error}", file=sys.stderr)
        sys.exit(1)
