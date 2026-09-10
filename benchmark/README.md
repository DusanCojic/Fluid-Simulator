# PBF benchmark and profiling

This workflow produces a reproducible performance report for the current PBF solver at
262,144 particles. It measures physics only; rendering is not included.

## Run

From the project root:

```sh
./benchmark/profile.py
```

The script performs the complete workflow:

1. Configures and builds Release CUDA code for the native GPU architecture.
2. Verifies that the benchmark binary contains code for the installed GPU.
3. Runs the correctness test suite and stops if a test fails.
4. Runs an unprofiled baseline: 10 warm-up frames followed by five deterministic
   60-frame repetitions.
5. Profiles all five measured repetitions with Nsight Systems.
6. Attempts a bounded Nsight Compute pass for occupancy and throughput counters.
7. Writes `benchmark/profiles/PERFORMANCE_REPORT.md` after every completed stage.

The Markdown report is the publishable artifact. Raw profiler reports, CSV files, and
logs remain in the same directory for audit and are ignored by Git.

## Requirements

- NVIDIA GPU and driver
- CUDA toolkit (`nvcc` and `cuobjdump`)
- CMake and CTest
- Nsight Systems CLI (`nsys`)
- Nsight Compute CLI (`ncu`), unless skipped

Nsight Compute hardware counters may be disabled by the host or driver policy. This does
not invalidate the unprofiled benchmark or Nsight Systems timing results; the report marks
the optional counter section unavailable. To intentionally omit that pass:

```sh
PBF_SKIP_COMPUTE=1 ./benchmark/profile.py
```

By default, Compute profiles the first 200 kernel launches inside the benchmark NVTX
ranges. This bounded sample keeps counter collection practical while Systems still lists
every unique kernel type and all launches. Override it when needed:

```sh
NCU_LAUNCH_COUNT=400 ./benchmark/profile.py
NCU_LAUNCH_COUNT=all ./benchmark/profile.py
```

Profiling every launch can take a long time because hardware-counter collection may replay
kernels. Use the unprofiled median time per frame—not profiler wall time—for FPS claims.

## Workload

The deterministic scene contains an open-top 8.2 x 4.0 x 4.2 m container, one static box,
one static sphere, and a regular block of water that falls under gravity and interacts with
both obstacles. Each frame uses two substeps and four PBF constraint iterations per
substep. Every measured repetition starts from identical particle positions and velocities.

The report records the Git revision and cleanliness, hardware, tool versions, build type,
compiled CUDA architecture, test result, run-to-run variability, every Systems kernel row,
CUDA memory operations, blocking API calls, and optional Compute counters.
