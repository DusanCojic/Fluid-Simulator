#include "pbf/pbf_solver.hpp"

#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr std::size_t kParticleCount = 262144;

constexpr std::size_t kParticlesX = 128;
constexpr std::size_t kParticlesZ = 64;
constexpr float kParticleSpacing = 0.06f;
constexpr float kParticleRadius = 0.025f;
constexpr float kWaterTop = 3.75f;

constexpr float3 kContainerMin = {0.0f, 0.0f, 0.0f};
constexpr float3 kContainerMax = {8.2f, 4.0f, 4.2f};
// A one-diameter floor clearance keeps the particle-expanded obstacles from
// overlapping the container's particle-center floor constraint.
constexpr BoxCollider kBox = {
    {2.4f, 0.675f, 2.1f},
    {0.65f, 0.625f, 0.75f}
};
constexpr SphereCollider kSphere = {
    {5.7f, 0.8f, 2.1f},
    0.75f
};

constexpr std::size_t kWarmupFrames = 10;
constexpr std::size_t kBenchmarkFrames = 60;
constexpr std::size_t kRepetitions = 5;
constexpr char kProfileRange[] = "PBF 262144 particles";

struct RunResults {
    std::vector<double> frameTimes;
    double totalMilliseconds;
    double averageMilliseconds;
    double p95Milliseconds;
    double minimumMilliseconds;
    double maximumMilliseconds;
};

struct BenchmarkResults {
    std::vector<RunResults> runs;
    RunResults medianRun;
    double runAverageStdDev;
    double particleFramesPerSecond;
    double particleSubstepsPerSecond;
};

SimulationParams simulationParams() {
    SimulationParams params{};
    params.dt = 1.0f / 60.0f;
    params.restDensity = 1000.0f;
    params.particleMass =
        params.restDensity * kParticleSpacing * kParticleSpacing * kParticleSpacing;
    params.particleRadius = kParticleRadius;
    params.collisionRestitution = 0.0f;
    params.collisionFriction = 0.03f;
    params.smoothingRadius = 0.12f;
    params.lambdaEpsilon = 100.0f;
    params.solverIterations = 4;
    params.substeps = 2;
    params.gravity = {0.0f, -9.81f, 0.0f};
    params.scorrK = 0.0001f;
    params.scorrN = 4;
    params.scorrDeltaQ = 0.036f;
    params.xsphViscosity = 0.00001f;
    params.vorticityStrength = 0.00001f;
    return params;
}

std::vector<float4> createParticles(std::size_t particleCount) {
    const std::size_t particlesPerLayer = kParticlesX * kParticlesZ;
    if (particleCount % particlesPerLayer != 0)
        throw std::invalid_argument("Particle count must contain whole water layers");

    const std::size_t layerCount = particleCount / particlesPerLayer;
    const float waterWidth = static_cast<float>(kParticlesX - 1) * kParticleSpacing;
    const float waterDepth = static_cast<float>(kParticlesZ - 1) * kParticleSpacing;
    const float startX = (kContainerMax.x - waterWidth) * 0.5f;
    const float startZ = (kContainerMax.z - waterDepth) * 0.5f;

    std::vector<float4> positions;
    positions.reserve(particleCount);

    // Fill downward from a fixed top surface in deterministic grid order.
    for (std::size_t y = 0; y < layerCount; ++y) {
        for (std::size_t z = 0; z < kParticlesZ; ++z) {
            for (std::size_t x = 0; x < kParticlesX; ++x) {
                positions.push_back({
                    startX + static_cast<float>(x) * kParticleSpacing,
                    kWaterTop - static_cast<float>(y) * kParticleSpacing,
                    startZ + static_cast<float>(z) * kParticleSpacing,
                    1.0f
                });
            }
        }
    }

    return positions;
}

void checkCuda(cudaError_t error) {
    if (error != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(error));
}

RunResults measureRun(PBFSolver& solver) {
    std::vector<double> frameTimes;
    frameTimes.reserve(kBenchmarkFrames);

    nvtxRangePushA(kProfileRange);
    for (std::size_t frame = 0; frame < kBenchmarkFrames; ++frame) {
        nvtxRangePushA("Physics frame");
        const auto start = std::chrono::steady_clock::now();
        solver.step();
        const auto stop = std::chrono::steady_clock::now();
        nvtxRangePop();

        frameTimes.push_back(
            std::chrono::duration<double, std::milli>(stop - start).count()
        );
    }
    nvtxRangePop();

    const double totalMilliseconds =
        std::accumulate(frameTimes.begin(), frameTimes.end(), 0.0);

    std::vector<double> sortedFrameTimes = frameTimes;
    std::sort(sortedFrameTimes.begin(), sortedFrameTimes.end());
    const std::size_t p95Index = static_cast<std::size_t>(
        std::ceil(0.95 * static_cast<double>(sortedFrameTimes.size()))
    ) - 1;

    return RunResults{
        std::move(frameTimes),
        totalMilliseconds,
        totalMilliseconds / static_cast<double>(kBenchmarkFrames),
        sortedFrameTimes[p95Index],
        sortedFrameTimes.front(),
        sortedFrameTimes.back()
    };
}

BenchmarkResults runBenchmark() {
    const SimulationParams params = simulationParams();
    const std::vector<float4> positions = createParticles(kParticleCount);
    const std::vector<float4> velocities(kParticleCount, {0.0f, 0.0f, 0.0f, 0.0f});

    PBFSolver solver;
    solver.initialize(kParticleCount, kContainerMin, kContainerMax, params);
    solver.setBoxes({kBox});
    solver.setSpheres({kSphere});
    solver.setParticles(positions.data(), velocities.data(), kParticleCount);

    // Warm CUDA, CUB, and GPU clocks, then restore the deterministic state.
    solver.run(kWarmupFrames);

    std::vector<RunResults> runs;
    runs.reserve(kRepetitions);
    for (std::size_t repetition = 0; repetition < kRepetitions; ++repetition) {
        solver.setParticles(positions.data(), velocities.data(), kParticleCount);
        checkCuda(cudaDeviceSynchronize());
        runs.push_back(measureRun(solver));
    }

    std::vector<std::size_t> order(kRepetitions);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&runs](std::size_t left, std::size_t right) {
        return runs[left].averageMilliseconds < runs[right].averageMilliseconds;
    });
    const RunResults medianRun = runs[order[kRepetitions / 2]];

    const double meanAverage = std::accumulate(
        runs.begin(), runs.end(), 0.0,
        [](double sum, const RunResults& run) { return sum + run.averageMilliseconds; }
    ) / static_cast<double>(kRepetitions);
    const double squaredDifferenceSum = std::accumulate(
        runs.begin(), runs.end(), 0.0,
        [meanAverage](double sum, const RunResults& run) {
            const double difference = run.averageMilliseconds - meanAverage;
            return sum + difference * difference;
        }
    );
    const double runAverageStdDev = std::sqrt(
        squaredDifferenceSum / static_cast<double>(kRepetitions)
    );
    const double measuredSeconds = medianRun.totalMilliseconds / 1000.0;
    const double particleFrames =
        static_cast<double>(kParticleCount * kBenchmarkFrames);

    return {
        std::move(runs),
        medianRun,
        runAverageStdDev,
        particleFrames / measuredSeconds,
        particleFrames * static_cast<double>(params.substeps) / measuredSeconds
    };
}

} // namespace

int main() {
    try {
        std::cout << "PBF Benchmark\n\n";
        const BenchmarkResults results = runBenchmark();
        std::cout << std::fixed << std::setprecision(3)
                  << "Particles: " << kParticleCount << '\n'
                  << "Frames per repetition: " << kBenchmarkFrames << '\n'
                  << "Warm-up frames: " << kWarmupFrames << '\n'
                  << "Repetitions: " << kRepetitions << '\n';
        for (std::size_t index = 0; index < results.runs.size(); ++index) {
            std::cout << "Run " << index + 1 << " average physics time/frame: "
                      << results.runs[index].averageMilliseconds << " ms\n"
                      << "Run " << index + 1 << " frame times (ms):";
            for (const double frameTime : results.runs[index].frameTimes)
                std::cout << ' ' << frameTime;
            std::cout << '\n';
        }
        std::cout << "Median total physics time: "
                  << results.medianRun.totalMilliseconds << " ms\n"
                  << "Median average physics time/frame: "
                  << results.medianRun.averageMilliseconds << " ms\n"
                  << "Median-run P95 physics time/frame: "
                  << results.medianRun.p95Milliseconds << " ms\n"
                  << "Median-run minimum physics time/frame: "
                  << results.medianRun.minimumMilliseconds << " ms\n"
                  << "Median-run maximum physics time/frame: "
                  << results.medianRun.maximumMilliseconds << " ms\n"
                  << "Run-average standard deviation: "
                  << results.runAverageStdDev << " ms\n"
                  << "Particle-frames/second: "
                  << results.particleFramesPerSecond << "\n"
                  << "Particle-substeps/second: "
                  << results.particleSubstepsPerSecond << "\n";
    } catch (const std::exception& error) {
        std::cerr << "Benchmark failed: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
