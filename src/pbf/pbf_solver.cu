#include "pbf/pbf_solver.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>

constexpr int kBlockSize = 256;
constexpr int kMaxNeighbors = 256;

void checkCuda(cudaError_t error) {
    if (error != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(error));
}

void checkKernelLaunch() {
    checkCuda(cudaGetLastError());
}

void PBFSolver::initialize(size_t maxParticles, float3 minBounds, float3 maxBounds,
                           const SimulationParams& params) {
    if (maxParticles == 0)
        throw std::invalid_argument("maxParticles must be greater than zero");

    if (maxParticles > std::numeric_limits<size_t>::max() /
                           static_cast<size_t>(kMaxNeighbors)) {
        throw std::length_error("Neighbor buffer size overflows size_t");
    }

    if (maxParticles > static_cast<size_t>(std::numeric_limits<int>::max()))
        throw std::length_error("Particle count exceeds CUDA kernel index range");

    if (!std::isfinite(params.dt) || params.dt <= 0.0f ||
        !std::isfinite(params.restDensity) || params.restDensity <= 0.0f ||
        !std::isfinite(params.particleMass) || params.particleMass <= 0.0f ||
        !std::isfinite(params.smoothingRadius) || params.smoothingRadius <= 0.0f ||
        !std::isfinite(params.lambdaEpsilon) || params.lambdaEpsilon < 0.0f ||
        !std::isfinite(params.gravity.x) || !std::isfinite(params.gravity.y) ||
        !std::isfinite(params.gravity.z) ||
        params.solverIterations <= 0 || params.substeps <= 0) {
        throw std::invalid_argument("Invalid PBF simulation parameters");
    }

    spatialGrid_.initialize(maxParticles, minBounds, maxBounds,
                            params.smoothingRadius);

    positions_.allocate(maxParticles);
    predictedPositions_.allocate(maxParticles);
    velocities_.allocate(maxParticles);
    neighbors_.allocate(maxParticles * static_cast<size_t>(kMaxNeighbors));
    neighborsCount_.allocate(maxParticles);
    density_.allocate(maxParticles);
    constraints_.allocate(maxParticles);
    lambda_.allocate(maxParticles);
    deltaPosition_.allocate(maxParticles);

    maxParticles_ = maxParticles;
    particleCount_ = 0;
    params_ = params;
}

void PBFSolver::setParticles(const float4* positions, const float4* velocities,
                             size_t particleCount) {
    if (maxParticles_ == 0)
        throw std::logic_error("PBFSolver must be initialized before setting particles");

    if (particleCount > maxParticles_)
        throw std::out_of_range("Particle count exceeds solver capacity");

    if (particleCount == 0) {
        particleCount_ = 0;
        return;
    }

    positions_.copyFromHostToDevice(positions, particleCount);
    velocities_.copyFromHostToDevice(velocities, particleCount);
    particleCount_ = particleCount;
}

void PBFSolver::step() {
    if (maxParticles_ == 0)
        throw std::logic_error("PBFSolver must be initialized before stepping");

    if (particleCount_ == 0)
        return;

    const float substepDt = params_.dt / static_cast<float>(params_.substeps);
    const int gridSize = static_cast<int>(
        (particleCount_ + kBlockSize - 1) / kBlockSize
    );

    for (int substep = 0; substep < params_.substeps; ++substep) {
        integrate<<<gridSize, kBlockSize>>>(
            positions_.data(), predictedPositions_.data(), velocities_.data(),
            static_cast<int>(particleCount_), substepDt, params_.gravity
        );
        checkKernelLaunch();

        for (int iteration = 0; iteration < params_.solverIterations; ++iteration) {
            spatialGrid_.build(predictedPositions_.data(), particleCount_);

            findNeighbors<<<gridSize, kBlockSize>>>(
                predictedPositions_.data(), spatialGrid_.sortedIndices(),
                spatialGrid_.cellStart(), spatialGrid_.cellEnd(),
                spatialGrid_.gridSize(), spatialGrid_.minBounds(),
                spatialGrid_.cellSize(), particleCount_, params_.smoothingRadius,
                neighbors_.data(), neighborsCount_.data(), kMaxNeighbors
            );
            checkKernelLaunch();

            computeDensity<<<gridSize, kBlockSize>>>(
                predictedPositions_.data(), neighbors_.data(), neighborsCount_.data(),
                kMaxNeighbors, particleCount_, params_.smoothingRadius,
                params_.particleMass, density_.data(), constraints_.data(),
                params_.restDensity
            );
            checkKernelLaunch();

            computeLambda<<<gridSize, kBlockSize>>>(
                predictedPositions_.data(), neighbors_.data(), neighborsCount_.data(),
                kMaxNeighbors, constraints_.data(), particleCount_,
                params_.smoothingRadius, params_.particleMass, params_.restDensity,
                params_.lambdaEpsilon, lambda_.data()
            );
            checkKernelLaunch();

            computeDeltaPosition<<<gridSize, kBlockSize>>>(
                predictedPositions_.data(), neighbors_.data(), neighborsCount_.data(),
                kMaxNeighbors, lambda_.data(), particleCount_,
                params_.smoothingRadius, params_.particleMass, params_.restDensity,
                deltaPosition_.data()
            );
            checkKernelLaunch();

            applyDeltaPosition<<<gridSize, kBlockSize>>>(
                predictedPositions_.data(), deltaPosition_.data(), particleCount_
            );
            checkKernelLaunch();
        }

        updateVelocityAndPosition<<<gridSize, kBlockSize>>>(
            positions_.data(), predictedPositions_.data(), velocities_.data(),
            particleCount_, 1.0f / substepDt
        );
        checkKernelLaunch();
    }

    checkCuda(cudaDeviceSynchronize());
}

void PBFSolver::run() {
    if (maxParticles_ == 0)
        throw std::logic_error("PBFSolver must be initialized before running");

    for (int iteration = 0; iteration < params_.solverIterations; ++iteration)
        step();
}

void PBFSolver::copyPositionsToHost(float4* positions, size_t particleCount) const {
    if (particleCount != particleCount_)
        throw std::out_of_range("Requested position count does not match active particles");

    if (particleCount != 0)
        positions_.copyFromDeviceToHost(positions, particleCount);
}

void PBFSolver::copyVelocitiesToHost(float4* velocities, size_t particleCount) const {
    if (particleCount != particleCount_)
        throw std::out_of_range("Requested velocity count does not match active particles");

    if (particleCount != 0)
        velocities_.copyFromDeviceToHost(velocities, particleCount);
}
