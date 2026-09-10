#include "pbf/pbf_solver.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>

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

    constexpr size_t neighborStrideAlignment = kBlockSize;
    const size_t neighborParticleStride =
        ((maxParticles + neighborStrideAlignment - 1) / neighborStrideAlignment) *
        neighborStrideAlignment;
    if (neighborParticleStride > std::numeric_limits<size_t>::max() /
                                     static_cast<size_t>(kMaxNeighbors)) {
        throw std::length_error("Neighbor buffer size overflows size_t");
    }

    if (!std::isfinite(params.dt) || params.dt <= 0.0f ||
        !std::isfinite(params.restDensity) || params.restDensity <= 0.0f ||
        !std::isfinite(params.particleMass) || params.particleMass <= 0.0f ||
        !std::isfinite(params.particleRadius) || params.particleRadius <= 0.0f ||
        !std::isfinite(params.collisionRestitution) || params.collisionRestitution < 0.0f ||
        params.collisionRestitution > 1.0f ||
        !std::isfinite(params.collisionFriction) || params.collisionFriction < 0.0f ||
        params.collisionFriction > 1.0f ||
        !std::isfinite(params.smoothingRadius) || params.smoothingRadius <= 0.0f ||
        !std::isfinite(params.lambdaEpsilon) || params.lambdaEpsilon <= 0.0f ||
        !std::isfinite(params.scorrK) || params.scorrK < 0.0f ||
        (params.scorrK > 0.0f &&
         (params.scorrN <= 0 || !std::isfinite(params.scorrDeltaQ) ||
          params.scorrDeltaQ < 0.0f || params.scorrDeltaQ >= params.smoothingRadius)) ||
        !std::isfinite(params.xsphViscosity) || params.xsphViscosity < 0.0f ||
        !std::isfinite(params.vorticityStrength) || params.vorticityStrength < 0.0f ||
        !std::isfinite(params.gravity.x) || !std::isfinite(params.gravity.y) ||
        !std::isfinite(params.gravity.z) ||
        params.solverIterations <= 0 || params.substeps <= 0) {
        throw std::invalid_argument("Invalid PBF simulation parameters");
    }

    const float substepDt = params.dt / static_cast<float>(params.substeps);
    if (!std::isfinite(substepDt) || substepDt <= 0.0f ||
        !std::isfinite(1.0f / substepDt)) {
        throw std::invalid_argument("Substep duration is not representable");
    }

    const double containerWidth =
        static_cast<double>(maxBounds.x) - static_cast<double>(minBounds.x);
    const double containerDepth =
        static_cast<double>(maxBounds.z) - static_cast<double>(minBounds.z);
    const double particleDiameter = 2.0 * static_cast<double>(params.particleRadius);
    if (containerWidth < particleDiameter || containerDepth < particleDiameter)
        throw std::invalid_argument("Particle diameter exceeds closed container dimensions");

    spatialGrid_.initialize(maxParticles, minBounds, maxBounds,
                            params.smoothingRadius);

    // From here allocation failure leaves the solver inactive, so partially
    // replaced buffers can never be used with the previous active count.
    maxParticles_ = 0;
    particleCount_ = 0;

    Container container;
    container.min = minBounds;
    container.max = maxBounds;
    _collisionSystem.setContainer(container);

    positions_.allocate(maxParticles);
    predictedPositions_.allocate(maxParticles);
    velocities_.allocate(maxParticles);
    xsphVelocities_.allocate(maxParticles);
    collisionInputVelocities_.allocate(maxParticles);
    vorticity_.allocate(maxParticles);
    reorderedPositions_.allocate(maxParticles);
    reorderedPredictedPositions_.allocate(maxParticles);
    reorderedVelocities_.allocate(maxParticles);
    reorderedCollisionInputVelocities_.allocate(maxParticles);
    stableParticleIds_.allocate(maxParticles);
    reorderedStableParticleIds_.allocate(maxParticles);
    neighbors_.allocate(neighborParticleStride * static_cast<size_t>(kMaxNeighbors));
    neighborsCount_.allocate(maxParticles);
    neighborOverflow_.allocate(1);
    density_.allocate(maxParticles);
    constraints_.allocate(maxParticles);
    lambda_.allocate(maxParticles);
    deltaPosition_.allocate(maxParticles);

    maxParticles_ = maxParticles;
    particleCount_ = 0;
    neighborParticleStride_ = neighborParticleStride;
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

    // Validate both arrays before either upload can change active state.
    if (positions == nullptr || velocities == nullptr)
        throw std::invalid_argument("Particle positions and velocities must not be null");
    for (size_t i = 0; i < particleCount; ++i) {
        if (!std::isfinite(positions[i].x) || !std::isfinite(positions[i].y) ||
            !std::isfinite(positions[i].z) || !std::isfinite(velocities[i].x) ||
            !std::isfinite(velocities[i].y) || !std::isfinite(velocities[i].z)) {
            throw std::invalid_argument("Particle coordinates and velocities must be finite");
        }
    }

    positions_.copyFromHostToDevice(positions, particleCount);
    velocities_.copyFromHostToDevice(velocities, particleCount);
    const int gridSize = static_cast<int>(
        (particleCount + kBlockSize - 1) / kBlockSize
    );
    initializeStableParticleIds<<<gridSize, kBlockSize>>>(
        stableParticleIds_.data(), particleCount
    );
    checkKernelLaunch();
    particleCount_ = particleCount;
}

void PBFSolver::reorderWorkingSet() {
    const int gridSize = static_cast<int>(
        (particleCount_ + kBlockSize - 1) / kBlockSize
    );
    gatherSolverState<<<gridSize, kBlockSize>>>(
        spatialGrid_.sortedIndices(), positions_.data(), predictedPositions_.data(),
        velocities_.data(), collisionInputVelocities_.data(), stableParticleIds_.data(),
        reorderedPositions_.data(), reorderedPredictedPositions_.data(),
        reorderedVelocities_.data(), reorderedCollisionInputVelocities_.data(),
        reorderedStableParticleIds_.data(), particleCount_
    );
    checkKernelLaunch();

    std::swap(positions_, reorderedPositions_);
    std::swap(predictedPositions_, reorderedPredictedPositions_);
    std::swap(velocities_, reorderedVelocities_);
    std::swap(collisionInputVelocities_, reorderedCollisionInputVelocities_);
    std::swap(stableParticleIds_, reorderedStableParticleIds_);
}

void PBFSolver::setSpheres(const std::vector<SphereCollider>& spheres) {
    _collisionSystem.setSpheres(spheres);
}

void PBFSolver::setBoxes(const std::vector<BoxCollider>& boxes) {
    _collisionSystem.setBoxes(boxes);
}

void PBFSolver::setPlanes(const std::vector<PlaneCollider>& planes) {
    _collisionSystem.setPlanes(planes);
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

        checkCuda(cudaMemcpy(
            collisionInputVelocities_.data(), velocities_.data(),
            particleCount_ * sizeof(float4), cudaMemcpyDeviceToDevice
        ));

        for (int iteration = 0; iteration < params_.solverIterations; ++iteration) {
            spatialGrid_.build(predictedPositions_.data(), particleCount_);
            reorderWorkingSet();

            neighborOverflow_.fillBytes(0);

            findNeighbors<<<gridSize, kBlockSize>>>(
                predictedPositions_.data(),
                spatialGrid_.cellStart(), spatialGrid_.cellEnd(),
                spatialGrid_.gridSize(), spatialGrid_.minBounds(),
                spatialGrid_.cellSize(), particleCount_, params_.smoothingRadius,
                neighbors_.data(), neighborsCount_.data(), kMaxNeighbors,
                neighborParticleStride_,
                neighborOverflow_.data()
            );
            checkKernelLaunch();

            int neighborOverflow = 0;
            neighborOverflow_.copyFromDeviceToHost(&neighborOverflow, 1);
            if (neighborOverflow != 0)
                throw std::overflow_error("Particle neighbor count exceeds solver capacity");

            computeDensity<<<gridSize, kBlockSize>>>(
                predictedPositions_.data(), neighbors_.data(), neighborsCount_.data(),
                kMaxNeighbors, particleCount_, params_.smoothingRadius,
                params_.particleMass, density_.data(), constraints_.data(),
                params_.restDensity, neighborParticleStride_
            );
            checkKernelLaunch();

            computeLambda<<<gridSize, kBlockSize>>>(
                predictedPositions_.data(), neighbors_.data(), neighborsCount_.data(),
                kMaxNeighbors, constraints_.data(), particleCount_,
                params_.smoothingRadius, params_.particleMass, params_.restDensity,
                params_.lambdaEpsilon, lambda_.data(), neighborParticleStride_
            );
            checkKernelLaunch();

            computeDeltaPosition<<<gridSize, kBlockSize>>>(
                predictedPositions_.data(), neighbors_.data(), neighborsCount_.data(),
                kMaxNeighbors, lambda_.data(), particleCount_,
                params_.smoothingRadius, params_.particleMass, params_.restDensity,
                params_.scorrK, params_.scorrN, params_.scorrDeltaQ,
                deltaPosition_.data(), neighborParticleStride_
            );
            checkKernelLaunch();

            applyDeltaPosition<<<gridSize, kBlockSize>>>(
                predictedPositions_.data(), deltaPosition_.data(), particleCount_
            );
            checkKernelLaunch();

            _collisionSystem.solve(
                predictedPositions_.data(), particleCount_, params_.particleRadius
            );
            checkKernelLaunch();
        }

        if (params_.xsphViscosity != 0.0f || params_.vorticityStrength != 0.0f) {
            // The last constraint correction and collision projection changed
            // positions after the final solver neighbor build.  Rebuild before
            // post-solve velocity effects so they use final neighborhoods.
            spatialGrid_.build(predictedPositions_.data(), particleCount_);
            reorderWorkingSet();
            neighborOverflow_.fillBytes(0);
            findNeighbors<<<gridSize, kBlockSize>>>(
                predictedPositions_.data(),
                spatialGrid_.cellStart(), spatialGrid_.cellEnd(),
                spatialGrid_.gridSize(), spatialGrid_.minBounds(),
                spatialGrid_.cellSize(), particleCount_, params_.smoothingRadius,
                neighbors_.data(), neighborsCount_.data(), kMaxNeighbors,
                neighborParticleStride_,
                neighborOverflow_.data()
            );
            checkKernelLaunch();

            int neighborOverflow = 0;
            neighborOverflow_.copyFromDeviceToHost(&neighborOverflow, 1);
            if (neighborOverflow != 0)
                throw std::overflow_error("Particle neighbor count exceeds solver capacity");
        }

        updateVelocityAndPosition<<<gridSize, kBlockSize>>>(
            positions_.data(), predictedPositions_.data(), velocities_.data(),
            particleCount_, 1.0f / substepDt
        );
        checkKernelLaunch();

        if (params_.xsphViscosity != 0.0f) {
            applyXsphViscosity<<<gridSize, kBlockSize>>>(
                predictedPositions_.data(), neighbors_.data(), neighborsCount_.data(),
                kMaxNeighbors, velocities_.data(), xsphVelocities_.data(), particleCount_,
                params_.smoothingRadius, params_.xsphViscosity,
                neighborParticleStride_
            );
            checkKernelLaunch();
            std::swap(velocities_, xsphVelocities_);
        }

        if (params_.vorticityStrength != 0.0f) {
            computeVorticity<<<gridSize, kBlockSize>>>(
                predictedPositions_.data(), velocities_.data(), neighbors_.data(),
                neighborsCount_.data(), kMaxNeighbors, particleCount_,
                params_.smoothingRadius, vorticity_.data(), neighborParticleStride_
            );
            checkKernelLaunch();

            applyVorticityConfinement<<<gridSize, kBlockSize>>>(
                predictedPositions_.data(), neighbors_.data(), neighborsCount_.data(),
                kMaxNeighbors, vorticity_.data(), velocities_.data(),
                xsphVelocities_.data(), particleCount_, params_.smoothingRadius,
                substepDt, params_.vorticityStrength, neighborParticleStride_
            );
            checkKernelLaunch();
            std::swap(velocities_, xsphVelocities_);
        }

        _collisionSystem.resolveVelocities(
            positions_.data(), collisionInputVelocities_.data(), velocities_.data(),
            particleCount_, params_.particleRadius, params_.collisionRestitution,
            params_.collisionFriction
        );
        checkKernelLaunch();
    }

    checkCuda(cudaDeviceSynchronize());
}

void PBFSolver::run() {
    if (maxParticles_ == 0)
        throw std::logic_error("PBFSolver must be initialized before running");

    step();
}

void PBFSolver::run(std::size_t frameCount) {
    if (maxParticles_ == 0)
        throw std::logic_error("PBFSolver must be initialized before running");

    for (std::size_t frame = 0; frame < frameCount; ++frame)
        step();
}

void PBFSolver::copyPositionsToHost(float4* positions, size_t particleCount) const {
    if (particleCount != particleCount_)
        throw std::out_of_range("Requested position count does not match active particles");

    if (particleCount != 0)
    {
        const int gridSize = static_cast<int>(
            (particleCount + kBlockSize - 1) / kBlockSize
        );
        scatterFloat4ByStableId<<<gridSize, kBlockSize>>>(
            positions_.data(), stableParticleIds_.data(),
            reorderedPositions_.data(), particleCount
        );
        checkKernelLaunch();
        reorderedPositions_.copyFromDeviceToHost(positions, particleCount);
    }
}

void PBFSolver::copyVelocitiesToHost(float4* velocities, size_t particleCount) const {
    if (particleCount != particleCount_)
        throw std::out_of_range("Requested velocity count does not match active particles");

    if (particleCount != 0)
    {
        const int gridSize = static_cast<int>(
            (particleCount + kBlockSize - 1) / kBlockSize
        );
        scatterFloat4ByStableId<<<gridSize, kBlockSize>>>(
            velocities_.data(), stableParticleIds_.data(),
            reorderedVelocities_.data(), particleCount
        );
        checkKernelLaunch();
        reorderedVelocities_.copyFromDeviceToHost(velocities, particleCount);
    }
}
