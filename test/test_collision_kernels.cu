#include "pbf/collision_kernels.cuh"
#include "pbf/cuda_buffer.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <vector>

namespace {

constexpr float tolerance = 1e-5f;
constexpr int blockSize = 256;

void checkCuda(cudaError_t error) {
    if (error != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(error));
}

void finishKernel() {
    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());
}

void expectFloat4Near(const float4& actual, const float4& expected) {
    EXPECT_NEAR(actual.x, expected.x, tolerance);
    EXPECT_NEAR(actual.y, expected.y, tolerance);
    EXPECT_NEAR(actual.z, expected.z, tolerance);
    EXPECT_FLOAT_EQ(actual.w, expected.w);
}

std::vector<float4> solveContainer(
    const std::vector<float4>& positions, const Container& container,
    float particleRadius, std::size_t particleCount = 0
) {
    if (particleCount == 0)
        particleCount = positions.size();

    CudaBuffer<float4> devicePositions(positions.size());
    CudaBuffer<Container> deviceContainer(1);
    devicePositions.copyFromHostToDevice(positions.data(), positions.size());
    deviceContainer.copyFromHostToDevice(&container, 1);

    const int gridSize = static_cast<int>((particleCount + blockSize - 1) / blockSize);
    solveContainerKernel<<<gridSize, blockSize>>>(
        devicePositions.data(), particleCount, deviceContainer.data(), particleRadius
    );
    finishKernel();

    std::vector<float4> result(positions.size());
    devicePositions.copyFromDeviceToHost(result.data(), result.size());
    return result;
}

std::vector<float4> solveSpheres(
    const std::vector<float4>& positions,
    const std::vector<SphereCollider>& spheres,
    float particleRadius
) {
    CudaBuffer<float4> devicePositions(positions.size());
    CudaBuffer<SphereCollider> deviceSpheres(spheres.size());
    devicePositions.copyFromHostToDevice(positions.data(), positions.size());
    deviceSpheres.copyFromHostToDevice(spheres.data(), spheres.size());

    const int gridSize = static_cast<int>((positions.size() + blockSize - 1) / blockSize);
    solveSpheresKernel<<<gridSize, blockSize>>>(
        devicePositions.data(), positions.size(), deviceSpheres.data(),
        spheres.size(), particleRadius
    );
    finishKernel();

    std::vector<float4> result(positions.size());
    devicePositions.copyFromDeviceToHost(result.data(), result.size());
    return result;
}

std::vector<float4> solveBoxes(
    const std::vector<float4>& positions,
    const std::vector<BoxCollider>& boxes,
    float particleRadius
) {
    CudaBuffer<float4> devicePositions(positions.size());
    CudaBuffer<BoxCollider> deviceBoxes(boxes.size());
    devicePositions.copyFromHostToDevice(positions.data(), positions.size());
    deviceBoxes.copyFromHostToDevice(boxes.data(), boxes.size());

    const int gridSize = static_cast<int>((positions.size() + blockSize - 1) / blockSize);
    solveBoxesKernel<<<gridSize, blockSize>>>(
        devicePositions.data(), positions.size(), deviceBoxes.data(),
        boxes.size(), particleRadius
    );
    finishKernel();

    std::vector<float4> result(positions.size());
    devicePositions.copyFromDeviceToHost(result.data(), result.size());
    return result;
}

std::vector<float4> solvePlanes(
    const std::vector<float4>& positions,
    const std::vector<PlaneCollider>& planes,
    float particleRadius
) {
    CudaBuffer<float4> devicePositions(positions.size());
    CudaBuffer<PlaneCollider> devicePlanes(planes.size());
    devicePositions.copyFromHostToDevice(positions.data(), positions.size());
    devicePlanes.copyFromHostToDevice(planes.data(), planes.size());

    const int gridSize = static_cast<int>((positions.size() + blockSize - 1) / blockSize);
    solvePlanesKernel<<<gridSize, blockSize>>>(
        devicePositions.data(), positions.size(), devicePlanes.data(),
        planes.size(), particleRadius
    );
    finishKernel();

    std::vector<float4> result(positions.size());
    devicePositions.copyFromDeviceToHost(result.data(), result.size());
    return result;
}

struct VelocityResult {
    std::vector<float4> positions;
    std::vector<float4> velocities;
};

VelocityResult resolveVelocities(
    const std::vector<float4>& positions,
    const std::vector<float4>& velocities,
    const Container& container,
    const std::vector<SphereCollider>& spheres = {},
    const std::vector<BoxCollider>& boxes = {},
    const std::vector<PlaneCollider>& planes = {},
    float particleRadius = 0.5f,
    float restitution = 0.0f,
    float friction = 0.0f,
    std::size_t particleCount = 0
) {
    if (particleCount == 0)
        particleCount = positions.size();

    CudaBuffer<float4> devicePositions(positions.size());
    CudaBuffer<float4> deviceVelocities(velocities.size());
    CudaBuffer<Container> deviceContainer(1);
    CudaBuffer<SphereCollider> deviceSpheres(spheres.size());
    CudaBuffer<BoxCollider> deviceBoxes(boxes.size());
    CudaBuffer<PlaneCollider> devicePlanes(planes.size());
    devicePositions.copyFromHostToDevice(positions.data(), positions.size());
    deviceVelocities.copyFromHostToDevice(velocities.data(), velocities.size());
    deviceContainer.copyFromHostToDevice(&container, 1);
    deviceSpheres.copyFromHostToDevice(spheres.data(), spheres.size());
    deviceBoxes.copyFromHostToDevice(boxes.data(), boxes.size());
    devicePlanes.copyFromHostToDevice(planes.data(), planes.size());

    const int gridSize = static_cast<int>((particleCount + blockSize - 1) / blockSize);
    resolveVelocitiesKernel<<<gridSize, blockSize>>>(
        devicePositions.data(), deviceVelocities.data(), particleCount,
        deviceContainer.data(), deviceSpheres.data(), spheres.size(),
        deviceBoxes.data(), boxes.size(), devicePlanes.data(), planes.size(),
        particleRadius, restitution, friction
    );
    finishKernel();

    VelocityResult result{ positions, velocities };
    devicePositions.copyFromDeviceToHost(result.positions.data(), result.positions.size());
    deviceVelocities.copyFromDeviceToHost(result.velocities.data(), result.velocities.size());
    return result;
}

Container largeContainer() {
    return {
        make_float3(-10.0f, -10.0f, -10.0f),
        make_float3(10.0f, 10.0f, 10.0f)
    };
}

} // namespace

TEST(SolveContainerKernelTest, ClampsFiveWallsAndLeavesTopOpen) {
    const Container container{ make_float3(0.0f, 0.0f, 0.0f), make_float3(10.0f, 10.0f, 10.0f) };
    const std::vector<float4> positions = {
        make_float4(-2.0f, -3.0f, -4.0f, 1.0f),
        make_float4(12.0f, 15.0f, 14.0f, 2.0f)
    };

    const std::vector<float4> result = solveContainer(positions, container, 1.0f);

    expectFloat4Near(result[0], make_float4(1.0f, 1.0f, 1.0f, 1.0f));
    expectFloat4Near(result[1], make_float4(9.0f, 15.0f, 9.0f, 2.0f));
}

TEST(SolveContainerKernelTest, LeavesInteriorAndBoundaryPositionsUnchanged) {
    const Container container{ make_float3(0.0f, 0.0f, 0.0f), make_float3(10.0f, 10.0f, 10.0f) };
    const std::vector<float4> positions = {
        make_float4(5.0f, 5.0f, 5.0f, 3.0f),
        make_float4(1.0f, 1.0f, 9.0f, 4.0f)
    };

    const std::vector<float4> result = solveContainer(positions, container, 1.0f);

    expectFloat4Near(result[0], positions[0]);
    expectFloat4Near(result[1], positions[1]);
}

TEST(SolveContainerKernelTest, DoesNotModifyElementsPastParticleCount) {
    const Container container{ make_float3(0.0f, 0.0f, 0.0f), make_float3(10.0f, 10.0f, 10.0f) };
    std::vector<float4> positions(258, make_float4(-1.0f, -1.0f, -1.0f, 5.0f));
    positions.back() = make_float4(-7.0f, -8.0f, -9.0f, 6.0f);

    const std::vector<float4> result = solveContainer(positions, container, 0.5f, 257);

    expectFloat4Near(result[256], make_float4(0.5f, 0.5f, 0.5f, 5.0f));
    expectFloat4Near(result.back(), positions.back());
}

TEST(SolveSpheresKernelTest, ProjectsPenetrationAndHandlesCenter) {
    const SphereCollider sphere{ make_float3(1.0f, 2.0f, 3.0f), 2.0f };
    const std::vector<float4> positions = {
        make_float4(2.0f, 2.0f, 3.0f, 1.0f),
        make_float4(1.0f, 2.0f, 3.0f, 2.0f)
    };

    const std::vector<float4> result = solveSpheres(positions, { sphere }, 0.5f);

    expectFloat4Near(result[0], make_float4(3.5f, 2.0f, 3.0f, 1.0f));
    expectFloat4Near(result[1], make_float4(3.5f, 2.0f, 3.0f, 2.0f));
}

TEST(SolveSpheresKernelTest, LeavesContactAndOutsidePositionsUnchanged) {
    const SphereCollider sphere{ make_float3(0.0f, 0.0f, 0.0f), 1.0f };
    const std::vector<float4> positions = {
        make_float4(1.5f, 0.0f, 0.0f, 3.0f),
        make_float4(2.0f, 0.0f, 0.0f, 4.0f)
    };

    const std::vector<float4> result = solveSpheres(positions, { sphere }, 0.5f);

    expectFloat4Near(result[0], positions[0]);
    expectFloat4Near(result[1], positions[1]);
}

TEST(SolveSpheresKernelTest, AppliesMultipleSpheresInArrayOrder) {
    const std::vector<SphereCollider> spheres = {
        { make_float3(0.0f, 0.0f, 0.0f), 1.0f },
        { make_float3(2.5f, 0.0f, 0.0f), 1.0f }
    };

    const std::vector<float4> result = solveSpheres(
        { make_float4(0.0f, 0.0f, 0.0f, 7.0f) }, spheres, 0.5f
    );

    expectFloat4Near(result[0], make_float4(1.0f, 0.0f, 0.0f, 7.0f));
}

TEST(SolveSpheresKernelTest, EmptyColliderArrayLeavesPositionsUnchanged) {
    const std::vector<float4> positions = { make_float4(1.0f, 2.0f, 3.0f, 4.0f) };
    const std::vector<float4> result = solveSpheres(positions, {}, 0.5f);
    expectFloat4Near(result[0], positions[0]);
}

TEST(SolveBoxesKernelTest, ProjectsFaceEdgeAndCornerPenetration) {
    const BoxCollider box{ make_float3(0.0f, 0.0f, 0.0f), make_float3(1.0f, 1.0f, 1.0f) };
    const std::vector<float4> positions = {
        make_float4(1.2f, 0.0f, 0.0f, 1.0f),
        make_float4(1.2f, 1.2f, 0.0f, 2.0f),
        make_float4(1.2f, 1.2f, 1.2f, 3.0f)
    };

    const std::vector<float4> result = solveBoxes(positions, { box }, 0.5f);

    expectFloat4Near(result[0], make_float4(1.5f, 0.0f, 0.0f, 1.0f));
    EXPECT_NEAR(result[1].x, 1.0f + 0.5f / std::sqrt(2.0f), tolerance);
    EXPECT_NEAR(result[1].y, 1.0f + 0.5f / std::sqrt(2.0f), tolerance);
    EXPECT_FLOAT_EQ(result[1].z, 0.0f);
    EXPECT_NEAR(result[2].x, 1.0f + 0.5f / std::sqrt(3.0f), tolerance);
    EXPECT_NEAR(result[2].y, 1.0f + 0.5f / std::sqrt(3.0f), tolerance);
    EXPECT_NEAR(result[2].z, 1.0f + 0.5f / std::sqrt(3.0f), tolerance);
}

TEST(SolveBoxesKernelTest, ProjectsInteriorToNearestPositiveAndNegativeFaces) {
    const BoxCollider box{ make_float3(0.0f, 0.0f, 0.0f), make_float3(1.0f, 1.0f, 1.0f) };
    const std::vector<float4> positions = {
        make_float4(0.8f, 0.0f, 0.0f, 1.0f),
        make_float4(-0.8f, 0.0f, 0.0f, 2.0f),
        make_float4(0.0f, 0.8f, 0.0f, 3.0f),
        make_float4(0.0f, 0.0f, -0.8f, 4.0f)
    };

    const std::vector<float4> result = solveBoxes(positions, { box }, 0.5f);

    expectFloat4Near(result[0], make_float4(1.5f, 0.0f, 0.0f, 1.0f));
    expectFloat4Near(result[1], make_float4(-1.5f, 0.0f, 0.0f, 2.0f));
    expectFloat4Near(result[2], make_float4(0.0f, 1.5f, 0.0f, 3.0f));
    expectFloat4Near(result[3], make_float4(0.0f, 0.0f, -1.5f, 4.0f));
}

TEST(SolveBoxesKernelTest, CenterTieUsesPositiveXFace) {
    const BoxCollider box{ make_float3(0.0f, 0.0f, 0.0f), make_float3(1.0f, 1.0f, 1.0f) };
    const std::vector<float4> result = solveBoxes(
        { make_float4(0.0f, 0.0f, 0.0f, 9.0f) }, { box }, 0.5f
    );
    expectFloat4Near(result[0], make_float4(1.5f, 0.0f, 0.0f, 9.0f));
}

TEST(SolveBoxesKernelTest, LeavesExactContactAndOutsidePositionsUnchanged) {
    const BoxCollider box{ make_float3(0.0f, 0.0f, 0.0f), make_float3(1.0f, 1.0f, 1.0f) };
    const std::vector<float4> positions = {
        make_float4(1.5f, 0.0f, 0.0f, 1.0f),
        make_float4(2.0f, 2.0f, 0.0f, 2.0f)
    };

    const std::vector<float4> result = solveBoxes(positions, { box }, 0.5f);

    expectFloat4Near(result[0], positions[0]);
    expectFloat4Near(result[1], positions[1]);
}

TEST(SolvePlanesKernelTest, UsesNormalizedNormalAndProjectsNegativeSide) {
    const PlaneCollider plane{ make_float3(0.0f, 1.0f, 0.0f), make_float3(0.0f, 2.0f, 0.0f) };
    const std::vector<float4> positions = {
        make_float4(2.0f, 1.2f, 3.0f, 1.0f),
        make_float4(2.0f, -4.0f, 3.0f, 2.0f)
    };

    const std::vector<float4> result = solvePlanes(positions, { plane }, 0.5f);

    expectFloat4Near(result[0], make_float4(2.0f, 1.5f, 3.0f, 1.0f));
    expectFloat4Near(result[1], make_float4(2.0f, 1.5f, 3.0f, 2.0f));
}

TEST(SolvePlanesKernelTest, LeavesContactAndPositiveSideUnchanged) {
    const PlaneCollider plane{ make_float3(0.0f, 1.0f, 0.0f), make_float3(0.0f, 1.0f, 0.0f) };
    const std::vector<float4> positions = {
        make_float4(0.0f, 1.5f, 0.0f, 3.0f),
        make_float4(0.0f, 2.0f, 0.0f, 4.0f)
    };

    const std::vector<float4> result = solvePlanes(positions, { plane }, 0.5f);

    expectFloat4Near(result[0], positions[0]);
    expectFloat4Near(result[1], positions[1]);
}

TEST(SolvePlanesKernelTest, IgnoresZeroNormalAndAppliesMultiplePlanes) {
    const std::vector<PlaneCollider> planes = {
        { make_float3(0.0f, 0.0f, 0.0f), make_float3(0.0f, 0.0f, 0.0f) },
        { make_float3(1.0f, 0.0f, 0.0f), make_float3(1.0f, 0.0f, 0.0f) },
        { make_float3(0.0f, 2.0f, 0.0f), make_float3(0.0f, 1.0f, 0.0f) }
    };

    const std::vector<float4> result = solvePlanes(
        { make_float4(0.0f, 0.0f, 3.0f, 5.0f) }, planes, 0.5f
    );

    expectFloat4Near(result[0], make_float4(1.5f, 2.5f, 3.0f, 5.0f));
}

TEST(CollisionPositionKernelsTest, EmptyBoxAndPlaneArraysLeavePositionsUnchanged) {
    const std::vector<float4> positions = { make_float4(1.0f, 2.0f, 3.0f, 4.0f) };
    expectFloat4Near(solveBoxes(positions, {}, 0.5f)[0], positions[0]);
    expectFloat4Near(solvePlanes(positions, {}, 0.5f)[0], positions[0]);
}

TEST(ResolveVelocitiesKernelTest, ResolvesEveryContainerWallAndLeavesTopOpen) {
    const Container container{ make_float3(0.0f, 0.0f, 0.0f), make_float3(10.0f, 10.0f, 10.0f) };
    const std::vector<float4> positions = {
        make_float4(0.5f, 5.0f, 5.0f, 1.0f),
        make_float4(9.5f, 5.0f, 5.0f, 2.0f),
        make_float4(5.0f, 0.5f, 5.0f, 3.0f),
        make_float4(5.0f, 5.0f, 0.5f, 4.0f),
        make_float4(5.0f, 5.0f, 9.5f, 5.0f),
        make_float4(5.0f, 20.0f, 5.0f, 6.0f)
    };
    const std::vector<float4> velocities = {
        make_float4(-2.0f, 4.0f, 0.0f, 11.0f),
        make_float4(2.0f, 4.0f, 0.0f, 12.0f),
        make_float4(4.0f, -2.0f, 0.0f, 13.0f),
        make_float4(4.0f, 0.0f, -2.0f, 14.0f),
        make_float4(4.0f, 0.0f, 2.0f, 15.0f),
        make_float4(1.0f, 2.0f, 3.0f, 16.0f)
    };

    const VelocityResult result = resolveVelocities(
        positions, velocities, container, {}, {}, {}, 0.5f, 0.5f, 0.25f
    );

    expectFloat4Near(result.velocities[0], make_float4(1.0f, 3.0f, 0.0f, 11.0f));
    expectFloat4Near(result.velocities[1], make_float4(-1.0f, 3.0f, 0.0f, 12.0f));
    expectFloat4Near(result.velocities[2], make_float4(3.0f, 1.0f, 0.0f, 13.0f));
    expectFloat4Near(result.velocities[3], make_float4(3.0f, 0.0f, 1.0f, 14.0f));
    expectFloat4Near(result.velocities[4], make_float4(3.0f, 0.0f, -1.0f, 15.0f));
    expectFloat4Near(result.velocities[5], velocities[5]);
}

TEST(ResolveVelocitiesKernelTest, LeavesInteriorAndSeparatingVelocitiesUnchanged) {
    const std::vector<float4> positions = {
        make_float4(0.5f, 5.0f, 5.0f, 1.0f),
        make_float4(5.0f, 5.0f, 5.0f, 2.0f)
    };
    const std::vector<float4> velocities = {
        make_float4(2.0f, 3.0f, 4.0f, 5.0f),
        make_float4(-2.0f, 3.0f, 4.0f, 6.0f)
    };
    const Container container{ make_float3(0.0f, 0.0f, 0.0f), make_float3(10.0f, 10.0f, 10.0f) };

    const VelocityResult result = resolveVelocities(positions, velocities, container);

    expectFloat4Near(result.velocities[0], velocities[0]);
    expectFloat4Near(result.velocities[1], velocities[1]);
}

TEST(ResolveVelocitiesKernelTest, UsesContactTolerance) {
    const Container container{ make_float3(0.0f, 0.0f, 0.0f), make_float3(10.0f, 10.0f, 10.0f) };
    const VelocityResult result = resolveVelocities(
        { make_float4(0.50005f, 5.0f, 5.0f, 1.0f) },
        { make_float4(-2.0f, 0.0f, 0.0f, 2.0f) },
        container, {}, {}, {}, 0.5f, 0.0f, 0.0f
    );
    expectFloat4Near(result.velocities[0], make_float4(0.0f, 0.0f, 0.0f, 2.0f));
}

TEST(ResolveVelocitiesKernelTest, UsesToleranceForEveryColliderType) {
    const SphereCollider sphere{ make_float3(0.0f, 0.0f, 0.0f), 1.0f };
    const BoxCollider box{ make_float3(0.0f, 0.0f, 0.0f), make_float3(1.0f, 1.0f, 1.0f) };
    const PlaneCollider plane{ make_float3(0.0f, 0.0f, 0.0f), make_float3(0.0f, 1.0f, 0.0f) };
    const float4 incomingX = make_float4(-2.0f, 0.0f, 0.0f, 1.0f);
    const float4 incomingY = make_float4(0.0f, -2.0f, 0.0f, 2.0f);

    const VelocityResult sphereResult = resolveVelocities(
        { make_float4(1.50005f, 0.0f, 0.0f, 3.0f) }, { incomingX },
        largeContainer(), { sphere }
    );
    const VelocityResult boxResult = resolveVelocities(
        { make_float4(1.50005f, 0.0f, 0.0f, 4.0f) }, { incomingX },
        largeContainer(), {}, { box }
    );
    const VelocityResult planeResult = resolveVelocities(
        { make_float4(0.0f, 0.50005f, 0.0f, 5.0f) }, { incomingY },
        largeContainer(), {}, {}, { plane }
    );

    EXPECT_NEAR(sphereResult.velocities[0].x, 0.0f, tolerance);
    EXPECT_NEAR(boxResult.velocities[0].x, 0.0f, tolerance);
    EXPECT_NEAR(planeResult.velocities[0].y, 0.0f, tolerance);
}

TEST(ResolveVelocitiesKernelTest, ResolvesSphereRestitutionFrictionAndDirection) {
    const SphereCollider sphere{ make_float3(0.0f, 0.0f, 0.0f), 1.0f };
    const std::vector<float4> positions = {
        make_float4(1.5f, 0.0f, 0.0f, 1.0f),
        make_float4(1.5f, 0.0f, 0.0f, 2.0f),
        make_float4(2.0f, 0.0f, 0.0f, 3.0f)
    };
    const std::vector<float4> velocities = {
        make_float4(-2.0f, 4.0f, 0.0f, 4.0f),
        make_float4(2.0f, 4.0f, 0.0f, 5.0f),
        make_float4(-2.0f, 4.0f, 0.0f, 6.0f)
    };

    const VelocityResult result = resolveVelocities(
        positions, velocities, largeContainer(), { sphere }, {}, {}, 0.5f, 0.5f, 0.25f
    );

    expectFloat4Near(result.velocities[0], make_float4(1.0f, 3.0f, 0.0f, 4.0f));
    expectFloat4Near(result.velocities[1], velocities[1]);
    expectFloat4Near(result.velocities[2], velocities[2]);
}

TEST(ResolveVelocitiesKernelTest, SupportsFullRestitutionAndFullFriction) {
    const SphereCollider sphere{ make_float3(0.0f, 0.0f, 0.0f), 1.0f };
    const VelocityResult result = resolveVelocities(
        { make_float4(1.5f, 0.0f, 0.0f, 1.0f) },
        { make_float4(-2.0f, 4.0f, 0.0f, 2.0f) },
        largeContainer(), { sphere }, {}, {}, 0.5f, 1.0f, 1.0f
    );
    expectFloat4Near(result.velocities[0], make_float4(2.0f, 0.0f, 0.0f, 2.0f));
}

TEST(ResolveVelocitiesKernelTest, IgnoresSphereCenterDegeneracy) {
    const SphereCollider sphere{ make_float3(0.0f, 0.0f, 0.0f), 1.0f };
    const float4 velocity = make_float4(-2.0f, 3.0f, 4.0f, 5.0f);
    const VelocityResult result = resolveVelocities(
        { make_float4(0.0f, 0.0f, 0.0f, 1.0f) }, { velocity },
        largeContainer(), { sphere }
    );
    expectFloat4Near(result.velocities[0], velocity);
}

TEST(ResolveVelocitiesKernelTest, ResolvesRoundedBoxEdgeAndInteriorFallback) {
    const BoxCollider box{ make_float3(0.0f, 0.0f, 0.0f), make_float3(1.0f, 1.0f, 1.0f) };
    const float edge = 1.0f + 0.5f / std::sqrt(2.0f);
    const std::vector<float4> positions = {
        make_float4(edge, edge, 0.0f, 1.0f),
        make_float4(0.8f, 0.0f, 0.0f, 2.0f),
        make_float4(3.0f, 0.0f, 0.0f, 3.0f)
    };
    const std::vector<float4> velocities = {
        make_float4(-2.0f, -2.0f, 2.0f, 4.0f),
        make_float4(-2.0f, 4.0f, 0.0f, 5.0f),
        make_float4(-2.0f, 4.0f, 0.0f, 6.0f)
    };

    const VelocityResult result = resolveVelocities(
        positions, velocities, largeContainer(), {}, { box }, {}, 0.5f, 0.0f, 0.5f
    );

    expectFloat4Near(result.velocities[0], make_float4(0.0f, 0.0f, 1.0f, 4.0f));
    expectFloat4Near(result.velocities[1], make_float4(0.0f, 2.0f, 0.0f, 5.0f));
    expectFloat4Near(result.velocities[2], velocities[2]);
}

TEST(ResolveVelocitiesKernelTest, ResolvesNonUnitPlaneAndIgnoresZeroNormal) {
    const std::vector<PlaneCollider> planes = {
        { make_float3(0.0f, 0.0f, 0.0f), make_float3(0.0f, 0.0f, 0.0f) },
        { make_float3(0.0f, 1.0f, 0.0f), make_float3(0.0f, 2.0f, 0.0f) }
    };
    const VelocityResult result = resolveVelocities(
        { make_float4(0.0f, 1.5f, 0.0f, 1.0f) },
        { make_float4(4.0f, -2.0f, 0.0f, 2.0f) },
        largeContainer(), {}, {}, planes, 0.5f, 0.5f, 0.25f
    );
    expectFloat4Near(result.velocities[0], make_float4(3.0f, 1.0f, 0.0f, 2.0f));
}

TEST(ResolveVelocitiesKernelTest, LeavesSeparatingAndDistantPlaneVelocitiesUnchanged) {
    const PlaneCollider plane{ make_float3(0.0f, 0.0f, 0.0f), make_float3(0.0f, 1.0f, 0.0f) };
    const std::vector<float4> velocities = {
        make_float4(1.0f, 2.0f, 3.0f, 4.0f),
        make_float4(1.0f, -2.0f, 3.0f, 5.0f)
    };
    const VelocityResult result = resolveVelocities(
        { make_float4(0.0f, 0.5f, 0.0f, 6.0f), make_float4(0.0f, 1.0f, 0.0f, 7.0f) },
        velocities, largeContainer(), {}, {}, { plane }
    );
    expectFloat4Near(result.velocities[0], velocities[0]);
    expectFloat4Near(result.velocities[1], velocities[1]);
}

TEST(ResolveVelocitiesKernelTest, LeavesPositionsAndElementsPastCountUnchanged) {
    std::vector<float4> positions(258, make_float4(-9.5f, 0.0f, 0.0f, 1.0f));
    std::vector<float4> velocities(258, make_float4(-2.0f, 0.0f, 0.0f, 2.0f));
    positions.back() = make_float4(-9.5f, 3.0f, 4.0f, 7.0f);
    velocities.back() = make_float4(-8.0f, 5.0f, 6.0f, 9.0f);

    const VelocityResult result = resolveVelocities(
        positions, velocities, largeContainer(), {}, {}, {}, 0.5f,
        0.0f, 0.0f, 257
    );

    EXPECT_EQ(result.positions.size(), positions.size());
    for (std::size_t i = 0; i < positions.size(); ++i)
        expectFloat4Near(result.positions[i], positions[i]);
    expectFloat4Near(result.velocities[256], make_float4(0.0f, 0.0f, 0.0f, 2.0f));
    expectFloat4Near(result.velocities.back(), velocities.back());
}
