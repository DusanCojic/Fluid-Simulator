#include "pbf/cuda_buffer.hpp"
#include "pbf/pbf_solver.hpp"

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <vector>

namespace {

__global__
void computeArtificialPressureForTest(const float* distanceSquared, std::size_t count,
                                      float smoothingRadius, float scorrK, int scorrN,
                                      float scorrDeltaQ, float* result) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (index < count) {
        result[index] = computeArtificialPressure(
            distanceSquared[index], smoothingRadius, scorrK, scorrN, scorrDeltaQ
        );
    }
}

float poly6Reference(float3 displacement, float smoothingRadius) {
    const float distanceSquared =
        displacement.x * displacement.x +
        displacement.y * displacement.y +
        displacement.z * displacement.z;
    const float radiusSquared = smoothingRadius * smoothingRadius;

    if (distanceSquared > radiusSquared)
        return 0.0f;

    constexpr float pi = 3.14159265358979323846f;
    const float radiusDifference = radiusSquared - distanceSquared;
    return 315.0f / (64.0f * pi * std::pow(smoothingRadius, 9.0f)) *
        radiusDifference * radiusDifference * radiusDifference;
}

float3 spikyGradientReference(float3 displacement, float smoothingRadius) {
    const float distanceSquared =
        displacement.x * displacement.x +
        displacement.y * displacement.y +
        displacement.z * displacement.z;
    const float distance = std::sqrt(distanceSquared);
    if (distance <= 0.0f || distance > smoothingRadius)
        return make_float3(0.0f, 0.0f, 0.0f);

    constexpr float pi = 3.14159265358979323846f;
    const float magnitude = -45.0f / (pi * std::pow(smoothingRadius, 6.0f)) *
        (smoothingRadius - distance) * (smoothingRadius - distance);
    return make_float3(
        displacement.x / distance * magnitude,
        displacement.y / distance * magnitude,
        displacement.z / distance * magnitude
    );
}

float3 crossReference(float3 left, float3 right) {
    return make_float3(
        left.y * right.z - left.z * right.y,
        left.z * right.x - left.x * right.z,
        left.x * right.y - left.y * right.x
    );
}

float lengthReference(float3 value) {
    return std::sqrt(value.x * value.x + value.y * value.y + value.z * value.z);
}

} // namespace

TEST(ArtificialPressureTest, MatchesPaperFormulaAndVanishesAtKernelBoundary) {
    constexpr float smoothingRadius = 1.0f;
    constexpr float scorrK = 0.001f;
    constexpr int scorrN = 4;
    constexpr float scorrDeltaQ = 0.3f;
    constexpr int blockSize = 32;

    const std::vector<float> distanceSquared = {
        scorrDeltaQ * scorrDeltaQ, 0.25f, 1.0f, 1.44f
    };
    CudaBuffer<float> deviceDistanceSquared(distanceSquared.size());
    CudaBuffer<float> deviceResult(distanceSquared.size());
    deviceDistanceSquared.copyFromHostToDevice(
        distanceSquared.data(), distanceSquared.size()
    );

    computeArtificialPressureForTest<<<1, blockSize>>>(
        deviceDistanceSquared.data(), distanceSquared.size(), smoothingRadius,
        scorrK, scorrN, scorrDeltaQ, deviceResult.data()
    );
    ASSERT_EQ(cudaSuccess, cudaGetLastError());
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

    std::vector<float> result(distanceSquared.size());
    deviceResult.copyFromDeviceToHost(result.data(), result.size());

    const float kernelRatio = std::pow(
        (1.0f - 0.25f) / (1.0f - scorrDeltaQ * scorrDeltaQ), 3.0f
    );
    const float expectedAtHalfRadius =
        -scorrK * std::pow(kernelRatio, static_cast<float>(scorrN));

    EXPECT_NEAR(result[0], -scorrK, 1e-7f);
    EXPECT_NEAR(result[1], expectedAtHalfRadius, 1e-7f);
    EXPECT_FLOAT_EQ(result[2], 0.0f);
    EXPECT_FLOAT_EQ(result[3], 0.0f);
}

TEST(ArtificialPressureTest, ReturnsZeroWhenDisabled) {
    constexpr float distanceSquared = 0.04f;
    CudaBuffer<float> deviceDistanceSquared(1);
    CudaBuffer<float> deviceResult(1);
    deviceDistanceSquared.copyFromHostToDevice(&distanceSquared, 1);

    computeArtificialPressureForTest<<<1, 1>>>(
        deviceDistanceSquared.data(), 1, 1.0f, 0.0f, 4, 0.3f,
        deviceResult.data()
    );
    ASSERT_EQ(cudaSuccess, cudaGetLastError());
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

    float result = 1.0f;
    deviceResult.copyFromDeviceToHost(&result, 1);
    EXPECT_FLOAT_EQ(result, 0.0f);
}

TEST(PbfSolverTest, ComputeDensityIncludesSelfAndNeighborsAndHandlesPartialBlocks) {
    constexpr std::size_t particleCount = 513;
    constexpr int maxNeighbors = 3;
    constexpr float smoothingRadius = 1.0f;
    constexpr float particleMass = 2.0f;
    constexpr float restDensity = 3.0f;
    constexpr int blockSize = 256;

    std::vector<float4> positions(particleCount, make_float4(5.0f, 5.0f, 5.0f, 9.0f));
    positions[0] = make_float4(0.0f, 0.0f, 0.0f, 7.0f);
    positions[1] = make_float4(0.5f, 0.5f, 0.25f, -3.0f);
    positions[2] = make_float4(0.0f, 0.6f, 0.8f, 2.0f);
    positions[3] = make_float4(0.0f, 0.0f, 1.25f, -8.0f);

    std::vector<std::uint32_t> neighbors(particleCount * maxNeighbors, 0);
    neighbors[0] = 1;
    neighbors[particleCount] = 2;
    neighbors[2 * particleCount] = 3;
    std::vector<int> neighborCounts(particleCount, 0);
    neighborCounts[0] = maxNeighbors;

    CudaBuffer<float4> devicePositions(particleCount);
    CudaBuffer<std::uint32_t> deviceNeighbors(neighbors.size());
    CudaBuffer<int> deviceNeighborCounts(particleCount);
    CudaBuffer<float> deviceDensity(particleCount);
    CudaBuffer<float> deviceConstraints(particleCount);
    devicePositions.copyFromHostToDevice(positions.data(), particleCount);
    deviceNeighbors.copyFromHostToDevice(neighbors.data(), neighbors.size());
    deviceNeighborCounts.copyFromHostToDevice(neighborCounts.data(), particleCount);

    computeDensity<<<(particleCount + blockSize - 1) / blockSize, blockSize>>>(
        devicePositions.data(), deviceNeighbors.data(), deviceNeighborCounts.data(), maxNeighbors,
        particleCount, smoothingRadius, particleMass, deviceDensity.data(),
        deviceConstraints.data(), restDensity, particleCount
    );
    ASSERT_EQ(cudaSuccess, cudaGetLastError());
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

    std::vector<float> density(particleCount);
    std::vector<float> constraints(particleCount);
    deviceDensity.copyFromDeviceToHost(density.data(), particleCount);
    deviceConstraints.copyFromDeviceToHost(constraints.data(), particleCount);

    constexpr float pi = 3.14159265358979323846f;
    const float selfWeight = 315.0f / (64.0f * pi);
    const float neighborWeight = selfWeight * std::pow(1.0f - 0.5625f, 3.0f);
    const float expectedDensity = particleMass * (selfWeight + neighborWeight);
    const float expectedSelfDensity = particleMass * selfWeight;

    EXPECT_NEAR(density[0], expectedDensity, 1e-5f);
    EXPECT_NEAR(constraints[0], expectedDensity / restDensity - 1.0f, 1e-5f);
    EXPECT_NEAR(density[1], expectedSelfDensity, 1e-5f);
    EXPECT_NEAR(constraints[1], expectedSelfDensity / restDensity - 1.0f, 1e-5f);
    EXPECT_NEAR(density.back(), expectedSelfDensity, 1e-5f);
}

TEST(PbfSolverTest, ComputeLambdaUsesAllGradientsAndZeroNeighborRegularization) {
    constexpr std::size_t particleCount = 257;
    constexpr int maxNeighbors = 3;
    constexpr float smoothingRadius = 1.0f;
    constexpr float particleMass = 2.0f;
    constexpr float restDensity = 4.0f;
    constexpr float epsilon = 0.01f;
    constexpr int blockSize = 256;

    std::vector<float4> positions(particleCount, make_float4(4.0f, 4.0f, 4.0f, 1.0f));
    positions[0] = make_float4(0.0f, 0.0f, 0.0f, 8.0f);
    positions[1] = make_float4(0.25f, 0.5f, 0.5f, -4.0f);
    positions[2] = make_float4(0.0f, 0.6f, 0.8f, 6.0f);
    positions[3] = make_float4(0.0f, 0.0f, 1.25f, -2.0f);
    std::vector<std::uint32_t> neighbors(particleCount * maxNeighbors, 0);
    neighbors[0] = 1;
    neighbors[particleCount] = 2;
    neighbors[2 * particleCount] = 3;
    std::vector<int> neighborCounts(particleCount, 0);
    neighborCounts[0] = maxNeighbors;
    std::vector<float> constraints(particleCount, 0.0f);
    constraints[0] = 0.5f;
    constraints[1] = -0.25f;
    constraints.back() = 0.75f;

    CudaBuffer<float4> devicePositions(particleCount);
    CudaBuffer<std::uint32_t> deviceNeighbors(neighbors.size());
    CudaBuffer<int> deviceNeighborCounts(particleCount);
    CudaBuffer<float> deviceConstraints(particleCount);
    CudaBuffer<float> deviceLambdas(particleCount);
    devicePositions.copyFromHostToDevice(positions.data(), particleCount);
    deviceNeighbors.copyFromHostToDevice(neighbors.data(), neighbors.size());
    deviceNeighborCounts.copyFromHostToDevice(neighborCounts.data(), particleCount);
    deviceConstraints.copyFromHostToDevice(constraints.data(), particleCount);

    computeLambda<<<(particleCount + blockSize - 1) / blockSize, blockSize>>>(
        devicePositions.data(), deviceNeighbors.data(), deviceNeighborCounts.data(), maxNeighbors,
        deviceConstraints.data(), particleCount, smoothingRadius, particleMass, restDensity,
        epsilon, deviceLambdas.data(), particleCount
    );
    ASSERT_EQ(cudaSuccess, cudaGetLastError());
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

    std::vector<float> lambdas(particleCount);
    deviceLambdas.copyFromDeviceToHost(lambdas.data(), particleCount);

    constexpr float pi = 3.14159265358979323846f;
    const float gradient = 45.0f / pi * std::pow(0.25f, 2.0f);
    const float gradientScale = particleMass / restDensity;
    const float gradientSum = 2.0f * std::pow(gradientScale * gradient, 2.0f);

    EXPECT_NEAR(lambdas[0], -constraints[0] / (gradientSum + epsilon), 1e-5f);
    EXPECT_NEAR(lambdas[1], -constraints[1] / epsilon, 1e-5f);
    EXPECT_NEAR(lambdas.back(), -constraints.back() / epsilon, 1e-5f);
}

TEST(PbfSolverTest, ComputeDeltaPositionAppliesLambdaCorrectionAndLeavesWZero) {
    constexpr std::size_t particleCount = 257;
    constexpr int maxNeighbors = 3;
    constexpr float smoothingRadius = 1.0f;
    constexpr float particleMass = 2.0f;
    constexpr float restDensity = 4.0f;
    constexpr int blockSize = 256;

    std::vector<float4> positions(particleCount, make_float4(4.0f, 4.0f, 4.0f, 1.0f));
    positions[0] = make_float4(0.0f, 0.0f, 0.0f, 3.0f);
    positions[1] = make_float4(0.25f, 0.5f, 0.5f, 5.0f);
    positions[2] = make_float4(0.0f, 0.6f, 0.8f, 7.0f);
    positions[3] = make_float4(0.0f, 0.0f, 1.25f, -1.0f);
    std::vector<std::uint32_t> neighbors(particleCount * maxNeighbors, 0);
    neighbors[0] = 1;
    neighbors[particleCount] = 2;
    neighbors[2 * particleCount] = 3;
    std::vector<int> neighborCounts(particleCount, 0);
    neighborCounts[0] = maxNeighbors;
    std::vector<float> lambdas(particleCount, 0.0f);
    lambdas[0] = 2.0f;
    lambdas[1] = -0.5f;
    lambdas[2] = 3.0f;

    CudaBuffer<float4> devicePositions(particleCount);
    CudaBuffer<std::uint32_t> deviceNeighbors(neighbors.size());
    CudaBuffer<int> deviceNeighborCounts(particleCount);
    CudaBuffer<float> deviceLambdas(particleCount);
    CudaBuffer<float4> deviceDeltas(particleCount);
    devicePositions.copyFromHostToDevice(positions.data(), particleCount);
    deviceNeighbors.copyFromHostToDevice(neighbors.data(), neighbors.size());
    deviceNeighborCounts.copyFromHostToDevice(neighborCounts.data(), particleCount);
    deviceLambdas.copyFromHostToDevice(lambdas.data(), particleCount);

    computeDeltaPosition<<<(particleCount + blockSize - 1) / blockSize, blockSize>>>(
        devicePositions.data(), deviceNeighbors.data(), deviceNeighborCounts.data(), maxNeighbors,
        deviceLambdas.data(), particleCount, smoothingRadius, particleMass, restDensity,
        0.0f, 0, 0.0f, deviceDeltas.data(), particleCount
    );
    ASSERT_EQ(cudaSuccess, cudaGetLastError());
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

    std::vector<float4> deltas(particleCount);
    deviceDeltas.copyFromDeviceToHost(deltas.data(), particleCount);

    constexpr float pi = 3.14159265358979323846f;
    const float gradientMagnitude = 45.0f / pi * std::pow(0.25f, 2.0f);
    const float lambdaScale = particleMass / restDensity * (lambdas[0] + lambdas[1]);
    const float expectedX = lambdaScale * gradientMagnitude / 3.0f;
    const float expectedY = 2.0f * expectedX;
    const float expectedZ = 2.0f * expectedX;

    EXPECT_NEAR(deltas[0].x, expectedX, 1e-5f);
    EXPECT_NEAR(deltas[0].y, expectedY, 1e-5f);
    EXPECT_NEAR(deltas[0].z, expectedZ, 1e-5f);
    EXPECT_FLOAT_EQ(deltas[0].w, 0.0f);
    EXPECT_FLOAT_EQ(deltas[1].x, 0.0f);
    EXPECT_FLOAT_EQ(deltas[1].y, 0.0f);
    EXPECT_FLOAT_EQ(deltas[1].z, 0.0f);
    EXPECT_FLOAT_EQ(deltas.back().w, 0.0f);
}

TEST(PbfSolverTest, ApplyDeltaPositionUpdatesOnlySpatialCoordinatesAcrossBlocks) {
    constexpr std::size_t particleCount = 513;
    constexpr int blockSize = 256;

    std::vector<float4> positions(particleCount);
    std::vector<float4> deltas(particleCount);
    for (std::size_t i = 0; i < particleCount; ++i) {
        positions[i] = make_float4(static_cast<float>(i), -static_cast<float>(i), 2.0f * i, 10.0f + i);
        deltas[i] = make_float4(-0.25f * i, 0.5f * i, -0.75f * i, -100.0f - i);
    }

    CudaBuffer<float4> devicePositions(particleCount);
    CudaBuffer<float4> deviceDeltas(particleCount);
    devicePositions.copyFromHostToDevice(positions.data(), particleCount);
    deviceDeltas.copyFromHostToDevice(deltas.data(), particleCount);

    applyDeltaPosition<<<(particleCount + blockSize - 1) / blockSize, blockSize>>>(
        devicePositions.data(), deviceDeltas.data(), particleCount
    );
    ASSERT_EQ(cudaSuccess, cudaGetLastError());
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

    std::vector<float4> result(particleCount);
    devicePositions.copyFromDeviceToHost(result.data(), particleCount);
    for (std::size_t i = 0; i < particleCount; ++i) {
        EXPECT_FLOAT_EQ(result[i].x, positions[i].x + deltas[i].x) << "particle " << i;
        EXPECT_FLOAT_EQ(result[i].y, positions[i].y + deltas[i].y) << "particle " << i;
        EXPECT_FLOAT_EQ(result[i].z, positions[i].z + deltas[i].z) << "particle " << i;
        EXPECT_FLOAT_EQ(result[i].w, positions[i].w) << "particle " << i;
    }
}

TEST(PbfSolverTest, UpdateVelocityAndPositionUpdatesStateAndPreservesVelocityWAcrossPartialBlocks) {
    constexpr std::size_t particleCount = 513;
    constexpr int blockSize = 256;
    constexpr float inverseDt = 8.0f;

    std::vector<float4> positions(particleCount);
    std::vector<float4> predictedPositions(particleCount);
    std::vector<float4> velocities(particleCount);

    for (std::size_t i = 0; i < particleCount; ++i) {
        const float index = static_cast<float>(i);
        positions[i] = make_float4(0.5f * index, -2.0f * index, index + 3.0f, -10.0f - index);
        predictedPositions[i] = make_float4(
            positions[i].x + 0.125f * (index + 1.0f),
            positions[i].y - 0.25f * (index + 2.0f),
            positions[i].z + 0.5f * (index + 3.0f),
            100.0f + index
        );
        velocities[i] = make_float4(-7.0f - index, 8.0f + index, 9.0f - index, 50.0f + index);
    }

    CudaBuffer<float4> devicePositions(particleCount);
    CudaBuffer<float4> devicePredictedPositions(particleCount);
    CudaBuffer<float4> deviceVelocities(particleCount);
    devicePositions.copyFromHostToDevice(positions.data(), particleCount);
    devicePredictedPositions.copyFromHostToDevice(predictedPositions.data(), particleCount);
    deviceVelocities.copyFromHostToDevice(velocities.data(), particleCount);

    updateVelocityAndPosition<<<(particleCount + blockSize - 1) / blockSize, blockSize>>>(
        devicePositions.data(), devicePredictedPositions.data(), deviceVelocities.data(),
        particleCount, inverseDt
    );
    ASSERT_EQ(cudaSuccess, cudaGetLastError());
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

    std::vector<float4> resultPositions(particleCount);
    std::vector<float4> resultPredictedPositions(particleCount);
    std::vector<float4> resultVelocities(particleCount);
    devicePositions.copyFromDeviceToHost(resultPositions.data(), particleCount);
    devicePredictedPositions.copyFromDeviceToHost(resultPredictedPositions.data(), particleCount);
    deviceVelocities.copyFromDeviceToHost(resultVelocities.data(), particleCount);

    for (std::size_t i = 0; i < particleCount; ++i) {
        EXPECT_FLOAT_EQ(resultPositions[i].x, predictedPositions[i].x) << "particle " << i;
        EXPECT_FLOAT_EQ(resultPositions[i].y, predictedPositions[i].y) << "particle " << i;
        EXPECT_FLOAT_EQ(resultPositions[i].z, predictedPositions[i].z) << "particle " << i;
        EXPECT_FLOAT_EQ(resultPositions[i].w, predictedPositions[i].w) << "particle " << i;

        EXPECT_FLOAT_EQ(resultVelocities[i].x,
                        (predictedPositions[i].x - positions[i].x) * inverseDt) << "particle " << i;
        EXPECT_FLOAT_EQ(resultVelocities[i].y,
                        (predictedPositions[i].y - positions[i].y) * inverseDt) << "particle " << i;
        EXPECT_FLOAT_EQ(resultVelocities[i].z,
                        (predictedPositions[i].z - positions[i].z) * inverseDt) << "particle " << i;
        EXPECT_FLOAT_EQ(resultVelocities[i].w, velocities[i].w) << "particle " << i;

        EXPECT_FLOAT_EQ(resultPredictedPositions[i].x, predictedPositions[i].x) << "particle " << i;
        EXPECT_FLOAT_EQ(resultPredictedPositions[i].y, predictedPositions[i].y) << "particle " << i;
        EXPECT_FLOAT_EQ(resultPredictedPositions[i].z, predictedPositions[i].z) << "particle " << i;
        EXPECT_FLOAT_EQ(resultPredictedPositions[i].w, predictedPositions[i].w) << "particle " << i;
    }
}

TEST(PbfSolverTest, UpdateVelocityAndPositionDoesNotModifyElementsPastParticleCount) {
    constexpr std::size_t particleCount = 257;
    constexpr std::size_t capacity = particleCount + 1;
    constexpr int blockSize = 256;
    constexpr float inverseDt = 2.0f;

    std::vector<float4> positions(capacity, make_float4(-1.0f, -2.0f, -3.0f, -4.0f));
    std::vector<float4> predictedPositions(capacity, make_float4(5.0f, 6.0f, 7.0f, 8.0f));
    std::vector<float4> velocities(capacity, make_float4(9.0f, 10.0f, 11.0f, 12.0f));

    CudaBuffer<float4> devicePositions(capacity);
    CudaBuffer<float4> devicePredictedPositions(capacity);
    CudaBuffer<float4> deviceVelocities(capacity);
    devicePositions.copyFromHostToDevice(positions.data(), capacity);
    devicePredictedPositions.copyFromHostToDevice(predictedPositions.data(), capacity);
    deviceVelocities.copyFromHostToDevice(velocities.data(), capacity);

    updateVelocityAndPosition<<<(particleCount + blockSize - 1) / blockSize, blockSize>>>(
        devicePositions.data(), devicePredictedPositions.data(), deviceVelocities.data(),
        particleCount, inverseDt
    );
    ASSERT_EQ(cudaSuccess, cudaGetLastError());
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

    std::vector<float4> resultPositions(capacity);
    std::vector<float4> resultVelocities(capacity);
    devicePositions.copyFromDeviceToHost(resultPositions.data(), capacity);
    deviceVelocities.copyFromDeviceToHost(resultVelocities.data(), capacity);

    for (std::size_t i = 0; i < particleCount; ++i) {
        EXPECT_FLOAT_EQ(resultPositions[i].x, predictedPositions[i].x) << "particle " << i;
        EXPECT_FLOAT_EQ(resultPositions[i].y, predictedPositions[i].y) << "particle " << i;
        EXPECT_FLOAT_EQ(resultPositions[i].z, predictedPositions[i].z) << "particle " << i;
        EXPECT_FLOAT_EQ(resultPositions[i].w, predictedPositions[i].w) << "particle " << i;
        EXPECT_FLOAT_EQ(resultVelocities[i].x, 12.0f) << "particle " << i;
        EXPECT_FLOAT_EQ(resultVelocities[i].y, 16.0f) << "particle " << i;
        EXPECT_FLOAT_EQ(resultVelocities[i].z, 20.0f) << "particle " << i;
        EXPECT_FLOAT_EQ(resultVelocities[i].w, velocities[i].w) << "particle " << i;
    }

    const std::size_t guardIndex = particleCount;
    EXPECT_FLOAT_EQ(resultPositions[guardIndex].x, positions[guardIndex].x);
    EXPECT_FLOAT_EQ(resultPositions[guardIndex].y, positions[guardIndex].y);
    EXPECT_FLOAT_EQ(resultPositions[guardIndex].z, positions[guardIndex].z);
    EXPECT_FLOAT_EQ(resultPositions[guardIndex].w, positions[guardIndex].w);
    EXPECT_FLOAT_EQ(resultVelocities[guardIndex].x, velocities[guardIndex].x);
    EXPECT_FLOAT_EQ(resultVelocities[guardIndex].y, velocities[guardIndex].y);
    EXPECT_FLOAT_EQ(resultVelocities[guardIndex].z, velocities[guardIndex].z);
    EXPECT_FLOAT_EQ(resultVelocities[guardIndex].w, velocities[guardIndex].w);
}

TEST(XsphViscosityTest, DisabledLeavesVelocitiesUnchanged) {
    constexpr int maxNeighbors = 2;
    const std::vector<float4> positions = {
        make_float4(0.0f, 0.0f, 0.0f, 1.0f),
        make_float4(0.25f, 0.0f, 0.0f, 2.0f)
    };
    const std::vector<float4> inputVelocities = {
        make_float4(1.0f, -2.0f, 3.0f, 4.0f),
        make_float4(-4.0f, 5.0f, -6.0f, 7.0f)
    };
    const std::vector<uint32_t> neighbors = {1, 0, 0, 0};
    const std::vector<int> neighborCounts = {1, 1};

    CudaBuffer<float4> devicePositions(positions.size());
    CudaBuffer<float4> deviceInputVelocities(inputVelocities.size());
    CudaBuffer<float4> deviceOutputVelocities(inputVelocities.size());
    CudaBuffer<uint32_t> deviceNeighbors(neighbors.size());
    CudaBuffer<int> deviceNeighborCounts(neighborCounts.size());
    devicePositions.copyFromHostToDevice(positions.data(), positions.size());
    deviceInputVelocities.copyFromHostToDevice(inputVelocities.data(), inputVelocities.size());
    deviceNeighbors.copyFromHostToDevice(neighbors.data(), neighbors.size());
    deviceNeighborCounts.copyFromHostToDevice(neighborCounts.data(), neighborCounts.size());

    applyXsphViscosity<<<1, 32>>>(
        devicePositions.data(), deviceNeighbors.data(), deviceNeighborCounts.data(),
        maxNeighbors, deviceInputVelocities.data(), deviceOutputVelocities.data(),
        positions.size(), 1.0f, 0.0f, positions.size()
    );
    ASSERT_EQ(cudaSuccess, cudaGetLastError());
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

    std::vector<float4> outputVelocities(inputVelocities.size());
    std::vector<float4> unchangedInputVelocities(inputVelocities.size());
    deviceOutputVelocities.copyFromDeviceToHost(outputVelocities.data(), outputVelocities.size());
    deviceInputVelocities.copyFromDeviceToHost(
        unchangedInputVelocities.data(), unchangedInputVelocities.size()
    );

    for (std::size_t i = 0; i < inputVelocities.size(); ++i) {
        EXPECT_FLOAT_EQ(outputVelocities[i].x, inputVelocities[i].x);
        EXPECT_FLOAT_EQ(outputVelocities[i].y, inputVelocities[i].y);
        EXPECT_FLOAT_EQ(outputVelocities[i].z, inputVelocities[i].z);
        EXPECT_FLOAT_EQ(outputVelocities[i].w, inputVelocities[i].w);
        EXPECT_FLOAT_EQ(unchangedInputVelocities[i].x, inputVelocities[i].x);
        EXPECT_FLOAT_EQ(unchangedInputVelocities[i].y, inputVelocities[i].y);
        EXPECT_FLOAT_EQ(unchangedInputVelocities[i].z, inputVelocities[i].z);
        EXPECT_FLOAT_EQ(unchangedInputVelocities[i].w, inputVelocities[i].w);
    }
}

TEST(XsphViscosityTest, NoNeighborsAndEqualVelocitiesLeaveVelocityUnchanged) {
    constexpr int maxNeighbors = 2;
    const std::vector<float4> positions = {
        make_float4(0.0f, 0.0f, 0.0f, 1.0f),
        make_float4(0.25f, 0.0f, 0.0f, 2.0f),
        make_float4(0.5f, 0.0f, 0.0f, 3.0f)
    };
    const std::vector<float4> velocities = {
        make_float4(2.0f, -1.0f, 0.5f, 4.0f),
        make_float4(2.0f, -1.0f, 0.5f, 5.0f),
        make_float4(-3.0f, 1.0f, 7.0f, 6.0f)
    };
    const std::vector<uint32_t> neighbors = {1, 0, 0, 0, 0, 0};
    const std::vector<int> neighborCounts = {1, 1, 0};

    CudaBuffer<float4> devicePositions(positions.size());
    CudaBuffer<float4> deviceVelocities(velocities.size());
    CudaBuffer<float4> deviceOutputVelocities(velocities.size());
    CudaBuffer<uint32_t> deviceNeighbors(neighbors.size());
    CudaBuffer<int> deviceNeighborCounts(neighborCounts.size());
    devicePositions.copyFromHostToDevice(positions.data(), positions.size());
    deviceVelocities.copyFromHostToDevice(velocities.data(), velocities.size());
    deviceNeighbors.copyFromHostToDevice(neighbors.data(), neighbors.size());
    deviceNeighborCounts.copyFromHostToDevice(neighborCounts.data(), neighborCounts.size());

    applyXsphViscosity<<<1, 32>>>(
        devicePositions.data(), deviceNeighbors.data(), deviceNeighborCounts.data(),
        maxNeighbors, deviceVelocities.data(), deviceOutputVelocities.data(),
        positions.size(), 1.0f, 0.25f, positions.size()
    );
    ASSERT_EQ(cudaSuccess, cudaGetLastError());
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

    std::vector<float4> outputVelocities(velocities.size());
    deviceOutputVelocities.copyFromDeviceToHost(outputVelocities.data(), outputVelocities.size());
    for (std::size_t i = 0; i < velocities.size(); ++i) {
        EXPECT_FLOAT_EQ(outputVelocities[i].x, velocities[i].x);
        EXPECT_FLOAT_EQ(outputVelocities[i].y, velocities[i].y);
        EXPECT_FLOAT_EQ(outputVelocities[i].z, velocities[i].z);
        EXPECT_FLOAT_EQ(outputVelocities[i].w, velocities[i].w);
    }
}

TEST(XsphViscosityTest, AppliesPoly6WeightedJacobiVelocityCorrection) {
    constexpr int maxNeighbors = 2;
    constexpr float smoothingRadius = 1.0f;
    constexpr float viscosity = 0.15f;
    const std::vector<float4> positions = {
        make_float4(0.0f, 0.0f, 0.0f, 1.0f),
        make_float4(0.5f, 0.0f, 0.0f, 2.0f),
        make_float4(2.0f, 0.0f, 0.0f, 3.0f)
    };
    const std::vector<float4> velocities = {
        make_float4(1.0f, 2.0f, -1.0f, 4.0f),
        make_float4(-3.0f, 4.0f, 5.0f, 5.0f),
        make_float4(10.0f, -2.0f, 1.0f, 6.0f)
    };
    // Particle 2 is deliberately listed for particle 0 but lies outside h.
    const std::vector<uint32_t> neighbors = {1, 0, 0, 2, 0, 0};
    const std::vector<int> neighborCounts = {2, 0, 0};

    CudaBuffer<float4> devicePositions(positions.size());
    CudaBuffer<float4> deviceVelocities(velocities.size());
    CudaBuffer<float4> deviceOutputVelocities(velocities.size());
    CudaBuffer<uint32_t> deviceNeighbors(neighbors.size());
    CudaBuffer<int> deviceNeighborCounts(neighborCounts.size());
    devicePositions.copyFromHostToDevice(positions.data(), positions.size());
    deviceVelocities.copyFromHostToDevice(velocities.data(), velocities.size());
    deviceNeighbors.copyFromHostToDevice(neighbors.data(), neighbors.size());
    deviceNeighborCounts.copyFromHostToDevice(neighborCounts.data(), neighborCounts.size());

    applyXsphViscosity<<<1, 32>>>(
        devicePositions.data(), deviceNeighbors.data(), deviceNeighborCounts.data(),
        maxNeighbors, deviceVelocities.data(), deviceOutputVelocities.data(),
        positions.size(), smoothingRadius, viscosity, positions.size()
    );
    ASSERT_EQ(cudaSuccess, cudaGetLastError());
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

    std::vector<float4> outputVelocities(velocities.size());
    deviceOutputVelocities.copyFromDeviceToHost(outputVelocities.data(), outputVelocities.size());

    const float weight = poly6Reference(make_float3(-0.5f, 0.0f, 0.0f), smoothingRadius);
    const float4 expected = make_float4(
        velocities[0].x + viscosity * (velocities[1].x - velocities[0].x) * weight,
        velocities[0].y + viscosity * (velocities[1].y - velocities[0].y) * weight,
        velocities[0].z + viscosity * (velocities[1].z - velocities[0].z) * weight,
        velocities[0].w
    );
    EXPECT_NEAR(outputVelocities[0].x, expected.x, 1e-6f);
    EXPECT_NEAR(outputVelocities[0].y, expected.y, 1e-6f);
    EXPECT_NEAR(outputVelocities[0].z, expected.z, 1e-6f);
    EXPECT_FLOAT_EQ(outputVelocities[0].w, expected.w);
    EXPECT_FLOAT_EQ(outputVelocities[1].x, velocities[1].x);
    EXPECT_FLOAT_EQ(outputVelocities[2].x, velocities[2].x);
}

TEST(VorticityConfinementTest, DisabledAndNoNeighborsLeaveVelocitiesUnchanged) {
    constexpr int maxNeighbors = 1;
    const std::vector<float4> positions = {
        make_float4(0.0f, 0.0f, 0.0f, 1.0f),
        make_float4(0.5f, 0.0f, 0.0f, 2.0f)
    };
    const std::vector<float4> velocities = {
        make_float4(1.0f, -2.0f, 3.0f, 4.0f),
        make_float4(-4.0f, 5.0f, -6.0f, 7.0f)
    };
    const std::vector<uint32_t> neighbors = {1, 0};
    const std::vector<int> neighborCounts = {1, 0};

    CudaBuffer<float4> devicePositions(positions.size());
    CudaBuffer<float4> deviceVelocities(velocities.size());
    CudaBuffer<float4> deviceVorticity(velocities.size());
    CudaBuffer<float4> deviceOutput(velocities.size());
    CudaBuffer<float4> deviceNoNeighborOutput(velocities.size());
    CudaBuffer<uint32_t> deviceNeighbors(neighbors.size());
    CudaBuffer<int> deviceNeighborCounts(neighborCounts.size());
    devicePositions.copyFromHostToDevice(positions.data(), positions.size());
    deviceVelocities.copyFromHostToDevice(velocities.data(), velocities.size());
    deviceNeighbors.copyFromHostToDevice(neighbors.data(), neighbors.size());
    deviceNeighborCounts.copyFromHostToDevice(neighborCounts.data(), neighborCounts.size());

    computeVorticity<<<1, 32>>>(
        devicePositions.data(), deviceVelocities.data(), deviceNeighbors.data(),
        deviceNeighborCounts.data(), maxNeighbors, positions.size(), 1.0f,
        deviceVorticity.data(), positions.size()
    );
    applyVorticityConfinement<<<1, 32>>>(
        devicePositions.data(), deviceNeighbors.data(), deviceNeighborCounts.data(),
        maxNeighbors, deviceVorticity.data(), deviceVelocities.data(), deviceOutput.data(),
        positions.size(), 1.0f, 0.1f, 0.0f, positions.size()
    );
    applyVorticityConfinement<<<1, 32>>>(
        devicePositions.data(), deviceNeighbors.data(), deviceNeighborCounts.data(),
        maxNeighbors, deviceVorticity.data(), deviceVelocities.data(),
        deviceNoNeighborOutput.data(), positions.size(), 1.0f, 0.1f, 2.0f,
        positions.size()
    );
    ASSERT_EQ(cudaSuccess, cudaGetLastError());
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

    std::vector<float4> omega(positions.size());
    std::vector<float4> output(positions.size());
    std::vector<float4> noNeighborOutput(positions.size());
    deviceVorticity.copyFromDeviceToHost(omega.data(), omega.size());
    deviceOutput.copyFromDeviceToHost(output.data(), output.size());
    deviceNoNeighborOutput.copyFromDeviceToHost(noNeighborOutput.data(), noNeighborOutput.size());
    EXPECT_FLOAT_EQ(omega[1].x, 0.0f);
    EXPECT_FLOAT_EQ(omega[1].y, 0.0f);
    EXPECT_FLOAT_EQ(omega[1].z, 0.0f);
    for (std::size_t i = 0; i < velocities.size(); ++i) {
        EXPECT_FLOAT_EQ(output[i].x, velocities[i].x);
        EXPECT_FLOAT_EQ(output[i].y, velocities[i].y);
        EXPECT_FLOAT_EQ(output[i].z, velocities[i].z);
        EXPECT_FLOAT_EQ(output[i].w, velocities[i].w);
    }
    EXPECT_FLOAT_EQ(noNeighborOutput[1].x, velocities[1].x);
    EXPECT_FLOAT_EQ(noNeighborOutput[1].y, velocities[1].y);
    EXPECT_FLOAT_EQ(noNeighborOutput[1].z, velocities[1].z);
    EXPECT_FLOAT_EQ(noNeighborOutput[1].w, velocities[1].w);
}

TEST(VorticityConfinementTest, UniformVelocityAndZeroEtaProduceFiniteUnchangedOutput) {
    constexpr int maxNeighbors = 2;
    const std::vector<float4> positions = {
        make_float4(0.0f, 0.0f, 0.0f, 1.0f),
        make_float4(0.5f, 0.0f, 0.0f, 2.0f),
        make_float4(0.0f, 0.5f, 0.0f, 3.0f)
    };
    const std::vector<float4> velocities(positions.size(), make_float4(2.0f, -1.0f, 0.5f, 9.0f));
    const std::vector<uint32_t> neighbors = {1, 0, 0, 2, 2, 1};
    const std::vector<int> neighborCounts = {2, 2, 2};

    CudaBuffer<float4> devicePositions(positions.size());
    CudaBuffer<float4> deviceVelocities(velocities.size());
    CudaBuffer<float4> deviceVorticity(velocities.size());
    CudaBuffer<float4> deviceOutput(velocities.size());
    CudaBuffer<uint32_t> deviceNeighbors(neighbors.size());
    CudaBuffer<int> deviceNeighborCounts(neighborCounts.size());
    devicePositions.copyFromHostToDevice(positions.data(), positions.size());
    deviceVelocities.copyFromHostToDevice(velocities.data(), velocities.size());
    deviceNeighbors.copyFromHostToDevice(neighbors.data(), neighbors.size());
    deviceNeighborCounts.copyFromHostToDevice(neighborCounts.data(), neighborCounts.size());

    computeVorticity<<<1, 32>>>(devicePositions.data(), deviceVelocities.data(),
        deviceNeighbors.data(), deviceNeighborCounts.data(), maxNeighbors, positions.size(), 1.0f,
        deviceVorticity.data(), positions.size());
    applyVorticityConfinement<<<1, 32>>>(devicePositions.data(), deviceNeighbors.data(),
        deviceNeighborCounts.data(), maxNeighbors, deviceVorticity.data(), deviceVelocities.data(),
        deviceOutput.data(), positions.size(), 1.0f, 0.1f, 3.0f, positions.size());
    ASSERT_EQ(cudaSuccess, cudaGetLastError());
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

    std::vector<float4> omega(positions.size());
    std::vector<float4> output(positions.size());
    deviceVorticity.copyFromDeviceToHost(omega.data(), omega.size());
    deviceOutput.copyFromDeviceToHost(output.data(), output.size());
    for (std::size_t i = 0; i < positions.size(); ++i) {
        EXPECT_FLOAT_EQ(omega[i].x, 0.0f);
        EXPECT_FLOAT_EQ(omega[i].y, 0.0f);
        EXPECT_FLOAT_EQ(omega[i].z, 0.0f);
        EXPECT_TRUE(std::isfinite(output[i].x));
        EXPECT_TRUE(std::isfinite(output[i].y));
        EXPECT_TRUE(std::isfinite(output[i].z));
        EXPECT_FLOAT_EQ(output[i].x, velocities[i].x);
        EXPECT_FLOAT_EQ(output[i].y, velocities[i].y);
        EXPECT_FLOAT_EQ(output[i].z, velocities[i].z);
    }
}

TEST(VorticityConfinementTest, MatchesCpuReferenceAndDoesNotModifyInputVelocity) {
    constexpr int maxNeighbors = 2;
    constexpr float smoothingRadius = 1.0f;
    constexpr float dt = 0.1f;
    constexpr float strength = 0.25f;
    const std::vector<float4> positions = {
        make_float4(0.0f, 0.0f, 0.0f, 1.0f),
        make_float4(0.5f, 0.0f, 0.0f, 2.0f),
        make_float4(0.0f, 0.5f, 0.0f, 3.0f)
    };
    // v = (-y, x, 0), a deterministic rotational velocity field.
    const std::vector<float4> velocities = {
        make_float4(0.0f, 0.0f, 0.0f, 4.0f),
        make_float4(0.0f, 0.5f, 0.0f, 5.0f),
        make_float4(-0.5f, 0.0f, 0.0f, 6.0f)
    };
    const std::vector<uint32_t> neighbors = {1, 0, 0, 2, 2, 1};
    const std::vector<int> neighborCounts = {2, 2, 2};

    std::vector<float3> expectedOmega(positions.size(), make_float3(0.0f, 0.0f, 0.0f));
    for (std::size_t i = 0; i < positions.size(); ++i) {
        for (int offset = 0; offset < neighborCounts[i]; ++offset) {
            const uint32_t j = neighbors[offset * positions.size() + i];
            const float3 displacement = make_float3(positions[i].x - positions[j].x,
                                                      positions[i].y - positions[j].y,
                                                      positions[i].z - positions[j].z);
            const float3 velocityDifference = make_float3(velocities[j].x - velocities[i].x,
                                                           velocities[j].y - velocities[i].y,
                                                           velocities[j].z - velocities[i].z);
            const float3 contribution = crossReference(
                spikyGradientReference(displacement, smoothingRadius), velocityDifference
            );
            expectedOmega[i].x += contribution.x;
            expectedOmega[i].y += contribution.y;
            expectedOmega[i].z += contribution.z;
        }
    }
    ASSERT_GT(lengthReference(expectedOmega[0]), 0.0f);

    std::vector<float4> expectedVelocity = velocities;
    for (std::size_t i = 0; i < positions.size(); ++i) {
        float3 eta = make_float3(0.0f, 0.0f, 0.0f);
        for (int offset = 0; offset < neighborCounts[i]; ++offset) {
            const uint32_t j = neighbors[offset * positions.size() + i];
            const float3 displacement = make_float3(positions[i].x - positions[j].x,
                                                      positions[i].y - positions[j].y,
                                                      positions[i].z - positions[j].z);
            const float difference = lengthReference(expectedOmega[j]) - lengthReference(expectedOmega[i]);
            const float3 gradient = spikyGradientReference(displacement, smoothingRadius);
            eta.x += difference * gradient.x;
            eta.y += difference * gradient.y;
            eta.z += difference * gradient.z;
        }
        const float etaLength = lengthReference(eta);
        const float3 normal = etaLength > 1.0e-6f
            ? make_float3(eta.x / etaLength, eta.y / etaLength, eta.z / etaLength)
            : make_float3(0.0f, 0.0f, 0.0f);
        const float3 force = crossReference(normal, expectedOmega[i]);
        expectedVelocity[i].x += dt * strength * force.x;
        expectedVelocity[i].y += dt * strength * force.y;
        expectedVelocity[i].z += dt * strength * force.z;
    }

    CudaBuffer<float4> devicePositions(positions.size());
    CudaBuffer<float4> deviceVelocities(velocities.size());
    CudaBuffer<float4> deviceVorticity(velocities.size());
    CudaBuffer<float4> deviceOutput(velocities.size());
    CudaBuffer<uint32_t> deviceNeighbors(neighbors.size());
    CudaBuffer<int> deviceNeighborCounts(neighborCounts.size());
    devicePositions.copyFromHostToDevice(positions.data(), positions.size());
    deviceVelocities.copyFromHostToDevice(velocities.data(), velocities.size());
    deviceNeighbors.copyFromHostToDevice(neighbors.data(), neighbors.size());
    deviceNeighborCounts.copyFromHostToDevice(neighborCounts.data(), neighborCounts.size());

    computeVorticity<<<1, 32>>>(devicePositions.data(), deviceVelocities.data(),
        deviceNeighbors.data(), deviceNeighborCounts.data(), maxNeighbors, positions.size(),
        smoothingRadius, deviceVorticity.data(), positions.size());
    applyVorticityConfinement<<<1, 32>>>(devicePositions.data(), deviceNeighbors.data(),
        deviceNeighborCounts.data(), maxNeighbors, deviceVorticity.data(), deviceVelocities.data(),
        deviceOutput.data(), positions.size(), smoothingRadius, dt, strength,
        positions.size());
    ASSERT_EQ(cudaSuccess, cudaGetLastError());
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

    std::vector<float4> actualOmega(positions.size());
    std::vector<float4> actualVelocity(positions.size());
    std::vector<float4> unchangedInput(positions.size());
    deviceVorticity.copyFromDeviceToHost(actualOmega.data(), actualOmega.size());
    deviceOutput.copyFromDeviceToHost(actualVelocity.data(), actualVelocity.size());
    deviceVelocities.copyFromDeviceToHost(unchangedInput.data(), unchangedInput.size());
    for (std::size_t i = 0; i < positions.size(); ++i) {
        EXPECT_NEAR(actualOmega[i].x, expectedOmega[i].x, 1e-5f);
        EXPECT_NEAR(actualOmega[i].y, expectedOmega[i].y, 1e-5f);
        EXPECT_NEAR(actualOmega[i].z, expectedOmega[i].z, 1e-5f);
        EXPECT_FLOAT_EQ(actualOmega[i].w, 0.0f);
        EXPECT_NEAR(actualVelocity[i].x, expectedVelocity[i].x, 1e-5f);
        EXPECT_NEAR(actualVelocity[i].y, expectedVelocity[i].y, 1e-5f);
        EXPECT_NEAR(actualVelocity[i].z, expectedVelocity[i].z, 1e-5f);
        EXPECT_FLOAT_EQ(actualVelocity[i].w, velocities[i].w);
        EXPECT_FLOAT_EQ(unchangedInput[i].x, velocities[i].x);
        EXPECT_FLOAT_EQ(unchangedInput[i].y, velocities[i].y);
        EXPECT_FLOAT_EQ(unchangedInput[i].z, velocities[i].z);
        EXPECT_FLOAT_EQ(unchangedInput[i].w, velocities[i].w);
    }

    // v = (-y, x, 0) has positive z curl. Check the physical sign instead
    // of only duplicating the kernel algebra in the CPU reference.
    EXPECT_GT(actualOmega[0].z, 0.0f);
}
