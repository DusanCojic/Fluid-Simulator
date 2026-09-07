#include "pbf/cuda_buffer.hpp"
#include "pbf/pbf_solver.hpp"

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <vector>

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
    neighbors[1] = 2;
    neighbors[2] = 3;
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
        particleCount, smoothingRadius, particleMass, deviceDensity.data(), deviceConstraints.data(), restDensity
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
    neighbors[1] = 2;
    neighbors[2] = 3;
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
        epsilon, deviceLambdas.data()
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
    neighbors[1] = 2;
    neighbors[2] = 3;
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
        deviceLambdas.data(), particleCount, smoothingRadius, particleMass, restDensity, deviceDeltas.data()
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
