#include "pbf/cuda_buffer.hpp"
#include "pbf/integration.cuh"

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <array>

TEST(IntegrationTest, MultipleParticleIntegrationTest) {
    constexpr int particleCount = 1025;

    std::array<float4, particleCount> hPosition{};
    std::array<float4, particleCount> hVelocity{};

    for (int i = 0; i < particleCount; ++i) {
        hPosition[i] = {
            1.0f + 0.01f * i,
            2.0f - 0.02f * i,
            3.0f + 0.03f * i,
            1.0f
        };
        hVelocity[i] = {
            -2.0f + 0.005f * i,
            0.5f - 0.003f * i,
            -1.0f + 0.002f * i,
            0.0f
        };
    }

    CudaBuffer<float4> dPosition(particleCount);
    CudaBuffer<float4> dPredicted(particleCount);
    CudaBuffer<float4> dVelocity(particleCount);

    dPosition.copyFromHostToDevice(hPosition.data(), particleCount);
    dVelocity.copyFromHostToDevice(hVelocity.data(), particleCount);

    float dt = 0.1f;
    float3 gravity = { 0.0f, -10.0f, 0.0f };

    int blockSize = 256;
    int gridSize = (particleCount + blockSize - 1) / blockSize;

    integrate<<<gridSize, blockSize>>>(
        dPosition.data(),
        dPredicted.data(),
        dVelocity.data(),
        particleCount,
        dt,
        gravity
    );

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(error));

    error = cudaDeviceSynchronize();
    if (error != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(error));

    std::array<float4, particleCount> resultPosition{};
    std::array<float4, particleCount> resultVelocity{};
    std::array<float4, particleCount> resultPredicted{};

    dPosition.copyFromDeviceToHost(resultPosition.data(), particleCount);
    dVelocity.copyFromDeviceToHost(resultVelocity.data(), particleCount);
    dPredicted.copyFromDeviceToHost(resultPredicted.data(), particleCount);

    constexpr float epsilon = 1e-5f;

    for (int i = 0; i < particleCount; i++) {
        EXPECT_EQ(resultPosition[i].x, hPosition[i].x);
        EXPECT_EQ(resultPosition[i].y, hPosition[i].y);
        EXPECT_EQ(resultPosition[i].z, hPosition[i].z);

        // v_new = v_old + g*dt
        float expectedVelocityX = hVelocity[i].x + gravity.x * dt;
        float expectedVelocityY = hVelocity[i].y + gravity.y * dt;
        float expectedVelocityZ = hVelocity[i].z + gravity.z * dt;

        EXPECT_NEAR(resultVelocity[i].x, expectedVelocityX, epsilon);
        EXPECT_NEAR(resultVelocity[i].y, expectedVelocityY, epsilon);
        EXPECT_NEAR(resultVelocity[i].z, expectedVelocityZ, epsilon);

        // p_pred = p + v*dt
        float expectedPositionX = hPosition[i].x + expectedVelocityX * dt;
        float expectedPositionY = hPosition[i].y + expectedVelocityY * dt;
        float expectedPositionZ = hPosition[i].z + expectedVelocityZ * dt;

        EXPECT_NEAR(resultPredicted[i].x, expectedPositionX, epsilon);
        EXPECT_NEAR(resultPredicted[i].y, expectedPositionY, epsilon);
        EXPECT_NEAR(resultPredicted[i].z, expectedPositionZ, epsilon);
    }
}
