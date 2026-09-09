#include "pbf/sph_helpers.cuh"
#include "pbf/cuda_buffer.hpp"

#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>

__global__
void Poly6Kernel(float3 displacement, float smoothingRadius, float* result) {
    *result = poly6(displacement, smoothingRadius);
}

__global__
void SpikyGradientsKernel(float3 displacement, float smoothingRadius, float3* result) {
    *result = spikyGradient(displacement, smoothingRadius);
}

void waitForKernel() {
    cudaError_t error = cudaGetLastError();

    if (error != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(error));

    error = cudaDeviceSynchronize();

    if (error != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(error));
}

float evaluatePoly6(float3 displacement, float smoothingRadius) {
    CudaBuffer<float> deviceResult(1);

    Poly6Kernel<<<1, 1>>>(displacement, smoothingRadius, deviceResult.data());
    waitForKernel();

    float result = 0.0f;
    deviceResult.copyFromDeviceToHost(&result, 1);
    return result;
}

float3 evaluateSpikyGradients(float3 displacement, float smoothingRadius) {
    CudaBuffer<float3> deviceResult(1);

    SpikyGradientsKernel<<<1, 1>>>(
        displacement,
        smoothingRadius,
        deviceResult.data()
    );
    waitForKernel();

    float3 result{};
    deviceResult.copyFromDeviceToHost(&result, 1);
    return result;
}

TEST(Poly6Test, ReturnsZeroOutsideRadius) {
    const float result = evaluatePoly6({3.0f, 4.0f, 0.0f}, 4.0f);

    EXPECT_FLOAT_EQ(result, 0.0f);
}

TEST(Poly6Test, ReturnsZeroAtRadius) {
    const float result = evaluatePoly6({0.0f, 0.0f, 2.0f}, 2.0f);

    EXPECT_FLOAT_EQ(result, 0.0f);
}

TEST(Poly6Test, MatchesExpectedValueInsideRadius) {
    constexpr float pi = 3.14159265358979323846f;
    constexpr float smoothingRadius = 2.0f;
    const float3 displacement = {1.0f, 0.0f, 0.0f};

    const float expected =
        315.0f / (64.0f * pi * std::pow(smoothingRadius, 9.0f)) *
        std::pow(smoothingRadius * smoothingRadius - 1.0f, 3.0f);

    const float result = evaluatePoly6(displacement, smoothingRadius);

    EXPECT_NEAR(result, expected, 1e-6f);
}

TEST(Poly6Test, SameDistanceGivesSameWeight) {
    constexpr float smoothingRadius = 4.0f;

    const float first =
        evaluatePoly6({1.0f, 2.0f, 2.0f}, smoothingRadius);
    const float second =
        evaluatePoly6({-2.0f, -1.0f, 2.0f}, smoothingRadius);

    EXPECT_FLOAT_EQ(first, second);
}

TEST(SpikyGradientsTest, ReturnsZeroAtOrigin) {
    const float3 result =
        evaluateSpikyGradients({0.0f, 0.0f, 0.0f}, 2.0f);

    EXPECT_FLOAT_EQ(result.x, 0.0f);
    EXPECT_FLOAT_EQ(result.y, 0.0f);
    EXPECT_FLOAT_EQ(result.z, 0.0f);
}

TEST(SpikyGradientsTest, ReturnsZeroOutsideRadius) {
    const float3 result =
        evaluateSpikyGradients({3.0f, 4.0f, 0.0f}, 4.0f);

    EXPECT_FLOAT_EQ(result.x, 0.0f);
    EXPECT_FLOAT_EQ(result.y, 0.0f);
    EXPECT_FLOAT_EQ(result.z, 0.0f);
}

TEST(SpikyGradientsTest, ReturnsZeroAtRadius) {
    const float3 result =
        evaluateSpikyGradients({0.0f, 0.0f, 2.0f}, 2.0f);

    EXPECT_FLOAT_EQ(result.x, 0.0f);
    EXPECT_FLOAT_EQ(result.y, 0.0f);
    EXPECT_FLOAT_EQ(result.z, 0.0f);
}

TEST(SpikyGradientsTest, MatchesExpectedValueInsideRadius) {
    constexpr float pi = 3.14159265358979323846f;
    constexpr float smoothingRadius = 4.0f;
    const float3 displacement = {1.0f, 2.0f, 2.0f};
    constexpr float distance = 3.0f;

    const float magnitude =
        -45.0f / (pi * std::pow(smoothingRadius, 6.0f)) *
        std::pow(smoothingRadius - distance, 2.0f);

    const float3 result =
        evaluateSpikyGradients(displacement, smoothingRadius);

    EXPECT_NEAR(result.x, magnitude / 3.0f, 1e-7f);
    EXPECT_NEAR(result.y, 2.0f * magnitude / 3.0f, 1e-7f);
    EXPECT_NEAR(result.z, 2.0f * magnitude / 3.0f, 1e-7f);
}

TEST(SpikyGradientsTest, OppositeDisplacementGivesOppositeGradient) {
    constexpr float smoothingRadius = 4.0f;

    const float3 first =
        evaluateSpikyGradients({1.0f, 2.0f, 2.0f}, smoothingRadius);
    const float3 second =
        evaluateSpikyGradients({-1.0f, -2.0f, -2.0f}, smoothingRadius);

    EXPECT_FLOAT_EQ(first.x, -second.x);
    EXPECT_FLOAT_EQ(first.y, -second.y);
    EXPECT_FLOAT_EQ(first.z, -second.z);
}

TEST(Poly6Test, RepresentableValuesSurviveExtremeIntermediatePowers) {
    for (float h : {1e-5f, 1e5f}) {
        const double expected = 315.0 / (64.0 * 3.14159265358979323846 * std::pow(double(h), 3));
        const float actual = evaluatePoly6({0, 0, 0}, h);
        EXPECT_TRUE(std::isfinite(actual));
        EXPECT_NEAR(actual, expected, expected * 3e-6);
        EXPECT_EQ(evaluatePoly6({h, 0, 0}, h), 0.0f);
    }
}

TEST(SpikyGradientsTest, RepresentableValuesSurviveExtremeIntermediatePowers) {
    const float h = 1e-7f;
    const double expected = -45.0 / (4.0 * 3.14159265358979323846 * std::pow(double(h), 4));
    const auto actual = evaluateSpikyGradients({h / 2, 0, 0}, h);
    EXPECT_TRUE(std::isfinite(actual.x));
    EXPECT_NEAR(actual.x, expected, -expected * 3e-6);
    EXPECT_EQ(actual.y, 0.0f);
    EXPECT_EQ(actual.z, 0.0f);
}
