#include "pbf/cuda_buffer.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

__global__
void multiplyByTwo(float* data, std::size_t count) {
    std::size_t i =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;

    if (i >= count)
        return;

    data[i] *= 2.0f;
}


__global__
void fillWithIndex(float* data, std::size_t count) {
    std::size_t i =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;

    if (i >= count)
        return;

    data[i] = static_cast<float>(i);
}


::testing::AssertionResult cudaKernelCompletedSuccessfully() {
    cudaError_t error = cudaGetLastError();

    if (error != cudaSuccess)
        return ::testing::AssertionFailure()
            << "CUDA kernel launch failed: " << cudaGetErrorString(error);

    error = cudaDeviceSynchronize();

    if (error != cudaSuccess)
        return ::testing::AssertionFailure()
            << "CUDA kernel execution failed: " << cudaGetErrorString(error);

    return ::testing::AssertionSuccess();
}

TEST(CudaBufferTest, CopiesDataFromHostToDeviceAndBack) {
    constexpr std::size_t bufferSize = 1024;

    std::vector<float> input(bufferSize);
    std::vector<float> output(bufferSize);

    for (std::size_t i = 0; i < bufferSize; i++)
        input[i] = static_cast<float>(i);

    CudaBuffer<float> buffer(bufferSize);

    buffer.copyFromHostToDevice(input.data(), bufferSize);
    buffer.copyFromDeviceToHost(output.data(), bufferSize);

    EXPECT_EQ(output, input);
}

TEST(CudaBufferTest, CopiesPartOfBuffer) {
    constexpr std::size_t bufferSize = 1024;
    constexpr std::size_t copySize = 128;

    std::vector<float> input(copySize);
    std::vector<float> output(copySize);

    for (std::size_t i = 0; i < copySize; i++)
        input[i] = static_cast<float>(i);

    CudaBuffer<float> buffer(bufferSize);

    buffer.copyFromHostToDevice(input.data(), copySize);
    buffer.copyFromDeviceToHost(output.data(), copySize);

    EXPECT_EQ(output, input);
}

TEST(CudaBufferTest, ExposesDataToCudaKernel) {
    constexpr std::size_t bufferSize = 10000;

    std::vector<float> input(bufferSize);
    std::vector<float> output(bufferSize);

    for (std::size_t i = 0; i < bufferSize; i++)
        input[i] = static_cast<float>(i);

    CudaBuffer<float> buffer(bufferSize);

    buffer.copyFromHostToDevice(input.data(), bufferSize);

    constexpr int blockSize = 256;

    const int gridSize =
        static_cast<int>(
            (bufferSize + blockSize - 1) / blockSize
        );

    multiplyByTwo<<<gridSize, blockSize>>>(buffer.data(), bufferSize);

    ASSERT_TRUE(cudaKernelCompletedSuccessfully());

    buffer.copyFromDeviceToHost(output.data(), bufferSize);

    for (std::size_t i = 0; i < bufferSize; ++i) {
        const float expected = input[i] * 2.0f;
        EXPECT_NEAR(output[i], expected, 1e-6f) << "at index " << i;
    }
}

TEST(CudaBufferTest, MoveConstructorTransfersGpuAllocation) {
    constexpr std::size_t bufferSize = 2048;

    std::vector<float> input(bufferSize);
    std::vector<float> output(bufferSize);

    for (std::size_t i = 0; i < bufferSize; i++)
        input[i] = static_cast<float>(i);

    CudaBuffer<float> source(bufferSize);
    source.copyFromHostToDevice(input.data(), bufferSize);

    float* originalPtr = source.data();

    CudaBuffer<float> destination(std::move(source));

    EXPECT_EQ(source.data(), nullptr);
    EXPECT_EQ(source.size(), 0U);

    EXPECT_EQ(destination.data(), originalPtr);
    EXPECT_EQ(destination.size(), bufferSize);

    destination.copyFromDeviceToHost(output.data(), bufferSize);

    EXPECT_EQ(output, input);
}

TEST(CudaBufferTest, MoveAssignmentTransfersGpuAllocation) {
    constexpr std::size_t bufferSize = 4096;

    std::vector<float> input(bufferSize);
    std::vector<float> output(bufferSize);

    for (std::size_t i = 0; i < bufferSize; ++i) {
        input[i] = static_cast<float>(i);
    }

    CudaBuffer<float> source(bufferSize);
    CudaBuffer<float> destination(128);

    source.copyFromHostToDevice(input.data(), bufferSize);

    float* originalPtr = source.data();

    destination = std::move(source);

    EXPECT_EQ(source.data(), nullptr);
    EXPECT_EQ(source.size(), 0U);

    EXPECT_EQ(destination.data(), originalPtr);
    EXPECT_EQ(destination.size(), bufferSize);

    destination.copyFromDeviceToHost(output.data(), bufferSize);

    EXPECT_EQ(output, input);
}

TEST(CudaBufferTest, ReallocatesBuffer) {
    constexpr std::size_t initialBufferSize = 128;
    constexpr std::size_t newBufferSize = 8192;

    CudaBuffer<float> buffer(initialBufferSize);

    buffer.allocate(newBufferSize);

    EXPECT_EQ(buffer.size(), newBufferSize);
    ASSERT_NE(buffer.data(), nullptr);

    constexpr int blockSize = 256;

    const int gridSize = static_cast<int>((newBufferSize + blockSize - 1) / blockSize);

    fillWithIndex<<<gridSize, blockSize>>>(buffer.data(), newBufferSize);

    ASSERT_TRUE(cudaKernelCompletedSuccessfully());

    std::vector<float> output(newBufferSize);

    buffer.copyFromDeviceToHost(output.data(), newBufferSize);

    for (std::size_t i = 0; i < newBufferSize; i++)
        EXPECT_FLOAT_EQ(output[i], static_cast<float>(i)) << "at index " << i;
}

TEST(CudaBufferTest, ReusesBufferAfterZeroSizedAllocation) {
    CudaBuffer<float> buffer(1024);

    buffer.allocate(0);

    EXPECT_EQ(buffer.data(), nullptr);
    EXPECT_EQ(buffer.size(), 0U);

    constexpr std::size_t newBufferSize = 2048;

    buffer.allocate(newBufferSize);

    ASSERT_NE(buffer.data(), nullptr);
    EXPECT_EQ(buffer.size(), newBufferSize);

    std::vector<float> input(newBufferSize, 42.0f);
    std::vector<float> output(newBufferSize);

    buffer.copyFromHostToDevice(input.data(), newBufferSize);
    buffer.copyFromDeviceToHost(output.data(), newBufferSize);

    EXPECT_EQ(output, input);
}

TEST(CudaBufferTest, CopiesFloat4Data) {
    constexpr std::size_t bufferSize = 256;

    CudaBuffer<float4> buffer(bufferSize);

    std::vector<float4> input(bufferSize);
    std::vector<float4> output(bufferSize);

    for (std::size_t i = 0; i < bufferSize; ++i) {
        input[i] = make_float4(
            static_cast<float>(i),
            static_cast<float>(i + 1),
            static_cast<float>(i + 2),
            1.0f
        );
    }

    buffer.copyFromHostToDevice(input.data(), bufferSize);
    buffer.copyFromDeviceToHost(output.data(), bufferSize);

    for (std::size_t i = 0; i < bufferSize; ++i) {
        SCOPED_TRACE(::testing::Message() << "at index " << i);
        EXPECT_FLOAT_EQ(output[i].x, input[i].x);
        EXPECT_FLOAT_EQ(output[i].y, input[i].y);
        EXPECT_FLOAT_EQ(output[i].z, input[i].z);
        EXPECT_FLOAT_EQ(output[i].w, input[i].w);
    }
}

TEST(CudaBufferTest, FillBytesZeroesEntireBuffer) {
    constexpr std::size_t bufferSize = 4097;

    CudaBuffer<float> buffer(bufferSize);
    std::vector<float> initial(bufferSize, 42.0f);
    std::vector<float> output(bufferSize);

    buffer.copyFromHostToDevice(initial.data(), bufferSize);

    float* originalPtr = buffer.data();
    buffer.fillBytes(0);

    ASSERT_TRUE(cudaKernelCompletedSuccessfully());

    EXPECT_EQ(buffer.data(), originalPtr);
    EXPECT_EQ(buffer.size(), bufferSize);

    buffer.copyFromDeviceToHost(output.data(), bufferSize);

    for (std::size_t i = 0; i < bufferSize; ++i)
        EXPECT_FLOAT_EQ(output[i], 0.0f) << "at index " << i;
}

TEST(CudaBufferTest, FillBytesSetsEveryIntegerToMinusOne) {
    constexpr std::size_t bufferSize = 1024;

    CudaBuffer<int> buffer(bufferSize);
    std::vector<int> output(bufferSize);

    buffer.fillBytes(-1);

    ASSERT_TRUE(cudaKernelCompletedSuccessfully());

    buffer.copyFromDeviceToHost(output.data(), bufferSize);

    for (std::size_t i = 0; i < bufferSize; ++i)
        EXPECT_EQ(output[i], -1) << "at index " << i;
}

TEST(CudaBufferTest, FillBytesUsesCudaByteFillSemantics) {
    constexpr std::size_t bufferSize = 256;
    constexpr std::uint32_t byteValue = 0xABU;
    constexpr std::uint32_t expected = 0xABABABABU;

    CudaBuffer<std::uint32_t> buffer(bufferSize);
    std::vector<std::uint32_t> output(bufferSize);

    buffer.fillBytes(byteValue);

    ASSERT_TRUE(cudaKernelCompletedSuccessfully());

    buffer.copyFromDeviceToHost(output.data(), bufferSize);

    for (std::size_t i = 0; i < bufferSize; ++i)
        EXPECT_EQ(output[i], expected) << "at index " << i;
}

TEST(CudaBufferTest, FillBytesWorksAfterReallocation) {
    CudaBuffer<int> buffer(16);
    buffer.fillBytes(-1);

    constexpr std::size_t newBufferSize = 2048;
    buffer.allocate(newBufferSize);
    buffer.fillBytes(0);

    ASSERT_TRUE(cudaKernelCompletedSuccessfully());

    std::vector<int> output(newBufferSize);
    buffer.copyFromDeviceToHost(output.data(), newBufferSize);

    EXPECT_EQ(output, std::vector<int>(newBufferSize, 0));
}

TEST(CudaBufferTest, FillBytesOnEmptyBufferDoesNothing) {
    CudaBuffer<int> buffer;

    EXPECT_NO_THROW(buffer.fillBytes(-1));

    buffer.allocate(0);

    EXPECT_NO_THROW(buffer.fillBytes(0));
    EXPECT_EQ(buffer.data(), nullptr);
    EXPECT_EQ(buffer.size(), 0U);
}

TEST(CudaBufferTest, ThrowsWhenCopyExceedsBufferSize) {
    CudaBuffer<float> buffer(100);

    std::vector<float> data(200);

    EXPECT_THROW(
        buffer.copyFromHostToDevice(data.data(), data.size()),
        std::out_of_range
    );
}

TEST(CudaBufferTest, ThrowsWhenHostSourceIsNull) {
    CudaBuffer<float> buffer(100);

    EXPECT_THROW(
        buffer.copyFromHostToDevice(nullptr, 100),
        std::invalid_argument
    );
}

TEST(CudaBufferTest, ThrowsWhenHostDestinationIsNull) {
    CudaBuffer<float> buffer(100);

    EXPECT_THROW(
        buffer.copyFromDeviceToHost(nullptr, 100),
        std::invalid_argument
    );
}

TEST(CudaBufferTest, OverflowReallocationPreservesAllocationAndContents) {
    CudaBuffer<float4> buffer(1);
    const float4 original = make_float4(1,2,3,4);
    buffer.copyFromHostToDevice(&original,1);
    auto* pointer = buffer.data();
    EXPECT_THROW(buffer.allocate(std::numeric_limits<std::size_t>::max()/sizeof(float4)+1),
        std::length_error);
    EXPECT_EQ(buffer.data(),pointer);
    EXPECT_EQ(buffer.size(),1U);
    float4 actual{};
    buffer.copyFromDeviceToHost(&actual,1);
    EXPECT_EQ(actual.x,original.x);
    EXPECT_EQ(actual.w,original.w);
}

TEST(CudaBufferTest, SelfMoveAndZeroLengthCopiesAreSafe) {
    CudaBuffer<int> buffer(1);
    const int input=42;
    buffer.copyFromHostToDevice(&input,1);
    auto* alias=&buffer;
    buffer=std::move(*alias);
    int actual=0;
    buffer.copyFromDeviceToHost(&actual,1);
    EXPECT_EQ(actual,input);
    CudaBuffer<int> empty;
    EXPECT_NO_THROW(empty.copyFromHostToDevice(nullptr,0));
    EXPECT_NO_THROW(empty.copyFromDeviceToHost(nullptr,0));
}
