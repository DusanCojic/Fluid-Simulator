#include "pbf/cuda_buffer.hpp"
#include "pbf/neighbors.cuh"
#include "pbf/spatial_grid.cuh"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

float4 position(float x, float y, float z) {
    return make_float4(x, y, z, 1.0f);
}

struct NeighborResult {
    std::vector<int> counts;
    std::vector<std::vector<std::uint32_t>> neighbors;
};

NeighborResult runNeighborSearch(
    const std::vector<float4>& positions,
    float smoothingRadius,
    float cellSize = 1.0f,
    int maxNeighbors = 256
) {
    SpatialGrid grid;
    grid.initialize(
        positions.size(),
        make_float3(0.0f, 0.0f, 0.0f),
        make_float3(5.0f, 5.0f, 5.0f),
        cellSize
    );

    CudaBuffer<float4> devicePositions(positions.size());
    CudaBuffer<float4> deviceSortedPositions(positions.size());
    CudaBuffer<std::uint32_t> deviceNeighbors(
        positions.size() * static_cast<std::size_t>(maxNeighbors)
    );
    CudaBuffer<int> deviceCounts(positions.size());

    devicePositions.copyFromHostToDevice(positions.data(), positions.size());
    grid.build(devicePositions.data(), positions.size());

    std::vector<std::uint32_t> permutation(positions.size());
    cudaError_t error = cudaMemcpy(
        permutation.data(), grid.sortedIndices(),
        permutation.size() * sizeof(std::uint32_t), cudaMemcpyDeviceToHost
    );
    if (error != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(error));
    std::vector<float4> sortedPositions(positions.size());
    for (std::size_t i = 0; i < positions.size(); ++i)
        sortedPositions[i] = positions[permutation[i]];
    deviceSortedPositions.copyFromHostToDevice(
        sortedPositions.data(), sortedPositions.size()
    );

    constexpr int blockSize = 256;
    const int blockCount = static_cast<int>((positions.size() + blockSize - 1) / blockSize);

    findNeighbors<<<blockCount, blockSize>>>(
        deviceSortedPositions.data(),
        grid.cellStart(),
        grid.cellEnd(),
        grid.gridSize(),
        grid.minBounds(),
        grid.cellSize(),
        positions.size(),
        smoothingRadius,
        deviceNeighbors.data(),
        deviceCounts.data(),
        maxNeighbors,
        positions.size()
    );

    error = cudaGetLastError();
    if (error != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(error));

    error = cudaDeviceSynchronize();
    if (error != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(error));

    NeighborResult result;
    std::vector<int> sortedCounts(positions.size());
    deviceCounts.copyFromDeviceToHost(sortedCounts.data(), positions.size());

    std::vector<std::uint32_t> flatNeighbors(deviceNeighbors.size());
    deviceNeighbors.copyFromDeviceToHost(flatNeighbors.data(), flatNeighbors.size());

    result.counts.resize(positions.size());
    result.neighbors.resize(positions.size());
    for (std::size_t sortedParticle = 0; sortedParticle < positions.size();
         ++sortedParticle) {
        const std::size_t originalParticle = permutation[sortedParticle];
        result.counts[originalParticle] = sortedCounts[sortedParticle];
        const int storedCount = std::min(sortedCounts[sortedParticle], maxNeighbors);
        auto& output = result.neighbors[originalParticle];
        output.reserve(storedCount);
        for (int offset = 0; offset < storedCount; ++offset) {
            const std::uint32_t sortedNeighbor =
                flatNeighbors[offset * positions.size() + sortedParticle];
            output.push_back(permutation[sortedNeighbor]);
        }
    }

    return result;
}

std::vector<std::vector<std::uint32_t>> bruteForce(
    const std::vector<float4>& positions,
    float smoothingRadius
) {
    std::vector<std::vector<std::uint32_t>> expected(positions.size());
    const float radiusSquared = smoothingRadius * smoothingRadius;

    for (std::size_t i = 0; i < positions.size(); ++i) {
        for (std::size_t j = 0; j < positions.size(); ++j) {
            if (i == j)
                continue;

            const float dx = positions[i].x - positions[j].x;
            const float dy = positions[i].y - positions[j].y;
            const float dz = positions[i].z - positions[j].z;

            if (dx * dx + dy * dy + dz * dz <= radiusSquared)
                expected[i].push_back(static_cast<std::uint32_t>(j));
        }
    }

    return expected;
}

void expectCorrect(const NeighborResult& result, const std::vector<float4>& positions, float radius) {
    const auto expected = bruteForce(positions, radius);

    for (std::size_t i = 0; i < positions.size(); ++i) {
        EXPECT_EQ(result.counts[i], expected[i].size()) << "particle " << i;

        auto actual = result.neighbors[i];
        std::sort(actual.begin(), actual.end());
        EXPECT_EQ(actual, expected[i]) << "particle " << i;
    }
}

void expectSymmetric(const NeighborResult& result) {
    for (std::size_t i = 0; i < result.neighbors.size(); ++i) {
        for (const std::uint32_t neighbor : result.neighbors[i]) {
            const auto& reverse = result.neighbors[neighbor];
            EXPECT_NE(
                std::find(reverse.begin(), reverse.end(), static_cast<std::uint32_t>(i)),
                reverse.end()
            );
        }
    }
}

std::vector<float4> lattice(int sideLength, float spacing) {
    std::vector<float4> positions;

    for (int z = 0; z < sideLength; ++z)
        for (int y = 0; y < sideLength; ++y)
            for (int x = 0; x < sideLength; ++x)
                positions.push_back(position(
                    1.0f + x * spacing,
                    1.0f + y * spacing,
                    1.0f + z * spacing
                ));

    return positions;
}

TEST(NeighborSearchTest, SingleParticleHasNoNeighbors) {
    const std::vector<float4> positions = {position(2.0f, 2.0f, 2.0f)};
    const NeighborResult result = runNeighborSearch(positions, 0.75f);

    EXPECT_EQ(result.counts[0], 0);
}

TEST(NeighborSearchTest, TwoParticlesWithinRadiusFindEachOther) {
    const std::vector<float4> positions = {
        position(1.0f, 1.0f, 1.0f),
        position(1.5f, 1.0f, 1.0f)
    };
    const NeighborResult result = runNeighborSearch(positions, 0.75f);

    EXPECT_EQ(result.neighbors[0], std::vector<std::uint32_t>({1}));
    EXPECT_EQ(result.neighbors[1], std::vector<std::uint32_t>({0}));
}

TEST(NeighborSearchTest, TwoParticlesOutsideRadiusAreNotNeighbors) {
    const std::vector<float4> positions = {
        position(1.0f, 1.0f, 1.0f),
        position(2.0f, 1.0f, 1.0f)
    };
    const NeighborResult result = runNeighborSearch(positions, 0.75f);

    EXPECT_EQ(result.counts, std::vector<int>({0, 0}));
}

TEST(NeighborSearchTest, ParticlesExactlyAtRadiusAreIncluded) {
    const std::vector<float4> positions = {
        position(1.0f, 1.0f, 1.0f),
        position(1.5f, 1.0f, 1.0f)
    };
    const NeighborResult result = runNeighborSearch(positions, 0.5f);

    EXPECT_EQ(result.counts, std::vector<int>({1, 1}));
}

TEST(NeighborSearchTest, DistinguishesJustBelowAndAboveRadius) {
    constexpr float radius = 0.5f;
    const float below = std::nextafter(radius, 0.0f);
    const float above = std::nextafter(radius, std::numeric_limits<float>::infinity());

    const NeighborResult belowResult = runNeighborSearch(
        {position(0.0f, 1.0f, 1.0f), position(below, 1.0f, 1.0f)},
        radius
    );
    const NeighborResult aboveResult = runNeighborSearch(
        {position(0.0f, 1.0f, 1.0f), position(above, 1.0f, 1.0f)},
        radius
    );

    EXPECT_EQ(belowResult.counts, std::vector<int>({1, 1}));
    EXPECT_EQ(aboveResult.counts, std::vector<int>({0, 0}));
}

TEST(NeighborSearchTest, FindsParticlesInSameGridCell) {
    const std::vector<float4> positions = {
        position(2.1f, 2.2f, 2.3f),
        position(2.4f, 2.4f, 2.4f),
        position(2.7f, 2.5f, 2.3f)
    };
    const NeighborResult result = runNeighborSearch(positions, 0.75f);

    expectCorrect(result, positions, 0.75f);
}

TEST(NeighborSearchTest, FindsParticlesInAdjacentGridCells) {
    const std::vector<float4> positions = {
        position(0.9f, 2.0f, 2.0f),
        position(1.1f, 2.0f, 2.0f)
    };
    const NeighborResult result = runNeighborSearch(positions, 0.3f);

    EXPECT_EQ(result.counts, std::vector<int>({1, 1}));
}

TEST(NeighborSearchTest, FindsParticlesAcrossCellBoundary) {
    const std::vector<float4> positions = {
        position(0.9999f, 2.0f, 2.0f),
        position(1.0001f, 2.0f, 2.0f)
    };
    const NeighborResult result = runNeighborSearch(positions, 0.01f);

    EXPECT_EQ(result.counts, std::vector<int>({1, 1}));
}

TEST(NeighborSearchTest, HandlesGridDomainBoundaries) {
    const std::vector<float4> positions = {
        position(0.01f, 0.01f, 0.01f),
        position(0.11f, 0.01f, 0.01f),
        position(4.89f, 4.99f, 4.99f),
        position(4.99f, 4.99f, 4.99f)
    };
    const NeighborResult result = runNeighborSearch(positions, 0.15f);

    EXPECT_EQ(result.counts, std::vector<int>({1, 1, 1, 1}));
}

TEST(NeighborSearchTest, FindsExpectedNeighborsAlongLine) {
    std::vector<float4> positions;
    for (int i = 0; i < 8; ++i)
        positions.push_back(position(0.5f + i * 0.5f, 2.0f, 2.0f));

    const NeighborResult result = runNeighborSearch(positions, 1.01f);

    EXPECT_EQ(result.counts, std::vector<int>({2, 3, 4, 4, 4, 4, 3, 2}));
    expectCorrect(result, positions, 1.01f);
}

TEST(NeighborSearchTest, ThreeCubedLatticeHasExpectedCountsAndSymmetry) {
    const std::vector<float4> positions = lattice(3, 0.5f);
    const NeighborResult result = runNeighborSearch(positions, 0.9f, 0.5f);

    expectCorrect(result, positions, 0.9f);
    EXPECT_EQ(result.counts[0], 7);   // corner
    EXPECT_EQ(result.counts[1], 11);  // edge
    EXPECT_EQ(result.counts[4], 17);  // face
    EXPECT_EQ(result.counts[13], 26); // interior
    expectSymmetric(result);
}

TEST(NeighborSearchTest, FiveCubedLatticeHasExpectedCountsAndSymmetry) {
    const std::vector<float4> positions = lattice(5, 0.5f);
    const NeighborResult result = runNeighborSearch(positions, 0.9f, 0.5f);

    expectCorrect(result, positions, 0.9f);
    EXPECT_EQ(result.counts[0], 7);   // corner
    EXPECT_EQ(result.counts[2], 11);  // edge
    EXPECT_EQ(result.counts[12], 17); // face
    EXPECT_EQ(result.counts[62], 26); // interior
    expectSymmetric(result);
}

TEST(NeighborSearchTest, SparseParticlesHaveNoNeighbors) {
    const std::vector<float4> positions = {
        position(0.1f, 0.1f, 0.1f),
        position(1.1f, 3.7f, 2.2f),
        position(2.5f, 0.8f, 4.1f),
        position(3.9f, 2.4f, 0.7f),
        position(4.8f, 4.8f, 4.8f)
    };
    const NeighborResult result = runNeighborSearch(positions, 0.2f);

    EXPECT_EQ(result.counts, std::vector<int>(positions.size(), 0));
}

TEST(NeighborSearchTest, DenseClusterFindsAllParticles) {
    std::vector<float4> positions;
    for (int i = 0; i < 32; ++i)
        positions.push_back(position(2.0f + 0.001f * i, 2.0f, 2.0f));

    const NeighborResult result = runNeighborSearch(positions, 0.5f);

    EXPECT_EQ(result.counts, std::vector<int>(32, 31));
}

TEST(NeighborSearchTest, IdenticalPositionsAreValidAndExcludeSelf) {
    const std::vector<float4> positions(4, position(2.0f, 2.0f, 2.0f));
    const NeighborResult result = runNeighborSearch(positions, 0.75f);

    expectCorrect(result, positions, 0.75f);
    EXPECT_EQ(result.counts, std::vector<int>({3, 3, 3, 3}));
}

TEST(NeighborSearchTest, FixedSeedRandomParticlesMatchBruteForce) {
    constexpr std::size_t particleCount = 257;
    std::mt19937 randomEngine(12345);
    std::uniform_real_distribution<float> distribution(0.0f, 5.0f);
    std::vector<float4> positions;

    for (std::size_t i = 0; i < particleCount; ++i)
        positions.push_back(position(
            distribution(randomEngine),
            distribution(randomEngine),
            distribution(randomEngine)
        ));

    const NeighborResult result = runNeighborSearch(positions, 0.85f, 0.6f);

    expectCorrect(result, positions, 0.85f);
}

TEST(NeighborSearchTest, ReportsMaximumNeighborOverflow) {
    constexpr int particleCount = 12;
    constexpr int maxNeighbors = 8;
    const std::vector<float4> positions(
        particleCount,
        position(2.0f, 2.0f, 2.0f)
    );

    const NeighborResult result = runNeighborSearch(
        positions,
        0.75f,
        1.0f,
        maxNeighbors
    );

    for (int count : result.counts)
        EXPECT_EQ(count, particleCount - 1);
}

TEST(NeighborSearchTest, SupportRadiusBeyondIntegerRangeVisitsEachCellOnce) {
    const std::vector<float4> positions = {
        position(-0.1f, 0, 0), position(1, 1, 1), position(5.1f, 5.1f, 5.1f)
    };
    const auto result = runNeighborSearch(positions, 1e10f);
    expectCorrect(result, positions, 1e10f);
}

TEST(NeighborSearchTest, AllSixDirectionsAndDiagonalAcrossCellBoundaries) {
    const std::vector<float4> positions = {
        position(2.01f,2.01f,2.01f), position(1.99f,2.01f,2.01f),
        position(2.01f,1.99f,2.01f), position(2.01f,2.01f,1.99f),
        position(1.99f,1.99f,1.99f), position(5.01f,5.01f,5.01f),
        position(4.99f,4.99f,4.99f), position(-0.01f,-0.01f,-0.01f),
        position(0.01f,0.01f,0.01f)
    };
    const auto result = runNeighborSearch(positions,0.1f);
    expectCorrect(result,positions,0.1f);
    expectSymmetric(result);
}

TEST(NeighborSearchTest, WritesSlotMajorNeighbors) {
    const std::vector<float4> positions = {
        position(3.1f, 1.0f, 1.0f),
        position(0.1f, 1.0f, 1.0f),
        position(3.3f, 1.0f, 1.0f),
        position(0.3f, 1.0f, 1.0f),
        position(0.5f, 1.0f, 1.0f)
    };
    constexpr float radius = 0.45f;
    constexpr int maxNeighbors = 4;
    constexpr std::size_t particleStride = 8;

    SpatialGrid grid;
    grid.initialize(positions.size(), {0.0f, 0.0f, 0.0f},
                    {5.0f, 5.0f, 5.0f}, 1.0f);
    CudaBuffer<float4> inputPositions(positions.size());
    inputPositions.copyFromHostToDevice(positions.data(), positions.size());
    grid.build(inputPositions.data(), positions.size());

    std::vector<std::uint32_t> permutation(positions.size());
    ASSERT_EQ(cudaSuccess, cudaMemcpy(
        permutation.data(), grid.sortedIndices(),
        permutation.size() * sizeof(std::uint32_t), cudaMemcpyDeviceToHost
    ));
    std::vector<float4> sortedPositions(positions.size());
    for (std::size_t i = 0; i < positions.size(); ++i)
        sortedPositions[i] = positions[permutation[i]];

    CudaBuffer<float4> deviceSortedPositions(sortedPositions.size());
    CudaBuffer<std::uint32_t> deviceNeighbors(particleStride * maxNeighbors);
    CudaBuffer<int> deviceCounts(sortedPositions.size());
    deviceSortedPositions.copyFromHostToDevice(
        sortedPositions.data(), sortedPositions.size()
    );
    deviceNeighbors.fillBytes(-1);

    findNeighbors<<<1, 32>>>(
        deviceSortedPositions.data(), grid.cellStart(), grid.cellEnd(),
        grid.gridSize(), grid.minBounds(), grid.cellSize(), sortedPositions.size(),
        radius, deviceNeighbors.data(), deviceCounts.data(), maxNeighbors,
        particleStride
    );
    ASSERT_EQ(cudaSuccess, cudaGetLastError());
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

    std::vector<int> counts(sortedPositions.size());
    std::vector<std::uint32_t> flatNeighbors(deviceNeighbors.size());
    deviceCounts.copyFromDeviceToHost(counts.data(), counts.size());
    deviceNeighbors.copyFromDeviceToHost(flatNeighbors.data(), flatNeighbors.size());
    const auto expected = bruteForce(sortedPositions, radius);

    for (std::size_t particle = 0; particle < sortedPositions.size(); ++particle) {
        ASSERT_EQ(counts[particle], expected[particle].size());
        std::vector<std::uint32_t> actual;
        for (int offset = 0; offset < counts[particle]; ++offset)
            actual.push_back(flatNeighbors[offset * particleStride + particle]);
        std::sort(actual.begin(), actual.end());
        EXPECT_EQ(actual, expected[particle]) << "particle " << particle;
    }
}
