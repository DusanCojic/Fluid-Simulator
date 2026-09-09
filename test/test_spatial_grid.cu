#include "pbf/cuda_buffer.hpp"
#include "pbf/spatial_grid.cuh"

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

struct GridResult {
    std::vector<std::uint32_t> sortedKeys;
    std::vector<std::uint32_t> sortedIndices;
    std::vector<int> cellStart;
    std::vector<int> cellEnd;
};

TEST(SpatialGridInitializationTest, FailedReinitializationPreservesConfiguration) {
    SpatialGrid grid;
    grid.initialize(2, make_float3(0, 0, 0), make_float3(2, 2, 2), 1.0f);
    EXPECT_THROW(grid.initialize(100, make_float3(0, 0, 0),
        make_float3(1e20f, 2, 2), 1.0f), std::length_error);
    EXPECT_EQ(grid.maxParticles(), 2U);
    EXPECT_EQ(grid.gridSize().x, 2);
    EXPECT_FLOAT_EQ(grid.maxBounds().x, 2.0f);
}

float4 position(float x, float y, float z) {
    return make_float4(x, y, z, 1.0f);
}

template<typename T>
std::vector<T> copyFromDevice(const T* source, std::size_t count) {
    std::vector<T> result(count);

    const cudaError_t error = cudaMemcpy(
        result.data(),
        source,
        count * sizeof(T),
        cudaMemcpyDeviceToHost
    );

    if (error != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(error));

    return result;
}

GridResult buildAndRead(SpatialGrid& grid, const std::vector<float4>& positions) {
    CudaBuffer<float4> devicePositions(positions.size());
    devicePositions.copyFromHostToDevice(positions.data(), positions.size());

    grid.build(devicePositions.data(), positions.size());

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(error));

    error = cudaDeviceSynchronize();
    if (error != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(error));

    return {
        copyFromDevice(grid.sortedKeys(), positions.size()),
        copyFromDevice(grid.sortedIndices(), positions.size()),
        copyFromDevice(grid.cellStart(), grid.numCells()),
        copyFromDevice(grid.cellEnd(), grid.numCells())
    };
}

std::uint32_t referenceKey(
    const float4& particlePosition,
    const SpatialGrid& grid
) {
    const float3 minBounds = grid.minBounds();
    const float cellSize = grid.cellSize();
    const int3 gridSize = grid.gridSize();

    int x = static_cast<int>(
        std::floor((particlePosition.x - minBounds.x) / cellSize)
    );
    int y = static_cast<int>(
        std::floor((particlePosition.y - minBounds.y) / cellSize)
    );
    int z = static_cast<int>(
        std::floor((particlePosition.z - minBounds.z) / cellSize)
    );

    x = std::clamp(x, 0, gridSize.x - 1);
    y = std::clamp(y, 0, gridSize.y - 1);
    z = std::clamp(z, 0, gridSize.z - 1);

    return static_cast<std::uint32_t>(
        x + y * gridSize.x + z * gridSize.x * gridSize.y
    );
}

GridResult buildReference(
    const SpatialGrid& grid,
    const std::vector<float4>& positions
) {
    GridResult result;
    result.sortedIndices.resize(positions.size());
    std::iota(result.sortedIndices.begin(), result.sortedIndices.end(), 0U);

    std::vector<std::uint32_t> particleKeys(positions.size());
    for (std::size_t i = 0; i < positions.size(); ++i)
        particleKeys[i] = referenceKey(positions[i], grid);

    std::stable_sort(
        result.sortedIndices.begin(),
        result.sortedIndices.end(),
        [&particleKeys](std::uint32_t left, std::uint32_t right) {
            return particleKeys[left] < particleKeys[right];
        }
    );

    result.sortedKeys.resize(positions.size());
    for (std::size_t i = 0; i < positions.size(); ++i)
        result.sortedKeys[i] = particleKeys[result.sortedIndices[i]];

    result.cellStart.assign(grid.numCells(), -1);
    result.cellEnd.assign(grid.numCells(), -1);

    for (std::size_t i = 0; i < result.sortedKeys.size(); ++i) {
        const std::uint32_t key = result.sortedKeys[i];

        if (i == 0 || result.sortedKeys[i - 1] != key)
            result.cellStart[key] = static_cast<int>(i);

        if (i + 1 == result.sortedKeys.size() || result.sortedKeys[i + 1] != key)
            result.cellEnd[key] = static_cast<int>(i + 1);
    }

    return result;
}

void expectResultsEqual(const GridResult& actual, const GridResult& expected) {
    EXPECT_EQ(actual.sortedKeys, expected.sortedKeys);
    EXPECT_EQ(actual.sortedIndices, expected.sortedIndices);
    EXPECT_EQ(actual.cellStart, expected.cellStart);
    EXPECT_EQ(actual.cellEnd, expected.cellEnd);
}

void expectParticleKeyAssociations(
    const GridResult& result,
    const std::vector<float4>& positions,
    const SpatialGrid& grid
) {
    ASSERT_EQ(result.sortedKeys.size(), positions.size());
    ASSERT_EQ(result.sortedIndices.size(), positions.size());

    for (std::size_t sortedIndex = 0;
         sortedIndex < result.sortedIndices.size();
         ++sortedIndex) {
        const std::uint32_t particleIndex = result.sortedIndices[sortedIndex];
        ASSERT_LT(particleIndex, positions.size());

        EXPECT_EQ(
            result.sortedKeys[sortedIndex],
            referenceKey(positions[particleIndex], grid)
        ) << "at sorted index " << sortedIndex;
    }
}

class SpatialGridTest : public ::testing::Test {
protected:
    static constexpr std::size_t maxParticles = 32;
    const float3 minBounds = make_float3(0.0f, 0.0f, 0.0f);
    const float3 maxBounds = make_float3(4.0f, 3.0f, 2.0f);
    static constexpr float cellSize = 1.0f;

    SpatialGrid initializedGrid(std::size_t capacity = maxParticles) const {
        SpatialGrid grid;
        grid.initialize(capacity, minBounds, maxBounds, cellSize);
        return grid;
    }

};

TEST(SpatialGridInitializationTest, CalculatesExactGridDimensions) {
    SpatialGrid grid;
    const float3 minBounds = make_float3(-2.0f, -4.0f, 1.0f);
    const float3 maxBounds = make_float3(4.0f, 2.0f, 7.0f);

    grid.initialize(10, minBounds, maxBounds, 2.0f);

    const int3 size = grid.gridSize();
    EXPECT_EQ(size.x, 3);
    EXPECT_EQ(size.y, 3);
    EXPECT_EQ(size.z, 3);
    EXPECT_EQ(grid.numCells(), 27U);
    EXPECT_EQ(grid.maxParticles(), 10U);
    EXPECT_FLOAT_EQ(grid.cellSize(), 2.0f);
    EXPECT_FLOAT_EQ(grid.minBounds().x, minBounds.x);
    EXPECT_FLOAT_EQ(grid.maxBounds().z, maxBounds.z);
}

TEST(SpatialGridInitializationTest, CeilsNonIntegralGridDimensions) {
    SpatialGrid grid;

    grid.initialize(
        10,
        make_float3(-1.0f, 2.0f, 0.0f),
        make_float3(4.1f, 7.9f, 2.01f),
        2.0f
    );

    const int3 size = grid.gridSize();
    EXPECT_EQ(size.x, 3);
    EXPECT_EQ(size.y, 3);
    EXPECT_EQ(size.z, 2);
    EXPECT_EQ(grid.numCells(), 18U);
}

TEST(SpatialGridInitializationTest, RejectsZeroMaxParticles) {
    SpatialGrid grid;

    EXPECT_THROW(
        grid.initialize(
            0,
            make_float3(0.0f, 0.0f, 0.0f),
            make_float3(1.0f, 1.0f, 1.0f),
            1.0f
        ),
        std::invalid_argument
    );
}

TEST(SpatialGridInitializationTest, RejectsZeroCellSize) {
    SpatialGrid grid;

    EXPECT_THROW(
        grid.initialize(
            1,
            make_float3(0.0f, 0.0f, 0.0f),
            make_float3(1.0f, 1.0f, 1.0f),
            0.0f
        ),
        std::invalid_argument
    );
}

TEST(SpatialGridInitializationTest, RejectsNegativeCellSize) {
    SpatialGrid grid;

    EXPECT_THROW(
        grid.initialize(
            1,
            make_float3(0.0f, 0.0f, 0.0f),
            make_float3(1.0f, 1.0f, 1.0f),
            -0.5f
        ),
        std::invalid_argument
    );
}

TEST(SpatialGridInitializationTest, RejectsNonFiniteBoundsAndCellSize) {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    SpatialGrid grid;

    EXPECT_THROW(
        grid.initialize(
            1, make_float3(nan, 0.0f, 0.0f),
            make_float3(1.0f, 1.0f, 1.0f), 1.0f
        ),
        std::invalid_argument
    );
    EXPECT_THROW(
        grid.initialize(
            1, make_float3(0.0f, 0.0f, 0.0f),
            make_float3(1.0f, 1.0f, 1.0f), nan
        ),
        std::invalid_argument
    );
}

TEST(SpatialGridInitializationTest, RejectsGridLargerThanKeyCapacity) {
    SpatialGrid grid;

    EXPECT_THROW(
        grid.initialize(
            1, make_float3(0.0f, 0.0f, 0.0f),
            make_float3(65536.0f, 65536.0f, 2.0f), 1.0f
        ),
        std::length_error
    );
}

TEST(SpatialGridInitializationTest, RejectsInvalidBoundsOnEachAxis) {
    SpatialGrid grid;
    const float3 validMin = make_float3(0.0f, 0.0f, 0.0f);

    EXPECT_THROW(
        grid.initialize(
            1, validMin, make_float3(0.0f, 2.0f, 2.0f), 1.0f
        ),
        std::invalid_argument
    ) << "X bounds";

    EXPECT_THROW(
        grid.initialize(
            1, validMin, make_float3(2.0f, -1.0f, 2.0f), 1.0f
        ),
        std::invalid_argument
    ) << "Y bounds";

    EXPECT_THROW(
        grid.initialize(
            1, validMin, make_float3(2.0f, 2.0f, 0.0f), 1.0f
        ),
        std::invalid_argument
    ) << "Z bounds";
}

TEST_F(SpatialGridTest, BuildsSingleParticle) {
    SpatialGrid grid = initializedGrid();
    const std::vector<float4> positions = {position(1.2f, 0.4f, 1.8f)};

    const GridResult actual = buildAndRead(grid, positions);

    expectResultsEqual(actual, buildReference(grid, positions));
}

TEST_F(SpatialGridTest, EmptyBuildClearsAllCellRanges) {
    SpatialGrid grid = initializedGrid();
    const std::vector<float4> positions = {position(1.2f, 0.4f, 1.8f)};

    const GridResult populated = buildAndRead(grid, positions);
    ASSERT_NE(populated.cellStart[13], -1);

    const GridResult empty = buildAndRead(grid, {});

    EXPECT_TRUE(empty.sortedKeys.empty());
    EXPECT_TRUE(empty.sortedIndices.empty());
    EXPECT_EQ(empty.cellStart, std::vector<int>(grid.numCells(), -1));
    EXPECT_EQ(empty.cellEnd, std::vector<int>(grid.numCells(), -1));
}

TEST_F(SpatialGridTest, BuildsParticlesInDifferentCells) {
    SpatialGrid grid = initializedGrid();
    const std::vector<float4> positions = {
        position(0.2f, 0.2f, 0.2f),
        position(1.2f, 0.2f, 0.2f),
        position(2.2f, 1.2f, 0.2f),
        position(3.2f, 2.2f, 1.2f)
    };

    const GridResult actual = buildAndRead(grid, positions);

    expectResultsEqual(actual, buildReference(grid, positions));
}

TEST_F(SpatialGridTest, BuildsParticlesInSameCell) {
    SpatialGrid grid = initializedGrid();
    const std::vector<float4> positions = {
        position(2.1f, 1.1f, 0.1f),
        position(2.4f, 1.8f, 0.3f),
        position(2.9f, 1.2f, 0.9f),
        position(2.5f, 1.5f, 0.5f)
    };

    const GridResult actual = buildAndRead(grid, positions);

    expectResultsEqual(actual, buildReference(grid, positions));
}

TEST_F(SpatialGridTest, BuildsParticlesAcrossSeveralCells) {
    SpatialGrid grid = initializedGrid();
    const std::vector<float4> positions = {
        position(0.2f, 0.2f, 0.2f),
        position(0.8f, 0.7f, 0.4f),
        position(2.2f, 1.2f, 0.2f),
        position(3.8f, 2.8f, 1.8f),
        position(2.6f, 1.7f, 0.9f),
        position(3.1f, 2.1f, 1.1f)
    };

    const GridResult actual = buildAndRead(grid, positions);

    expectResultsEqual(actual, buildReference(grid, positions));
}

TEST_F(SpatialGridTest, SortsParticlesGivenInUnsortedSpatialOrder) {
    SpatialGrid grid = initializedGrid();
    const std::vector<float4> positions = {
        position(3.2f, 2.2f, 1.2f),
        position(1.2f, 0.2f, 0.2f),
        position(3.8f, 2.8f, 1.8f),
        position(0.2f, 0.2f, 0.2f),
        position(2.2f, 1.2f, 0.2f)
    };

    const GridResult actual = buildAndRead(grid, positions);

    expectResultsEqual(actual, buildReference(grid, positions));
    EXPECT_NE(actual.sortedIndices, std::vector<std::uint32_t>({0, 1, 2, 3, 4}));
}

TEST_F(SpatialGridTest, ProducesAscendingSortedKeys) {
    SpatialGrid grid = initializedGrid();
    const std::vector<float4> positions = {
        position(3.9f, 2.9f, 1.9f),
        position(0.1f, 2.1f, 0.1f),
        position(2.1f, 0.1f, 1.1f),
        position(0.1f, 0.1f, 0.1f),
        position(1.1f, 1.1f, 0.1f)
    };

    const GridResult actual = buildAndRead(grid, positions);

    EXPECT_TRUE(std::is_sorted(actual.sortedKeys.begin(), actual.sortedKeys.end()));
}

TEST_F(SpatialGridTest, PreservesParticleToKeyAssociationAfterSorting) {
    SpatialGrid grid = initializedGrid();
    const std::vector<float4> positions = {
        position(3.2f, 1.2f, 0.2f),
        position(0.4f, 2.4f, 1.4f),
        position(1.7f, 0.7f, 0.7f),
        position(2.3f, 2.3f, 1.3f),
        position(0.6f, 0.6f, 0.6f)
    };

    const GridResult actual = buildAndRead(grid, positions);

    expectParticleKeyAssociations(actual, positions, grid);
}

TEST_F(SpatialGridTest, CellRangesUseHalfOpenIntervals) {
    SpatialGrid grid = initializedGrid();
    const std::vector<float4> positions = {
        position(0.1f, 0.1f, 0.1f),
        position(0.2f, 0.2f, 0.2f),
        position(0.3f, 0.3f, 0.3f),
        position(2.1f, 0.1f, 0.1f),
        position(2.2f, 0.2f, 0.2f)
    };

    const GridResult actual = buildAndRead(grid, positions);

    EXPECT_EQ(actual.cellStart[0], 0);
    EXPECT_EQ(actual.cellEnd[0], 3);
    EXPECT_EQ(actual.cellStart[2], 3);
    EXPECT_EQ(actual.cellEnd[2], 5);

    for (const std::uint32_t key : {0U, 2U}) {
        for (int i = actual.cellStart[key]; i < actual.cellEnd[key]; ++i)
            EXPECT_EQ(actual.sortedKeys[static_cast<std::size_t>(i)], key);
    }
}

TEST_F(SpatialGridTest, LeavesEmptyCellsMarkedWithMinusOne) {
    SpatialGrid grid = initializedGrid();
    const std::vector<float4> positions = {
        position(0.1f, 0.1f, 0.1f),
        position(2.1f, 1.1f, 1.1f)
    };

    const GridResult actual = buildAndRead(grid, positions);

    for (std::size_t cell = 0; cell < grid.numCells(); ++cell) {
        if (cell == 0 || cell == 18)
            continue;

        EXPECT_EQ(actual.cellStart[cell], -1) << "cell " << cell;
        EXPECT_EQ(actual.cellEnd[cell], -1) << "cell " << cell;
    }
}

TEST_F(SpatialGridTest, MapsParticleAtMinBoundsToFirstCell) {
    SpatialGrid grid = initializedGrid();
    const std::vector<float4> positions = {position(0.0f, 0.0f, 0.0f)};

    const GridResult actual = buildAndRead(grid, positions);

    ASSERT_EQ(actual.sortedKeys.size(), 1U);
    EXPECT_EQ(actual.sortedKeys[0], 0U);
    EXPECT_EQ(actual.sortedIndices[0], 0U);
}

TEST_F(SpatialGridTest, MapsParticleNearMaxBoundsToLastCell) {
    SpatialGrid grid = initializedGrid();
    const std::vector<float4> positions = {position(3.999f, 2.999f, 1.999f)};

    const GridResult actual = buildAndRead(grid, positions);

    ASSERT_EQ(actual.sortedKeys.size(), 1U);
    EXPECT_EQ(actual.sortedKeys[0], 23U);
}

TEST_F(SpatialGridTest, ClampsParticleBelowMinBounds) {
    SpatialGrid grid = initializedGrid();
    const std::vector<float4> positions = {position(-5.0f, 1.2f, 0.2f)};

    const GridResult actual = buildAndRead(grid, positions);

    ASSERT_EQ(actual.sortedKeys.size(), 1U);
    EXPECT_EQ(actual.sortedKeys[0], 4U);
}

TEST_F(SpatialGridTest, ClampsParticleAboveMaxBounds) {
    SpatialGrid grid = initializedGrid();
    const std::vector<float4> positions = {position(10.0f, 1.2f, 0.2f)};

    const GridResult actual = buildAndRead(grid, positions);

    ASSERT_EQ(actual.sortedKeys.size(), 1U);
    EXPECT_EQ(actual.sortedKeys[0], 7U);
}

TEST_F(SpatialGridTest, ClampsParticlesOutsideMultipleAxes) {
    SpatialGrid grid = initializedGrid();
    const std::vector<float4> positions = {
        position(-2.0f, 8.0f, -4.0f),
        position(9.0f, -3.0f, 7.0f)
    };

    const GridResult actual = buildAndRead(grid, positions);

    EXPECT_EQ(actual.sortedKeys, std::vector<std::uint32_t>({8U, 15U}));
    expectParticleKeyAssociations(actual, positions, grid);
}

TEST_F(SpatialGridTest, HandlesParticlesInFirstAndLastCells) {
    SpatialGrid grid = initializedGrid();
    const std::vector<float4> positions = {
        position(3.5f, 2.5f, 1.5f),
        position(0.5f, 0.5f, 0.5f)
    };

    const GridResult actual = buildAndRead(grid, positions);

    EXPECT_EQ(actual.sortedKeys, std::vector<std::uint32_t>({0U, 23U}));
    EXPECT_EQ(actual.sortedIndices, std::vector<std::uint32_t>({1U, 0U}));
    EXPECT_EQ(actual.cellStart[0], 0);
    EXPECT_EQ(actual.cellEnd[0], 1);
    EXPECT_EQ(actual.cellStart[23], 1);
    EXPECT_EQ(actual.cellEnd[23], 2);
}

TEST_F(SpatialGridTest, ClampsMultipleParticlesIntoSameBoundaryCell) {
    SpatialGrid grid = initializedGrid();
    const std::vector<float4> positions = {
        position(-1.0f, -1.0f, -1.0f),
        position(-10.0f, -2.0f, -3.0f),
        position(-0.01f, -0.01f, -0.01f)
    };

    const GridResult actual = buildAndRead(grid, positions);

    EXPECT_EQ(actual.sortedKeys, std::vector<std::uint32_t>({0U, 0U, 0U}));
    EXPECT_EQ(actual.cellStart[0], 0);
    EXPECT_EQ(actual.cellEnd[0], 3);
}

TEST_F(SpatialGridTest, AcceptsParticleCountSmallerThanCapacity) {
    SpatialGrid grid = initializedGrid(8);
    const std::vector<float4> positions = {
        position(0.2f, 0.2f, 0.2f),
        position(1.2f, 1.2f, 1.2f),
        position(3.2f, 2.2f, 1.2f)
    };

    const GridResult actual = buildAndRead(grid, positions);

    EXPECT_EQ(actual.sortedKeys.size(), positions.size());
    expectResultsEqual(actual, buildReference(grid, positions));
}

TEST_F(SpatialGridTest, AcceptsParticleCountEqualToCapacity) {
    SpatialGrid grid = initializedGrid(4);
    const std::vector<float4> positions = {
        position(3.2f, 2.2f, 1.2f),
        position(0.2f, 0.2f, 0.2f),
        position(2.2f, 1.2f, 0.2f),
        position(1.2f, 0.2f, 0.2f)
    };

    const GridResult actual = buildAndRead(grid, positions);

    EXPECT_EQ(actual.sortedKeys.size(), grid.maxParticles());
    expectResultsEqual(actual, buildReference(grid, positions));
}

TEST_F(SpatialGridTest, RejectsParticleCountGreaterThanCapacity) {
    SpatialGrid grid = initializedGrid(3);
    CudaBuffer<float4> devicePositions(4);

    EXPECT_THROW(grid.build(devicePositions.data(), 4), std::out_of_range);
}

TEST_F(SpatialGridTest, SupportsRepeatedBuildCallsWithNewPositions) {
    SpatialGrid grid = initializedGrid(6);
    const std::vector<float4> firstPositions = {
        position(0.1f, 0.1f, 0.1f),
        position(0.2f, 0.2f, 0.2f),
        position(3.1f, 2.1f, 1.1f)
    };
    const std::vector<float4> secondPositions = {
        position(2.1f, 0.1f, 0.1f),
        position(1.1f, 2.1f, 1.1f),
        position(2.2f, 0.2f, 0.2f),
        position(1.2f, 2.2f, 1.2f)
    };

    const GridResult first = buildAndRead(grid, firstPositions);
    const GridResult second = buildAndRead(grid, secondPositions);

    expectResultsEqual(first, buildReference(grid, firstPositions));
    expectResultsEqual(second, buildReference(grid, secondPositions));
}

TEST_F(SpatialGridTest, ClearsOldCellRangesBetweenBuilds) {
    SpatialGrid grid = initializedGrid(6);
    const std::vector<float4> firstPositions = {
        position(0.1f, 0.1f, 0.1f),
        position(1.1f, 1.1f, 0.1f),
        position(3.1f, 2.1f, 1.1f)
    };
    const std::vector<float4> secondPositions = {
        position(2.1f, 0.1f, 0.1f),
        position(2.2f, 0.2f, 0.2f)
    };

    const GridResult first = buildAndRead(grid, firstPositions);
    ASSERT_NE(first.cellStart[0], -1);
    ASSERT_NE(first.cellStart[5], -1);
    ASSERT_NE(first.cellStart[23], -1);

    const GridResult second = buildAndRead(grid, secondPositions);

    EXPECT_EQ(second.cellStart[0], -1);
    EXPECT_EQ(second.cellEnd[0], -1);
    EXPECT_EQ(second.cellStart[5], -1);
    EXPECT_EQ(second.cellEnd[5], -1);
    EXPECT_EQ(second.cellStart[23], -1);
    EXPECT_EQ(second.cellEnd[23], -1);
    EXPECT_EQ(second.cellStart[2], 0);
    EXPECT_EQ(second.cellEnd[2], 2);
}

TEST(SpatialGridRandomizedTest, MatchesCpuReferenceForLargeParticleSet) {
    constexpr std::size_t particleCount = 8193;
    const float3 minBounds = make_float3(-8.0f, -5.0f, -3.0f);
    const float3 maxBounds = make_float3(9.0f, 7.0f, 6.0f);
    constexpr float cellSize = 0.75f;

    SpatialGrid grid;
    grid.initialize(particleCount, minBounds, maxBounds, cellSize);

    std::mt19937 randomEngine(0x5A17U);
    std::uniform_real_distribution<float> xDistribution(-12.0f, 13.0f);
    std::uniform_real_distribution<float> yDistribution(-9.0f, 11.0f);
    std::uniform_real_distribution<float> zDistribution(-7.0f, 10.0f);

    std::vector<float4> positions;
    positions.reserve(particleCount);

    for (std::size_t i = 0; i < particleCount; ++i) {
        positions.push_back(position(
            xDistribution(randomEngine),
            yDistribution(randomEngine),
            zDistribution(randomEngine)
        ));
    }

    const GridResult actual = buildAndRead(grid, positions);
    const GridResult expected = buildReference(grid, positions);

    expectResultsEqual(actual, expected);
    EXPECT_TRUE(std::is_sorted(actual.sortedKeys.begin(), actual.sortedKeys.end()));
    expectParticleKeyAssociations(actual, positions, grid);
}

TEST_F(SpatialGridTest, ClampsFinitePositionsBeyondIntegerCoordinateRange) {
    auto grid = initializedGrid();
    const auto result = buildAndRead(grid, {
        position(1e30f, 1e30f, 1e30f), position(-1e30f, -1e30f, -1e30f)
    });
    EXPECT_EQ(result.sortedKeys, (std::vector<std::uint32_t>{0, 23}));
    EXPECT_EQ(result.sortedIndices, (std::vector<std::uint32_t>{1, 0}));
}

TEST(SpatialGridInitializationTest, RejectsNullBuildInputBeforeLaunching) {
    SpatialGrid grid;
    grid.initialize(2, make_float3(0, 0, 0), make_float3(2, 2, 2), 1.0f);
    EXPECT_THROW(grid.build(nullptr, 1), std::invalid_argument);
}
