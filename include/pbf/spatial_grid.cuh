#ifndef SPATIAL_GRID_H
#define SPATIAL_GRID_H

#include "pbf/cuda_buffer.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

class SpatialGrid {
public:
    SpatialGrid() = default;

    void initialize(std::size_t maxParticles, float3 minBounds, float3 maxBounds, float cellSize);

    void build(const float4* predictedPositions, std::size_t particleCount);

    const std::uint32_t* sortedKeys() const;
    const std::uint32_t* sortedIndices() const;

    const int* cellStart() const;
    const int* cellEnd() const;

    int3 gridSize() const;

    std::size_t numCells() const;

    std::size_t maxParticles() const;

    float cellSize() const;

    float3 minBounds() const;
    float3 maxBounds() const;

private:
    // Unsorted particle grid data
    CudaBuffer<std::uint32_t> _keys;
    CudaBuffer<std::uint32_t> _indices;

    // Particle grid data after radix sort
    CudaBuffer<std::uint32_t> _sortedKeys;
    CudaBuffer<std::uint32_t> _sortedIndices;

    // Start and end range of particles for each grid cell
    CudaBuffer<int> _cellStart;
    CudaBuffer<int> _cellEnd;

    // Temporary memory required by CUB radix sort
    std::size_t _sortTempStorageBytes = 0;
    CudaBuffer<std::byte> _sortTempStorage;

    float3 _minBounds{};
    float3 _maxBounds{};

    int3 _gridSize{};

    std::size_t _numCells = 0;
    std::size_t _maxParticles = 0;

    float _cellSize = 0.0f;
};

// Grid helper functions
__device__ inline int3 positionToCell(float4 position, float3 minBounds, float cellSize) {
    int cellX = static_cast<int>(
        floorf((position.x - minBounds.x) / cellSize)
    );

    int cellY = static_cast<int>(
        floorf((position.y - minBounds.y) / cellSize)
    );

    int cellZ = static_cast<int>(
        floorf((position.z - minBounds.z) / cellSize)
    );

    return { cellX, cellY, cellZ };
}

__device__ inline bool isCellValid(int3 cell, int3 gridSize) {
    return (cell.x >= 0 && cell.x < gridSize.x) &&
        (cell.y >= 0 && cell.y < gridSize.y) &&
        (cell.z >= 0 && cell.z < gridSize.z);
}

__device__ inline std::uint32_t cellToKey(int3 cell, int3 gridSize) {
    return static_cast<std::uint32_t>(
        cell.x + cell.y * gridSize.x + cell.z * gridSize.x * gridSize.y
    );
}

__device__ inline int3 clampCellToGrid(int3 cell, int3 gridSize) {
    cell.x = max(0, min(cell.x, gridSize.x - 1));
    cell.y = max(0, min(cell.y, gridSize.y - 1));
    cell.z = max(0, min(cell.z, gridSize.z - 1));

    return cell;
}

#endif
