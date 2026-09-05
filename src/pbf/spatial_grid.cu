#include "pbf/spatial_grid.cuh"

#include <cub/cub.cuh>

#include <cmath>
#include <cstddef>
#include <stdexcept>

void SpatialGrid::initialize(std::size_t maxParticles, float3 minBounds, float3 maxBounds, float cellSize) {
    // Validate configuration
    if (maxParticles == 0)
        throw std::invalid_argument("maxParticles must be greater than zero");

    if (cellSize <= 0.0f)
        throw std::invalid_argument("cellSize must be greater than zero");

    if (maxBounds.x <= minBounds.x || maxBounds.y <= minBounds.y || maxBounds.z <= minBounds.z)
        throw std::invalid_argument("Invalid spatial grid bounds");

    // Store configuration
    _maxParticles = maxParticles;
    _minBounds = minBounds;
    _maxBounds = maxBounds;
    _cellSize = cellSize;

    // Calculate grid dimensions
    _gridSize.x = static_cast<int>(
        std::ceil((_maxBounds.x - _minBounds.x) / _cellSize)
    );

    _gridSize.y = static_cast<int>(
        std::ceil((_maxBounds.y - _minBounds.y) / _cellSize)
    );

    _gridSize.z = static_cast<int>(
        std::ceil((_maxBounds.z - _minBounds.z) / _cellSize)
    );

    _numCells =
        static_cast<std::size_t>(_gridSize.x) *
        static_cast<std::size_t>(_gridSize.y) *
        static_cast<std::size_t>(_gridSize.z);

    // Allocate per-particle buffers
    _keys.allocate(_maxParticles);
    _indices.allocate(_maxParticles);

    _sortedKeys.allocate(_maxParticles);
    _sortedIndices.allocate(_maxParticles);

    // Allocate per-cell buffers
    _cellStart.allocate(_numCells);
    _cellEnd.allocate(_numCells);

    // Ask CUB how much temporary storage radix sort needs
    _sortTempStorageBytes = 0;

    cudaError_t error = cub::DeviceRadixSort::SortPairs(
        nullptr,
        _sortTempStorageBytes,
        _keys.data(),
        _sortedKeys.data(),
        _indices.data(),
        _sortedIndices.data(),
        _maxParticles
    );

    if (error != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(error));

    // Allocate CUB temporary storage once
    _sortTempStorage.allocate(_sortTempStorageBytes);
}

// Computes the spatial grid key and original index for each particle.
__global__
void computeParticleKeys(const float4* predictedPositions, std::uint32_t* keys, std::uint32_t* indices,
    std::size_t particleCount, float3 minBounds, float cellSize, int3 gridSize
) {
    std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= particleCount)
        return;

    float4 predictedPos = predictedPositions[index];

    int3 cell = positionToCell(predictedPos, minBounds, cellSize);

    if (!isCellValid(cell, gridSize))
        cell = clampCellToGrid(cell, gridSize);

    std::uint32_t key = cellToKey(cell, gridSize);

    keys[index] = key;
    indices[index] = static_cast<std::uint32_t>(index);;
}

// calculates boundries of each cell
__global__
void buildCellRanges(const std::uint32_t* sortedKeys, int* cellStart, int* cellEnd, std::size_t particleCount) {
    std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (index >= particleCount)
        return;
        
    std::uint32_t key = sortedKeys[index];

    if (index == 0 || sortedKeys[index - 1] != key)
        cellStart[key] = static_cast<int>(index);

    if (index == particleCount - 1 || sortedKeys[index + 1] != key)
        cellEnd[key] = static_cast<int>(index + 1);
}

void SpatialGrid::build(const float4* predictedPositions, std::size_t particleCount) {
    if (particleCount > _maxParticles)
        throw std::out_of_range(
            "Particle count exceeds maximum number of particles"
        );

    // An empty build still has to clear ranges left by a previous build, but it
    // must not launch a kernel with a zero-sized grid.
    if (particleCount == 0) {
        _cellStart.fillBytes(-1);
        _cellEnd.fillBytes(-1);

        const cudaError_t error = cudaDeviceSynchronize();
        if (error != cudaSuccess)
            throw std::runtime_error(cudaGetErrorString(error));

        return;
    }

    constexpr int blockSize = 256;

    const int gridSize = static_cast<int>(
        (particleCount + blockSize - 1) / blockSize
    );

    // Compute key for each particle

    computeParticleKeys<<<gridSize, blockSize>>>(
        predictedPositions,
        _keys.data(),
        _indices.data(),
        particleCount,
        _minBounds,
        _cellSize,
        _gridSize
    );

    cudaError_t error = cudaGetLastError();

    if (error != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(error));

    // Sort particles by cell key

    error = cub::DeviceRadixSort::SortPairs(
        _sortTempStorage.data(),
        _sortTempStorageBytes,
        _keys.data(),
        _sortedKeys.data(),
        _indices.data(),
        _sortedIndices.data(),
        particleCount
    );

    if (error != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(error));

    // Mark all cells as empty

    _cellStart.fillBytes(-1);
    _cellEnd.fillBytes(-1);

    // Find the start and end of each occupied cell

    buildCellRanges<<<gridSize, blockSize>>>(
        _sortedKeys.data(),
        _cellStart.data(),
        _cellEnd.data(),
        particleCount
    );

    error = cudaGetLastError();

    if (error != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(error));

    // cudaGetLastError only validates the launch. Synchronizing here ensures
    // that asynchronous execution failures are reported by this API call.
    error = cudaDeviceSynchronize();

    if (error != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(error));
}

// Getters

const std::uint32_t* SpatialGrid::sortedKeys() const {
    return _sortedKeys.data();
}

const std::uint32_t* SpatialGrid::sortedIndices() const {
    return _sortedIndices.data();
}

const int* SpatialGrid::cellStart() const {
    return _cellStart.data();
}

const int* SpatialGrid::cellEnd() const {
    return _cellEnd.data();
}

int3 SpatialGrid::gridSize() const {
    return _gridSize;
}

std::size_t SpatialGrid::numCells() const {
    return _numCells;
}

std::size_t SpatialGrid::maxParticles() const {
    return _maxParticles;
}

float SpatialGrid::cellSize() const {
    return _cellSize;
}

float3 SpatialGrid::minBounds() const {
    return _minBounds;
}

float3 SpatialGrid::maxBounds() const {
    return _maxBounds;
}
