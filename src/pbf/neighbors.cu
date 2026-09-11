#include "pbf/neighbors.cuh"

__global__
void findNeighbors(const float4* predictedPositions,
                   const int* cellStart, const int* cellEnd,
                   int3 gridSize, float3 minBounds, float cellSize,
                   size_t particleCount, float smoothingRadius,
                   uint32_t* neighbors, int* neighborsCount,
                   int maxNeighbors, size_t particleStride,
                   int* overflowFlag) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    // Particles are ordered by spatial cell before this kernel is launched, so
    // neighbors frequently belong to the same CUDA block. Cache that block's
    // positions to avoid repeatedly fetching them through indirect global
    // memory accesses. The fallback keeps launches larger than the solver's
    // normal 256-thread block size correct.
    constexpr unsigned int cachedBlockCapacity = 256;
    __shared__ float4 cachedBlockPositions[cachedBlockCapacity];
    const bool cacheBlock = blockDim.x <= cachedBlockCapacity;
    const size_t blockStart = static_cast<size_t>(blockIdx.x) * blockDim.x;
    const size_t blockEnd = min(blockStart + blockDim.x, particleCount);

    if (cacheBlock && index < particleCount)
        cachedBlockPositions[threadIdx.x] = predictedPositions[index];

    // Every thread must reach the barrier, including inactive threads in the
    // final partial block.
    __syncthreads();

    if (index >= particleCount)
        return;

    const float4 currentPos = predictedPositions[index];
    const int3 currentCell = clampCellToGrid(
        positionToCell(currentPos, minBounds, cellSize), gridSize
    );
    const float smoothingRadiusSquared = smoothingRadius * smoothingRadius;
    const double radius = ceil(static_cast<double>(smoothingRadius) / cellSize);
    const int3 first = {
        static_cast<int>(fmax(0.0, currentCell.x - radius)),
        static_cast<int>(fmax(0.0, currentCell.y - radius)),
        static_cast<int>(fmax(0.0, currentCell.z - radius))
    };
    const int3 last = {
        static_cast<int>(fmin(static_cast<double>(gridSize.x - 1), currentCell.x + radius)),
        static_cast<int>(fmin(static_cast<double>(gridSize.y - 1), currentCell.y + radius)),
        static_cast<int>(fmin(static_cast<double>(gridSize.z - 1), currentCell.z + radius))
    };

    int count = 0;
    for (int z = first.z; z <= last.z; ++z) {
        for (int y = first.y; y <= last.y; ++y) {
            for (int x = first.x; x <= last.x; ++x) {
                const uint32_t key = cellToKey({x, y, z}, gridSize);
                const int start = cellStart[key];
                const int end = cellEnd[key];
                if (start == -1)
                    continue;

                for (int neighborIndex = start; neighborIndex < end; ++neighborIndex) {
                    if (static_cast<size_t>(neighborIndex) == index)
                        continue;

                    const size_t neighbor = static_cast<size_t>(neighborIndex);
                    const float4 neighborPos =
                        cacheBlock && neighbor >= blockStart && neighbor < blockEnd
                            ? cachedBlockPositions[neighbor - blockStart]
                            : predictedPositions[neighborIndex];
                    const float diffX = currentPos.x - neighborPos.x;
                    const float diffY = currentPos.y - neighborPos.y;
                    const float diffZ = currentPos.z - neighborPos.z;
                    const float distanceSquared =
                        diffX * diffX + diffY * diffY + diffZ * diffZ;

                    if (distanceSquared <= smoothingRadiusSquared) {
                        if (count < maxNeighbors) {
                            neighbors[static_cast<size_t>(count) * particleStride + index] =
                                static_cast<uint32_t>(neighborIndex);
                        }
                        ++count;
                    }
                }
            }
        }
    }

    neighborsCount[index] = count;
    if (count > maxNeighbors && overflowFlag != nullptr)
        atomicExch(overflowFlag, 1);
}
