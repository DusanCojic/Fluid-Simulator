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

                    const float4 neighborPos = predictedPositions[neighborIndex];
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
