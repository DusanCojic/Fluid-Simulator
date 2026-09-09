#include "pbf/neighbors.cuh"

__global__
void findNeighbors(const float4* predictedPositions, const uint32_t* sortedIndices, const int* cellStart, const int* cellEnd,
                   int3 gridSize, float3 minBounds, float cellSize, size_t particleCount, float smoothingRadius,
                   uint32_t* neighbors, int* neighborsCount, int maxNeighbors,
                   int* overflowFlag) {

    size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (index >= particleCount)
        return;

    // Each thread finds neighbors for one particle.
    const float4 currentPos = predictedPositions[index];

    // Use the same boundary behavior as the spatial grid builder.
    const int3 currentCell = clampCellToGrid(
        positionToCell(currentPos, minBounds, cellSize),
        gridSize
    );
    const float smoothingRadiusSquared = smoothingRadius * smoothingRadius;

    // Bound the search before integer conversion/addition. A support radius
    // larger than the domain must visit the domain once, without overflowing
    // signed offsets or looping over billions of invalid cells.
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
                const int3 cell = {x, y, z};

                uint32_t key = cellToKey(cell, gridSize);

                int start = cellStart[key];
                int end = cellEnd[key];

                if (start == -1)
                    continue;

                // Test particles stored in this cell against the smoothing radius.
                for (int sortedIndex = start; sortedIndex < end; ++sortedIndex) {
                    uint32_t neighborIndex = sortedIndices[sortedIndex];

                    if (neighborIndex == index)
                        continue;

                    float4 neighborPos = predictedPositions[neighborIndex];

                    float diffX = currentPos.x - neighborPos.x;
                    float diffY = currentPos.y - neighborPos.y;
                    float diffZ = currentPos.z - neighborPos.z;

                    float distanceSquared =
                        diffX * diffX +
                        diffY * diffY +
                        diffZ * diffZ;

                    if (distanceSquared <= smoothingRadiusSquared) {
                        // Store only what fits, but count every neighbor so overflow is visible.
                        if (count < maxNeighbors)
                            neighbors[index * maxNeighbors + count] = neighborIndex;

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
