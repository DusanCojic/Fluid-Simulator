#include "pbf/neighbors.cuh"

__global__
void findNeighbors(const float4* predictedPositions, const uint32_t* sortedIndices, const int* cellStart, const int* cellEnd,
                   int3 gridSize, float3 minBounds, float cellSize, size_t particleCount, float smoothingRadius,
                   uint32_t* neighbors, int* neighborsCount, int maxNeighbors) {

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

    // Number of grid cells needed to cover the smoothing radius.
    const int cellSearchRadius = static_cast<int>(ceilf(smoothingRadius / cellSize));

    int count = 0;

    // Check every grid cell that can contain a neighbor.
    for (int cellOffsetZ = -cellSearchRadius; cellOffsetZ <= cellSearchRadius; ++cellOffsetZ) {
        for (int cellOffsetY = -cellSearchRadius; cellOffsetY <= cellSearchRadius; ++cellOffsetY) {
            for (int cellOffsetX = -cellSearchRadius; cellOffsetX <= cellSearchRadius; ++cellOffsetX) {

                int3 cell = {
                    currentCell.x + cellOffsetX,
                    currentCell.y + cellOffsetY,
                    currentCell.z + cellOffsetZ
                };

                if (!isCellValid(cell, gridSize))
                    continue;

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
}
