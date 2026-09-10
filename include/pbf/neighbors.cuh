#ifndef NEIGHBORS_H
#define NEIGHBORS_H


#include "pbf/spatial_grid.cuh"

#include <cuda_runtime.h>
#include <cstdint>

// predictedPositions must be in the same cell-sorted
// order described by cellStart/cellEnd, so cell ranges contain working particle
// indices directly. At most maxNeighbors indices are stored per particle;
// neighborsCount reports the full count so truncation is visible. Neighbor
// slots are stored slot-major:
// neighbors[neighborOffset * particleStride + particleIndex].
__global__
void findNeighbors(const float4* predictedPositions,
                   const int* cellStart, const int* cellEnd,
                   int3 gridSize, float3 minBounds, float cellSize,
                   size_t particleCount, float smoothingRadius,
                   uint32_t* neighbors, int* neighborsCount,
                   int maxNeighbors, size_t particleStride,
                   int* overflowFlag = nullptr);


#endif
