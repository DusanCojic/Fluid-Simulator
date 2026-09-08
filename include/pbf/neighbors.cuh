#ifndef NEIGHBORS_H
#define NEIGHBORS_H


#include "pbf/spatial_grid.cuh"

#include <cuda_runtime.h>
#include <cstdint>

// Finds particles within the smoothing radius. At most maxNeighbors indices are
// stored per particle; neighborsCount reports the full count so truncation is visible.
__global__
void findNeighbors(const float4* predictedPositions, const uint32_t* sortedIndices, const int* cellStart, const int* cellEnd,
                   int3 gridSize, float3 minBounds, float cellSize, size_t particleCount, float smoothingRadius,
                   uint32_t* neighbors, int* neighborsCount, int maxNeighbors,
                   int* overflowFlag = nullptr);


#endif
