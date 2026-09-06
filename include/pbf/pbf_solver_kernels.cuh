#ifndef PBF_SOLVER_KERNELS_H
#define PBF_SOLVER_KERNELS_H


#include "pbf/sph_helpers.cuh"

#include <cuda_runtime.h>
#include <cstdint>

// neighborsCount must be in [0, maxNeighbors] for every particle. A larger
// count means findNeighbors truncated the fixed-stride list; solver kernels
// reject that particle rather than compute from an incomplete neighborhood.
__global__
void computeDensity(const float4* predictedPosition, const uint32_t* neighbors, const int* neighborsCount, const int maxNeighbors,
                    size_t particleCount, float smoothingRadius, float particleMass, float* density, float* constraints, const float restDensity);

__global__
void computeLambda(const float4* predictedPosition, const uint32_t* neighbors, const int* neighborsCount, int maxNeighbors, 
    const float* constraints, size_t particleCount, float smoothingRadius, float particleMass, float restDensity, float epsilon, float* lambdas);


    __global__
void computeDeltaPosition(const float4* predictedPosition, const uint32_t* neighbors, const int* neighborsCount, int maxNeighbors, 
    const float* lambdas, size_t particleCount, float smoothingRadius, float particleMass, float restDensity, float4* deltaPositions);


__global__
void applyDeltaPosition(float4* predictedPosition, const float4* deltaPositions, size_t particleCount);

__global__
void updateVelocityAndPosition(float4* positions, const float4* predictedPositions, float4* velocities, 
    std::size_t particleCount, float inverseDt);


#endif
