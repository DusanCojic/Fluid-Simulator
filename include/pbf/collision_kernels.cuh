#ifndef COLLISION_KERNELS_H
#define COLLISION_KERNELS_H


#include <cuda_runtime.h>
#include "pbf/colliders.hpp"

__global__
void solveContainerKernel(float4* predictedPositions, std::size_t particleCount,
    const Container* container, float particleRadius, int* correctionFlag = nullptr);

__global__
void solveSpheresKernel(float4* predictedPositions, std::size_t particleCount,
    const SphereCollider* spheres, std::size_t sphereCount, float particleRadius,
    int* correctionFlag = nullptr);

__global__
void solveBoxesKernel(float4* predictedPositions, std::size_t particleCount,
    const BoxCollider* boxes, std::size_t boxCount, float particleRadius,
    int* correctionFlag = nullptr);

__global__
void solvePlanesKernel(float4* predictedPositions, std::size_t particleCount,
    const PlaneCollider* planes, std::size_t planeCount, float particleRadius,
    int* correctionFlag = nullptr);

__global__
void resolveVelocitiesKernel(const float4* positions, const float4* incomingVelocities,
    float4* velocities,
    std::size_t particleCount, const Container* container,
    const SphereCollider* spheres, std::size_t sphereCount,
    const BoxCollider* boxes, std::size_t boxCount,
    const PlaneCollider* planes, std::size_t planeCount,
    float particleRadius, float restitution, float friction);


#endif
