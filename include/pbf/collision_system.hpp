#ifndef COLLISION_SYSTEM_H
#define COLLISION_SYSTEM_H


#include "pbf/colliders.hpp"
#include "pbf/cuda_buffer.hpp"
#include "pbf/collision_kernels.cuh"

#include <vector>

class CollisionSystem {
public:
    CollisionSystem() = default;
    CollisionSystem(const Container& container);
    void setContainer(const Container& container);

    void setSpheres(const std::vector<SphereCollider>& spheres);
    void clearSpheres();

    void setBoxes(const std::vector<BoxCollider>& boxes);
    void clearBoxes();

    void setPlanes(const std::vector<PlaneCollider>& planes);
    void clearPlanes();

    void solve(float4* predictedPositions, std::size_t particleCount, float particleRadius);

    void resolveVelocities(const float4* positions, float4* velocities,
        std::size_t particleCount, float particleRadius, float restitution, float friction);

private:
    void solveContainer(float4* predictedPositions, std::size_t particleCount, float particleRadius);

    void solveSpheres(float4* predictedPositions, std::size_t particleCount, float particleRadius);

    void solveBoxes(float4* predictedPositions, std::size_t particleCount, float particleRadius);

    void solvePlanes(float4* predictedPositions, std::size_t particleCount, float particleRadius);


    CudaBuffer<Container> _container;

    CudaBuffer<SphereCollider> _spheres;
    CudaBuffer<BoxCollider> _boxes;
    CudaBuffer<PlaneCollider> _planes;

    std::size_t _sphereCount = 0;
    std::size_t _boxCount = 0;
    std::size_t _planeCount = 0;
};


#endif
