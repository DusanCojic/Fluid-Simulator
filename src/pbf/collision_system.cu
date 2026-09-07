#include "pbf/collision_system.hpp"

#include <cmath>

CollisionSystem::CollisionSystem(const Container& container) {
    _container.allocate(1);
    _container.copyFromHostToDevice(&container, 1);
}

void CollisionSystem::setContainer(const Container& container) {
    if (_container.size() == 0)
        _container.allocate(1);

    _container.copyFromHostToDevice(&container, 1);
}

void CollisionSystem::setSpheres(const std::vector<SphereCollider>& spheres) {
    if (spheres.size() == 0) {
        clearSpheres();
        return;
    }

    for (const SphereCollider& sphere : spheres) {
        if (!std::isfinite(sphere.radius) || sphere.radius <= 0.0f)
            throw std::invalid_argument("Sphere radius must be positive");
    }

    _sphereCount = spheres.size();

    _spheres.allocate(_sphereCount);

    _spheres.copyFromHostToDevice(spheres.data(), _sphereCount);
}

void CollisionSystem::clearSpheres() {
    _sphereCount = 0;
    _spheres.allocate(0);
}

void CollisionSystem::setBoxes(const std::vector<BoxCollider>& boxes) {
    if (boxes.size() == 0) {
        clearBoxes();
        return;
    }

    for (const BoxCollider& box : boxes) {
        if (!std::isfinite(box.halfExtents.x) || box.halfExtents.x <= 0.0f ||
            !std::isfinite(box.halfExtents.y) || box.halfExtents.y <= 0.0f ||
            !std::isfinite(box.halfExtents.z) || box.halfExtents.z <= 0.0f) {
            throw std::invalid_argument("Box half extents must be positive");
        }
    }

    _boxCount = boxes.size();

    _boxes.allocate(_boxCount);

    _boxes.copyFromHostToDevice(boxes.data(), _boxCount);
}

void CollisionSystem::clearBoxes() {
    _boxCount = 0;
    _boxes.allocate(0);
}

void CollisionSystem::setPlanes(const std::vector<PlaneCollider>& planes) {
    if (planes.size() == 0) {
        clearPlanes();
        return;
    }

    for (const PlaneCollider& plane : planes) {
        const float normalLengthSquared = plane.normal.x * plane.normal.x +
            plane.normal.y * plane.normal.y + plane.normal.z * plane.normal.z;

        if (!std::isfinite(normalLengthSquared) || normalLengthSquared == 0.0f)
            throw std::invalid_argument("Plane normal must not be zero");
    }

    _planeCount = planes.size();

    _planes.allocate(_planeCount);

    _planes.copyFromHostToDevice(planes.data(), _planeCount);
}

void CollisionSystem::clearPlanes() {
    _planeCount = 0;
    _planes.allocate(0);
}

void CollisionSystem::solve(float4* predictedPositions, std::size_t particleCount, float particleRadius) {
     if (predictedPositions == nullptr)
        throw std::invalid_argument("predictedPositions must not be null");

    if (particleCount == 0)
        return;

    if (!std::isfinite(particleRadius) || particleRadius <= 0.0f)
        throw std::invalid_argument("particleRadius must be positive");

    for (int iteration = 0; iteration < 2; ++iteration) {
        solveContainer(predictedPositions, particleCount, particleRadius);

        if (_sphereCount > 0)
            solveSpheres(predictedPositions, particleCount, particleRadius);

        if (_boxCount > 0)
            solveBoxes(predictedPositions, particleCount, particleRadius);

        if (_planeCount > 0)
            solvePlanes(predictedPositions, particleCount, particleRadius);
    }
}

void CollisionSystem::resolveVelocities(const float4* positions, float4* velocities,
    std::size_t particleCount, float particleRadius, float restitution, float friction) {
    if (positions == nullptr || velocities == nullptr)
        throw std::invalid_argument("positions and velocities must not be null");

    if (particleCount == 0)
        return;

    if (!std::isfinite(particleRadius) || particleRadius <= 0.0f)
        throw std::invalid_argument("particleRadius must be positive");

    if (!std::isfinite(restitution) || restitution < 0.0f || restitution > 1.0f)
        throw std::invalid_argument("restitution must be between zero and one");

    if (!std::isfinite(friction) || friction < 0.0f || friction > 1.0f)
        throw std::invalid_argument("friction must be between zero and one");

    constexpr int blockSize = 256;
    const int gridSize = static_cast<int>((particleCount + blockSize - 1) / blockSize);

    resolveVelocitiesKernel<<<gridSize, blockSize>>>(
        positions, velocities, particleCount, _container.data(),
        _spheres.data(), _sphereCount, _boxes.data(), _boxCount,
        _planes.data(), _planeCount, particleRadius, restitution, friction
    );

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(error));
}

void CollisionSystem::solveContainer(float4* predictedPositions, std::size_t particleCount, float particleRadius) {
    constexpr int blockSize = 256;
    const int gridSize = static_cast<int>((particleCount + blockSize - 1) / blockSize);

    solveContainerKernel<<<gridSize, blockSize>>>(
        predictedPositions, 
        particleCount, 
        _container.data(), 
        particleRadius
    );

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(error));
}

void CollisionSystem::solveSpheres(float4* predictedPositions, std::size_t particleCount, float particleRadius) {
    constexpr int blockSize = 256;
    const int gridSize = static_cast<int>((particleCount + blockSize - 1) / blockSize);

    solveSpheresKernel<<<gridSize, blockSize>>>(
        predictedPositions, particleCount, _spheres.data(), _sphereCount, particleRadius
    );

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(error));
}

void CollisionSystem::solveBoxes(float4* predictedPositions, std::size_t particleCount, float particleRadius) {
    constexpr int blockSize = 256;
    const int gridSize = static_cast<int>((particleCount + blockSize - 1) / blockSize);

    solveBoxesKernel<<<gridSize, blockSize>>>(
        predictedPositions, particleCount, _boxes.data(), _boxCount, particleRadius
    );

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(error));
}

void CollisionSystem::solvePlanes(float4* predictedPositions, std::size_t particleCount, float particleRadius) {
    constexpr int blockSize = 256;
    const int gridSize = static_cast<int>((particleCount + blockSize - 1) / blockSize);

    solvePlanesKernel<<<gridSize, blockSize>>>(
        predictedPositions, particleCount, _planes.data(), _planeCount, particleRadius
    );

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(error));
}
