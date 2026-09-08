#include "pbf/collision_system.hpp"

#include <cmath>

namespace {

void validateParticleFitsContainer(const Container& container, float particleRadius) {
    const double particleDiameter = 2.0 * static_cast<double>(particleRadius);
    const double containerWidth =
        static_cast<double>(container.max.x) - static_cast<double>(container.min.x);
    const double containerDepth =
        static_cast<double>(container.max.z) - static_cast<double>(container.min.z);

    if (containerWidth < particleDiameter || containerDepth < particleDiameter)
        throw std::invalid_argument("Particle diameter exceeds closed container dimensions");
}

} // namespace

CollisionSystem::CollisionSystem(const Container& container) {
    setContainer(container);
}

void CollisionSystem::setContainer(const Container& container) {
    if (!std::isfinite(container.min.x) || !std::isfinite(container.min.y) ||
        !std::isfinite(container.min.z) || !std::isfinite(container.max.x) ||
        !std::isfinite(container.max.y) || !std::isfinite(container.max.z) ||
        container.max.x <= container.min.x || container.max.y <= container.min.y ||
        container.max.z <= container.min.z) {
        throw std::invalid_argument("Container bounds must be finite and ordered");
    }

    if (_container.size() == 0)
        _container.allocate(1);

    if (_correctionFlag.size() == 0)
        _correctionFlag.allocate(1);

    _container.copyFromHostToDevice(&container, 1);
    _hostContainer = container;
    _hasContainer = true;
}

void CollisionSystem::setSpheres(const std::vector<SphereCollider>& spheres) {
    if (spheres.size() == 0) {
        clearSpheres();
        return;
    }

    for (const SphereCollider& sphere : spheres) {
        if (!std::isfinite(sphere.center.x) || !std::isfinite(sphere.center.y) ||
            !std::isfinite(sphere.center.z) || !std::isfinite(sphere.radius) ||
            sphere.radius <= 0.0f) {
            throw std::invalid_argument("Sphere radius must be positive");
        }
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
        if (!std::isfinite(box.center.x) || !std::isfinite(box.center.y) ||
            !std::isfinite(box.center.z) ||
            !std::isfinite(box.halfExtents.x) || box.halfExtents.x <= 0.0f ||
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

        if (!std::isfinite(plane.point.x) || !std::isfinite(plane.point.y) ||
            !std::isfinite(plane.point.z) || !std::isfinite(normalLengthSquared) ||
            normalLengthSquared == 0.0f) {
            throw std::invalid_argument("Plane normal must not be zero");
        }
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

    if (!_hasContainer)
        throw std::logic_error("CollisionSystem container has not been configured");

    if (!std::isfinite(particleRadius) || particleRadius <= 0.0f)
        throw std::invalid_argument("particleRadius must be positive");

    validateParticleFitsContainer(_hostContainer, particleRadius);

    constexpr int maxCollisionIterations = 32;

    for (int iteration = 0; iteration < maxCollisionIterations; ++iteration) {
        _correctionFlag.fillBytes(0);

        solveContainer(predictedPositions, particleCount, particleRadius);

        if (_sphereCount > 0)
            solveSpheres(predictedPositions, particleCount, particleRadius);

        if (_boxCount > 0)
            solveBoxes(predictedPositions, particleCount, particleRadius);

        if (_planeCount > 0)
            solvePlanes(predictedPositions, particleCount, particleRadius);

        int correctionMade = 0;
        _correctionFlag.copyFromDeviceToHost(&correctionMade, 1);

        if (correctionMade == 0)
            return;
    }

    throw std::runtime_error("Collision constraints did not converge");
}

void CollisionSystem::resolveVelocities(const float4* positions, float4* velocities,
    std::size_t particleCount, float particleRadius, float restitution, float friction) {
    resolveVelocities(
        positions, velocities, velocities, particleCount, particleRadius,
        restitution, friction
    );
}

void CollisionSystem::resolveVelocities(const float4* positions,
    const float4* incomingVelocities, float4* velocities,
    std::size_t particleCount, float particleRadius, float restitution,
    float friction) {
    if (positions == nullptr || incomingVelocities == nullptr || velocities == nullptr)
        throw std::invalid_argument("positions and velocities must not be null");

    if (particleCount == 0)
        return;

    if (!_hasContainer)
        throw std::logic_error("CollisionSystem container has not been configured");

    if (!std::isfinite(particleRadius) || particleRadius <= 0.0f)
        throw std::invalid_argument("particleRadius must be positive");

    validateParticleFitsContainer(_hostContainer, particleRadius);

    if (!std::isfinite(restitution) || restitution < 0.0f || restitution > 1.0f)
        throw std::invalid_argument("restitution must be between zero and one");

    if (!std::isfinite(friction) || friction < 0.0f || friction > 1.0f)
        throw std::invalid_argument("friction must be between zero and one");

    constexpr int blockSize = 256;
    const int gridSize = static_cast<int>((particleCount + blockSize - 1) / blockSize);

    resolveVelocitiesKernel<<<gridSize, blockSize>>>(
        positions, incomingVelocities, velocities, particleCount, _container.data(),
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
        particleRadius,
        _correctionFlag.data()
    );

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(error));
}

void CollisionSystem::solveSpheres(float4* predictedPositions, std::size_t particleCount, float particleRadius) {
    constexpr int blockSize = 256;
    const int gridSize = static_cast<int>((particleCount + blockSize - 1) / blockSize);

    solveSpheresKernel<<<gridSize, blockSize>>>(
        predictedPositions, particleCount, _spheres.data(), _sphereCount, particleRadius,
        _correctionFlag.data()
    );

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(error));
}

void CollisionSystem::solveBoxes(float4* predictedPositions, std::size_t particleCount, float particleRadius) {
    constexpr int blockSize = 256;
    const int gridSize = static_cast<int>((particleCount + blockSize - 1) / blockSize);

    solveBoxesKernel<<<gridSize, blockSize>>>(
        predictedPositions, particleCount, _boxes.data(), _boxCount, particleRadius,
        _correctionFlag.data()
    );

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(error));
}

void CollisionSystem::solvePlanes(float4* predictedPositions, std::size_t particleCount, float particleRadius) {
    constexpr int blockSize = 256;
    const int gridSize = static_cast<int>((particleCount + blockSize - 1) / blockSize);

    solvePlanesKernel<<<gridSize, blockSize>>>(
        predictedPositions, particleCount, _planes.data(), _planeCount, particleRadius,
        _correctionFlag.data()
    );

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(error));
}
