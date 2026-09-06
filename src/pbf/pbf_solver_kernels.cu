#include "pbf/pbf_solver_kernels.cuh"

#include <cassert>
#include <math_constants.h>

namespace {

__device__
bool hasValidNeighborCount(int neighborCount, int maxNeighbors) {
    const bool isValid = neighborCount >= 0 && neighborCount <= maxNeighbors;

    assert(isValid && "neighbor list overflow: resize and rebuild the neighbor list");
    return isValid;
}

} // namespace

__global__
void computeDensity(const float4* predictedPosition, const uint32_t* neighbors, const int* neighborsCount, const int maxNeighbors,
                    std::size_t particleCount, float smoothingRadius, float particleMass, float* density, float* constraints, const float restDensity) {

    std::size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (index >= particleCount)
        return;

    const float4 particlePosition = predictedPosition[index];

    float particleDensity =
        particleMass * poly6({0.0f, 0.0f, 0.0f}, smoothingRadius);

    const int neighborsCnt = neighborsCount[index];

    if (!hasValidNeighborCount(neighborsCnt, maxNeighbors)) {
        density[index] = CUDART_NAN_F;
        constraints[index] = CUDART_NAN_F;
        return;
    }

    for (int i = 0; i < neighborsCnt; ++i) {
        uint32_t neighborIndex = neighbors[index * maxNeighbors + i];

        const float4 neighborPosition = predictedPosition[neighborIndex];

        const float diffX = particlePosition.x - neighborPosition.x;
        const float diffY = particlePosition.y - neighborPosition.y;
        const float diffZ = particlePosition.z - neighborPosition.z;

        const float3 displacement = {diffX, diffY, diffZ};

        particleDensity +=
            particleMass * poly6(displacement, smoothingRadius);
    }

    density[index] = particleDensity;

    constraints[index] = particleDensity / restDensity - 1.0f;
}

__global__
void computeLambda(const float4* predictedPosition, const uint32_t* neighbors, const int* neighborsCount, int maxNeighbors, 
    const float* constraints, size_t particleCount, float smoothingRadius, float particleMass, float restDensity, float epsilon, float* lambdas) {

    const size_t index =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (index >= particleCount)
        return;

    const float4 particlePosition = predictedPosition[index];
    const float constraint = constraints[index];

    const int neighborsCnt = neighborsCount[index];

    if (!hasValidNeighborCount(neighborsCnt, maxNeighbors)) {
        lambdas[index] = CUDART_NAN_F;
        return;
    }

    const float gradientScale =
        particleMass / restDensity;

    float3 gradientI = {0.0f, 0.0f, 0.0f};
    float sumGradientSquared = 0.0f;

    for (int i = 0; i < neighborsCnt; ++i) {
        const uint32_t neighborIndex = neighbors[index * maxNeighbors + i];

        const float4 neighborPosition = predictedPosition[neighborIndex];

        const float diffX =
            particlePosition.x - neighborPosition.x;

        const float diffY =
            particlePosition.y - neighborPosition.y;

        const float diffZ =
            particlePosition.z - neighborPosition.z;

        const float3 displacement =
            {diffX, diffY, diffZ};

        const float3 gradientW =
            spikyGradient(displacement, smoothingRadius);

        const float3 gradientJ = {
            -gradientScale * gradientW.x,
            -gradientScale * gradientW.y,
            -gradientScale * gradientW.z
        };

        sumGradientSquared +=
            gradientJ.x * gradientJ.x +
            gradientJ.y * gradientJ.y +
            gradientJ.z * gradientJ.z;

        gradientI.x -= gradientJ.x;
        gradientI.y -= gradientJ.y;
        gradientI.z -= gradientJ.z;
    }

    sumGradientSquared +=
        gradientI.x * gradientI.x +
        gradientI.y * gradientI.y +
        gradientI.z * gradientI.z;

    lambdas[index] =
        -constraint / (sumGradientSquared + epsilon);
}


    __global__
void computeDeltaPosition(const float4* predictedPosition, const uint32_t* neighbors, const int* neighborsCount, int maxNeighbors, 
    const float* lambdas, size_t particleCount, float smoothingRadius, float particleMass, float restDensity, float4* deltaPositions) {

    const size_t index =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (index >= particleCount)
        return;

    const float4 particlePosition = predictedPosition[index];
    const float lambdaI = lambdas[index];

    const int neighborsCnt = neighborsCount[index];

    if (!hasValidNeighborCount(neighborsCnt, maxNeighbors)) {
        deltaPositions[index] = {
            CUDART_NAN_F,
            CUDART_NAN_F,
            CUDART_NAN_F,
            0.0f
        };
        return;
    }

    float3 deltaPosition = {0.0f, 0.0f, 0.0f};

    for (int i = 0; i < neighborsCnt; ++i) {
        const uint32_t neighborIndex =
            neighbors[index * maxNeighbors + i];

        const float4 neighborPosition =
            predictedPosition[neighborIndex];

        const float diffX =
            particlePosition.x - neighborPosition.x;

        const float diffY =
            particlePosition.y - neighborPosition.y;

        const float diffZ =
            particlePosition.z - neighborPosition.z;

        const float3 displacement =
            {diffX, diffY, diffZ};

        const float3 gradientW =
            spikyGradient(displacement, smoothingRadius);

        const float lambdaSum =
            lambdaI + lambdas[neighborIndex];

        deltaPosition.x += lambdaSum * gradientW.x;
        deltaPosition.y += lambdaSum * gradientW.y;
        deltaPosition.z += lambdaSum * gradientW.z;
    }

    const float correctionScale =
        particleMass / restDensity;

    deltaPositions[index] = {
        correctionScale * deltaPosition.x,
        correctionScale * deltaPosition.y,
        correctionScale * deltaPosition.z,
        0.0f
    };
}

__global__
void applyDeltaPosition(float4* predictedPosition, const float4* deltaPositions, size_t particleCount) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (index >= particleCount) return;

    predictedPosition[index].x += deltaPositions[index].x;
    predictedPosition[index].y += deltaPositions[index].y;
    predictedPosition[index].z += deltaPositions[index].z;
}

__global__
void updateVelocityAndPosition(float4* positions, const float4* predictedPositions, float4* velocities, 
    std::size_t particleCount, float inverseDt) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (index >= particleCount)
        return;

    const float4 position = positions[index];
    const float4 predictedPosition = predictedPositions[index];

    velocities[index].x = (predictedPosition.x - position.x) * inverseDt;
    velocities[index].y = (predictedPosition.y - position.y) * inverseDt;
    velocities[index].z = (predictedPosition.z - position.z) * inverseDt;
    positions[index] = predictedPosition;
}