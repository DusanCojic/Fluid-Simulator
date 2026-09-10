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

__device__
std::uint32_t neighborAt(const std::uint32_t* neighbors,
                         std::size_t particleIndex, int neighborOffset,
                         std::size_t particleStride) {
    return neighbors[static_cast<std::size_t>(neighborOffset) * particleStride +
                     particleIndex];
}

} // namespace

__global__
void computeLambda(const float4* predictedPositions, const uint32_t* neighbors,
                   const int* neighborsCount, int maxNeighbors,
                   size_t particleCount, float smoothingRadius,
                   float particleMass, float restDensity, float epsilon,
                   float* lambdas, std::size_t neighborParticleStride) {

    const size_t index =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (index >= particleCount)
        return;

    const float4 particlePosition = predictedPositions[index];

    const int neighborsCnt = neighborsCount[index];
    if (!hasValidNeighborCount(neighborsCnt, maxNeighbors)) {
        lambdas[index] = CUDART_NAN_F;
        return;
    }

    const float gradientScale =
        particleMass / restDensity;

    float density = particleMass * poly6({0.0f, 0.0f, 0.0f}, smoothingRadius);
    float3 gradientI = {0.0f, 0.0f, 0.0f};
    float sumGradientSquared = 0.0f;

    for (int i = 0; i < neighborsCnt; ++i) {
        const uint32_t neighborIndex =
            neighborAt(neighbors, index, i, neighborParticleStride);

        const float4 neighborPosition = predictedPositions[neighborIndex];

        const float diffX =
            particlePosition.x - neighborPosition.x;

        const float diffY =
            particlePosition.y - neighborPosition.y;

        const float diffZ =
            particlePosition.z - neighborPosition.z;

        const float3 displacement =
            {diffX, diffY, diffZ};
        const float distanceSquared =
            diffX * diffX + diffY * diffY + diffZ * diffZ;

        density += particleMass *
            poly6FromDistanceSquared(distanceSquared, smoothingRadius);

        const float3 gradientW =
            spikyGradientFromDistanceSquared(
                displacement, distanceSquared, smoothingRadius
            );

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

    const float constraint = density / restDensity - 1.0f;
    lambdas[index] = -constraint / (sumGradientSquared + epsilon);
}


__global__
void computeDeltaPosition(const float4* predictedPosition, const uint32_t* neighbors, const int* neighborsCount, int maxNeighbors, 
    const float* lambdas, size_t particleCount, float smoothingRadius, float particleMass, float restDensity,
    float scorrK, int scorrN, float scorrDeltaQ, float4* deltaPositions,
    std::size_t neighborParticleStride) {

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
            neighborAt(neighbors, index, i, neighborParticleStride);

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

        const float distanceSquared =
            diffX * diffX + diffY * diffY + diffZ * diffZ;

        const float sCorr = computeArtificialPressure(
            distanceSquared, smoothingRadius, scorrK, scorrN, scorrDeltaQ
        );

        const float lambdaSum =
            lambdaI + lambdas[neighborIndex] + sCorr;

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

__global__
void applyXsphViscosity(const float4* predictedPositions, const uint32_t* neighbors, const int* neighborsCount, 
    int maxNeighbors, const float4* inputVelocities, float4* outputVelocities, std::size_t particleCount, 
    float smoothingRadius, float xsphViscosity, std::size_t neighborParticleStride) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (index >= particleCount)
        return;

    const float4 velocity = inputVelocities[index];
    const int neighborsCnt = neighborsCount[index];
    if (!hasValidNeighborCount(neighborsCnt, maxNeighbors)) {
        outputVelocities[index] = {
            CUDART_NAN_F,
            CUDART_NAN_F,
            CUDART_NAN_F,
            velocity.w
        };
        return;
    }

    if (xsphViscosity == 0.0f) {
        outputVelocities[index] = velocity;
        return;
    }

    const float4 particlePosition = predictedPositions[index];
    float3 correction = {0.0f, 0.0f, 0.0f};

    for (int neighborOffset = 0; neighborOffset < neighborsCnt; ++neighborOffset) {
        const uint32_t neighborIndex =
            neighborAt(neighbors, index, neighborOffset, neighborParticleStride);
        const float4 neighborPosition = predictedPositions[neighborIndex];
        const float4 neighborVelocity = inputVelocities[neighborIndex];
        const float3 displacement = {
            particlePosition.x - neighborPosition.x,
            particlePosition.y - neighborPosition.y,
            particlePosition.z - neighborPosition.z
        };
        const float weight = poly6(displacement, smoothingRadius);

        correction.x += (neighborVelocity.x - velocity.x) * weight;
        correction.y += (neighborVelocity.y - velocity.y) * weight;
        correction.z += (neighborVelocity.z - velocity.z) * weight;
    }

    outputVelocities[index] = {
        velocity.x + xsphViscosity * correction.x,
        velocity.y + xsphViscosity * correction.y,
        velocity.z + xsphViscosity * correction.z,
        velocity.w
    };
}

__global__
void computeVorticity(const float4* positions, const float4* velocities,
                      const uint32_t* neighbors, const int* neighborsCount,
                      int maxNeighbors, std::size_t particleCount,
                      float smoothingRadius, float4* vorticity,
                      std::size_t neighborParticleStride) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (index >= particleCount)
        return;

    const int neighborsCnt = neighborsCount[index];
    if (!hasValidNeighborCount(neighborsCnt, maxNeighbors)) {
        vorticity[index] = {CUDART_NAN_F, CUDART_NAN_F, CUDART_NAN_F, 0.0f};
        return;
    }

    const float4 position = positions[index];
    const float4 velocity = velocities[index];
    float3 omega = {0.0f, 0.0f, 0.0f};

    for (int neighborOffset = 0; neighborOffset < neighborsCnt; ++neighborOffset) {
        const uint32_t neighborIndex =
            neighborAt(neighbors, index, neighborOffset, neighborParticleStride);
        const float4 neighborPosition = positions[neighborIndex];
        const float4 neighborVelocity = velocities[neighborIndex];
        const float3 displacement = {
            position.x - neighborPosition.x,
            position.y - neighborPosition.y,
            position.z - neighborPosition.z
        };
        const float3 gradient = spikyGradient(displacement, smoothingRadius);
        const float3 velocityDifference = {
            neighborVelocity.x - velocity.x,
            neighborVelocity.y - velocity.y,
            neighborVelocity.z - velocity.z
        };

        // curl(v) uses grad_i W x (v_j - v_i).  Equivalently, the PBF
        // paper writes (v_j - v_i) x grad_j W.
        omega.x += gradient.y * velocityDifference.z - gradient.z * velocityDifference.y;
        omega.y += gradient.z * velocityDifference.x - gradient.x * velocityDifference.z;
        omega.z += gradient.x * velocityDifference.y - gradient.y * velocityDifference.x;
    }

    vorticity[index] = {omega.x, omega.y, omega.z, 0.0f};
}

__global__
void applyVorticityConfinement(const float4* positions, const uint32_t* neighbors,
                               const int* neighborsCount, int maxNeighbors,
                               const float4* vorticity,
                               const float4* inputVelocities,
                               float4* outputVelocities,
                               std::size_t particleCount,
                               float smoothingRadius, float dt,
                               float vorticityStrength,
                               std::size_t neighborParticleStride) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (index >= particleCount)
        return;

    const float4 velocity = inputVelocities[index];
    const int neighborsCnt = neighborsCount[index];
    if (!hasValidNeighborCount(neighborsCnt, maxNeighbors)) {
        outputVelocities[index] = {CUDART_NAN_F, CUDART_NAN_F, CUDART_NAN_F, velocity.w};
        return;
    }

    const float4 position = positions[index];
    const float4 omegaI = vorticity[index];
    const float omegaILength = sqrtf(
        omegaI.x * omegaI.x + omegaI.y * omegaI.y + omegaI.z * omegaI.z
    );
    float3 eta = {0.0f, 0.0f, 0.0f};

    for (int neighborOffset = 0; neighborOffset < neighborsCnt; ++neighborOffset) {
        const uint32_t neighborIndex =
            neighborAt(neighbors, index, neighborOffset, neighborParticleStride);
        const float4 neighborPosition = positions[neighborIndex];
        const float4 omegaJ = vorticity[neighborIndex];
        const float omegaJLength = sqrtf(
            omegaJ.x * omegaJ.x + omegaJ.y * omegaJ.y + omegaJ.z * omegaJ.z
        );
        const float3 displacement = {
            position.x - neighborPosition.x,
            position.y - neighborPosition.y,
            position.z - neighborPosition.z
        };
        const float3 gradient = spikyGradient(displacement, smoothingRadius);
        const float magnitudeDifference = omegaJLength - omegaILength;
        eta.x += magnitudeDifference * gradient.x;
        eta.y += magnitudeDifference * gradient.y;
        eta.z += magnitudeDifference * gradient.z;
    }

    constexpr float normalizationEpsilon = 1.0e-6f;
    const float etaLength = sqrtf(eta.x * eta.x + eta.y * eta.y + eta.z * eta.z);
    float3 normal = {0.0f, 0.0f, 0.0f};
    if (etaLength > normalizationEpsilon) {
        normal = {eta.x / etaLength, eta.y / etaLength, eta.z / etaLength};
    }

    const float3 confinement = {
        vorticityStrength * (normal.y * omegaI.z - normal.z * omegaI.y),
        vorticityStrength * (normal.z * omegaI.x - normal.x * omegaI.z),
        vorticityStrength * (normal.x * omegaI.y - normal.y * omegaI.x)
    };
    outputVelocities[index] = {
        velocity.x + dt * confinement.x,
        velocity.y + dt * confinement.y,
        velocity.z + dt * confinement.z,
        velocity.w
    };
}

__global__
void gatherSolverState(const std::uint32_t* sortedIndices,
                       const float4* positions, const float4* predictedPositions,
                       const float4* velocities, const float4* collisionInputVelocities,
                       const std::uint32_t* stableParticleIds,
                       float4* sortedPositions, float4* sortedPredictedPositions,
                       float4* sortedVelocities, float4* sortedCollisionInputVelocities,
                       std::uint32_t* sortedStableParticleIds,
                       std::size_t particleCount) {
    const std::size_t sortedIndex =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (sortedIndex >= particleCount)
        return;

    const std::uint32_t sourceIndex = sortedIndices[sortedIndex];
    sortedPositions[sortedIndex] = positions[sourceIndex];
    sortedPredictedPositions[sortedIndex] = predictedPositions[sourceIndex];
    sortedVelocities[sortedIndex] = velocities[sourceIndex];
    sortedCollisionInputVelocities[sortedIndex] = collisionInputVelocities[sourceIndex];
    sortedStableParticleIds[sortedIndex] = stableParticleIds[sourceIndex];
}

__global__
void initializeStableParticleIds(std::uint32_t* stableParticleIds,
                                 std::size_t particleCount) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < particleCount)
        stableParticleIds[index] = static_cast<std::uint32_t>(index);
}

__global__
void scatterFloat4ByStableId(const float4* values,
                             const std::uint32_t* stableParticleIds,
                             float4* valuesInOriginalOrder,
                             std::size_t particleCount) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < particleCount)
        valuesInOriginalOrder[stableParticleIds[index]] = values[index];
}
