#ifndef PBF_SOLVER_KERNELS_H
#define PBF_SOLVER_KERNELS_H


#include "pbf/sph_helpers.cuh"

#include <cuda_runtime.h>
#include <cstdint>
#include "simulation_params.hpp"

// neighborsCount must be in [0, maxNeighbors] for every particle. A larger
// count means findNeighbors truncated the fixed-stride list; solver kernels
// reject that particle rather than compute from an incomplete neighborhood.
// Neighbor lists are slot-major, with particles contiguous for each neighbor
// offset: neighbors[offset * neighborParticleStride + particle].
__global__
void computeLambda(const float4* predictedPositions, const uint32_t* neighbors,
                   const int* neighborsCount, int maxNeighbors,
                   size_t particleCount, float smoothingRadius,
                   float particleMass, float restDensity, float epsilon,
                   float* lambdas, size_t neighborParticleStride);


__device__
inline float computeArtificialPressure(float distanceSquared, float smoothingRadius, float scorrK,
                                       int scorrN, float scorrDeltaQ) {
    // Macklin & Mueller, Eq. (13):
    // s_corr = -k * (W(p_i - p_j, h) / W(delta_q, h))^n.
    if (scorrK == 0.0f || scorrN <= 0)
        return 0.0f;

    const float radiusSquared = smoothingRadius * smoothingRadius;
    const float referenceDistanceSquared = scorrDeltaQ * scorrDeltaQ;

    if (distanceSquared >= radiusSquared ||
        referenceDistanceSquared >= radiusSquared)
        return 0.0f;

    // The Poly6 normalization cancels in the quotient. Each remaining term
    // is cubed because Poly6 is proportional to (h^2 - r^2)^3.
    const float kernelRatio =
        powf((radiusSquared - distanceSquared) /
                 (radiusSquared - referenceDistanceSquared),
             3.0f);

    return -scorrK * powf(kernelRatio, static_cast<float>(scorrN));
}

__global__
void computeDeltaPosition(const float4* predictedPosition, const uint32_t* neighbors, const int* neighborsCount, int maxNeighbors, 
    const float* lambdas, size_t particleCount, float smoothingRadius, float particleMass, float restDensity,
    float scorrK, int scorrN, float scorrDeltaQ, float4* deltaPositions,
    size_t neighborParticleStride);


__global__
void applyDeltaPosition(float4* predictedPosition, const float4* deltaPositions, size_t particleCount);

__global__
void updateVelocityAndPosition(float4* positions, const float4* predictedPositions, float4* velocities, 
    std::size_t particleCount, float inverseDt);

// Reads reconstructed velocities and writes XSPH-smoothed velocities to a
// separate buffer so all particles use the same input velocity state.
__global__
void applyXsphViscosity(const float4* predictedPositions, const uint32_t* neighbors,
                        const int* neighborsCount, int maxNeighbors,
                        const float4* inputVelocities, float4* outputVelocities,
                        std::size_t particleCount, float smoothingRadius,
                        float xsphViscosity, std::size_t neighborParticleStride);

// Computes omega_i = sum_j grad_i W(p_i - p_j) x (v_j - v_i).  This must
// finish before applyVorticityConfinement consumes neighboring omega values.
__global__
void computeVorticity(const float4* positions, const float4* velocities,
                      const uint32_t* neighbors, const int* neighborsCount,
                      int maxNeighbors, std::size_t particleCount,
                      float smoothingRadius, float4* vorticity,
                      std::size_t neighborParticleStride);

// Uses eta_i = sum_j (|omega_j| - |omega_i|) grad W(p_i - p_j), a
// difference-based SPH approximation of grad |omega|, then writes a Jacobi
// velocity correction to outputVelocities.
__global__
void applyVorticityConfinement(const float4* positions, const uint32_t* neighbors,
                               const int* neighborsCount, int maxNeighbors,
                               const float4* vorticity,
                               const float4* inputVelocities,
                               float4* outputVelocities,
                               std::size_t particleCount,
                               float smoothingRadius, float dt,
                               float vorticityStrength,
                               std::size_t neighborParticleStride);

// Applies the grid's sorted-slot -> previous-working-slot permutation to all
// fields whose particle identity must survive a solver reorder.
__global__
void gatherSolverState(const std::uint32_t* sortedIndices,
                       const float4* positions, const float4* predictedPositions,
                       const float4* velocities, const float4* collisionInputVelocities,
                       const std::uint32_t* stableParticleIds,
                       float4* sortedPositions, float4* sortedPredictedPositions,
                       float4* sortedVelocities, float4* sortedCollisionInputVelocities,
                       std::uint32_t* sortedStableParticleIds,
                       std::size_t particleCount);

__global__
void initializeStableParticleIds(std::uint32_t* stableParticleIds,
                                 std::size_t particleCount);

__global__
void scatterFloat4ByStableId(const float4* values,
                             const std::uint32_t* stableParticleIds,
                             float4* valuesInOriginalOrder,
                             std::size_t particleCount);


#endif
