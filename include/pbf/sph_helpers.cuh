#ifndef SPH_HELPERS_H
#define SPH_HELPERS_H


#include <cuda_runtime.h>

namespace sph {

constexpr float pi = 3.14159265358979323846f;

} // namespace sph

// calculates how much one particle contributes to the estimated density around
// another based on the distance between them
__device__
inline float poly6(float3 displacement, float smoothingRadius) {
    const float distanceSquared =
        displacement.x * displacement.x +
        displacement.y * displacement.y +
        displacement.z * displacement.z;

    const float radiusSquared =
        smoothingRadius * smoothingRadius;

    if (distanceSquared >= radiusSquared)
        return 0.0f;

    // Normalize the distance first to avoid overflowing intermediate powers of h.
    const float normalization =
        315.0f / (64.0f * sph::pi) / smoothingRadius / smoothingRadius / smoothingRadius;

    const float radiusDifference =
        1.0f - distanceSquared / radiusSquared;

    const float radiusDifferenceCubed =
        radiusDifference *
        radiusDifference *
        radiusDifference;

    return normalization * radiusDifferenceCubed;
}


// Computes the Spiky kernel gradient, which describes how the smoothing influence
// changes with the distance and direction between two neighboring particles.
__device__
inline float3 spikyGradient(float3 displacement, float smoothingRadius) {
    const float distanceSquared =
        displacement.x * displacement.x +
        displacement.y * displacement.y +
        displacement.z * displacement.z;

    const float distance = sqrtf(distanceSquared);

    if (distance <= 0.0f || distance >= smoothingRadius)
        return {0.0f, 0.0f, 0.0f};

    // Normalize the distance first to avoid overflowing intermediate powers of h.
    const float normalization =
        -45.0f / sph::pi / smoothingRadius / smoothingRadius / smoothingRadius / smoothingRadius;

    const float radiusDifference =
        1.0f - distance / smoothingRadius;

    const float distanceTerm =
        radiusDifference * radiusDifference;

    const float3 direction = {
        displacement.x / distance,
        displacement.y / distance,
        displacement.z / distance
    };

    const float magnitude =
        normalization * distanceTerm;

    return {
        direction.x * magnitude,
        direction.y * magnitude,
        direction.z * magnitude
    };
}


#endif 
