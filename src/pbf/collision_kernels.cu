#include "pbf/collision_kernels.cuh"

#include <cfloat>

namespace {

// Projection back to a float position can leave a few ulps of penetration.
// Such residuals are not evidence that the collision iteration failed.
__device__ float projectionTolerance(float4 position, float3 origin, float radius) {
    float scale = fmaxf(radius, fmaxf(fabsf(position.x), fabsf(origin.x)));
    scale = fmaxf(scale, fmaxf(fabsf(position.y), fabsf(origin.y)));
    scale = fmaxf(scale, fmaxf(fabsf(position.z), fabsf(origin.z)));
    return 8.0f * FLT_EPSILON * scale;
}

__device__ void markCorrection(int* correctionFlag) {
    if (correctionFlag != nullptr)
        atomicExch(correctionFlag, 1);
}

__device__ void resolveContactVelocity(float3 normal, const float4& incomingVelocity,
    float restitution, float tangentScale, float4& velocity) {
    const float normalVelocity =
        velocity.x * normal.x + velocity.y * normal.y + velocity.z * normal.z;
    const float incomingNormalVelocity = incomingVelocity.x * normal.x +
        incomingVelocity.y * normal.y + incomingVelocity.z * normal.z;
    const float targetNormalVelocity = incomingNormalVelocity < 0.0f
        ? -restitution * incomingNormalVelocity
        : 0.0f;

    if (normalVelocity > targetNormalVelocity ||
        (normalVelocity == targetNormalVelocity && incomingNormalVelocity >= 0.0f))
        return;

    const float3 tangent = make_float3(
        velocity.x - normalVelocity * normal.x,
        velocity.y - normalVelocity * normal.y,
        velocity.z - normalVelocity * normal.z
    );
    velocity.x = targetNormalVelocity * normal.x + tangentScale * tangent.x;
    velocity.y = targetNormalVelocity * normal.y + tangentScale * tangent.y;
    velocity.z = targetNormalVelocity * normal.z + tangentScale * tangent.z;
}

} // namespace

__global__
void solveContainerKernel(float4* predictedPositions, std::size_t particleCount,
    const Container* container, float particleRadius, int* correctionFlag) {

    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (index >= particleCount)
        return;

    float4 position = predictedPositions[index];
    const float4 originalPosition = position;

    const float minX = container->min.x + particleRadius;
    const float maxX = container->max.x - particleRadius;

    const float minY = container->min.y + particleRadius;

    const float minZ = container->min.z + particleRadius;
    const float maxZ = container->max.z - particleRadius;

    // left/right walls
    position.x = fminf(fmaxf(position.x, minX), maxX);

    // floor only - container is open at the top
    position.y = fmaxf(position.y, minY);

    // front/back walls
    position.z = fminf(fmaxf(position.z, minZ), maxZ);

    if (position.x != originalPosition.x || position.y != originalPosition.y ||
        position.z != originalPosition.z) {
        markCorrection(correctionFlag);
    }

    predictedPositions[index] = position;
}

__global__
void solveSpheresKernel(float4* predictedPositions, std::size_t particleCount,
    const SphereCollider* spheres, std::size_t sphereCount, float particleRadius,
    int* correctionFlag) {

    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (index >= particleCount)
        return;

    float4 position = predictedPositions[index];

    for (std::size_t sphereIndex = 0; sphereIndex < sphereCount; ++sphereIndex) {
        const SphereCollider sphere = spheres[sphereIndex];
        const float minDistance = sphere.radius + particleRadius;
        const float3 offset = make_float3(
            position.x - sphere.center.x,
            position.y - sphere.center.y,
            position.z - sphere.center.z
        );
        const float distanceSquared = offset.x * offset.x + offset.y * offset.y + offset.z * offset.z;

        if (distanceSquared < minDistance * minDistance) {
            if (minDistance - sqrtf(distanceSquared) <=
                projectionTolerance(position, sphere.center, minDistance))
                continue;
            markCorrection(correctionFlag);

            if (distanceSquared == 0.0f) {
                position.x = sphere.center.x + minDistance;
            } else {
                const float scale = minDistance / sqrtf(distanceSquared);
                position.x = sphere.center.x + offset.x * scale;
                position.y = sphere.center.y + offset.y * scale;
                position.z = sphere.center.z + offset.z * scale;
            }
        }
    }

    predictedPositions[index] = position;
}

__global__
void solveBoxesKernel(float4* predictedPositions, std::size_t particleCount,
    const BoxCollider* boxes, std::size_t boxCount, float particleRadius,
    int* correctionFlag) {
        
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;

    if (index >= particleCount)
        return;

    float4 position = predictedPositions[index];

    for (std::size_t boxIndex = 0; boxIndex < boxCount; ++boxIndex) {
        const BoxCollider box = boxes[boxIndex];
        const float3 min = make_float3(
            box.center.x - box.halfExtents.x,
            box.center.y - box.halfExtents.y,
            box.center.z - box.halfExtents.z
        );
        const float3 max = make_float3(
            box.center.x + box.halfExtents.x,
            box.center.y + box.halfExtents.y,
            box.center.z + box.halfExtents.z
        );
        const float3 closest = make_float3(
            fminf(fmaxf(position.x, min.x), max.x),
            fminf(fmaxf(position.y, min.y), max.y),
            fminf(fmaxf(position.z, min.z), max.z)
        );
        const float3 offset = make_float3(
            position.x - closest.x,
            position.y - closest.y,
            position.z - closest.z
        );
        const float distanceSquared = offset.x * offset.x + offset.y * offset.y + offset.z * offset.z;

        if (distanceSquared > 0.0f && distanceSquared < particleRadius * particleRadius) {
            if (particleRadius - sqrtf(distanceSquared) <=
                projectionTolerance(position, closest, particleRadius))
                continue;
            markCorrection(correctionFlag);
            const float scale = particleRadius / sqrtf(distanceSquared);
            position.x = closest.x + offset.x * scale;
            position.y = closest.y + offset.y * scale;
            position.z = closest.z + offset.z * scale;
        } else if (distanceSquared == 0.0f) {
            markCorrection(correctionFlag);
            const float distanceX = fminf(position.x - min.x, max.x - position.x);
            const float distanceY = fminf(position.y - min.y, max.y - position.y);
            const float distanceZ = fminf(position.z - min.z, max.z - position.z);

            if (distanceX <= distanceY && distanceX <= distanceZ)
                position.x = position.x - min.x < max.x - position.x ? min.x - particleRadius : max.x + particleRadius;
            else if (distanceY <= distanceZ)
                position.y = position.y - min.y < max.y - position.y ? min.y - particleRadius : max.y + particleRadius;
            else
                position.z = position.z - min.z < max.z - position.z ? min.z - particleRadius : max.z + particleRadius;
        }
    }

    predictedPositions[index] = position;
}

__global__
void solvePlanesKernel(float4* predictedPositions, std::size_t particleCount,
    const PlaneCollider* planes, std::size_t planeCount, float particleRadius,
    int* correctionFlag) {

    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (index >= particleCount)
        return;

    float4 position = predictedPositions[index];

    for (std::size_t planeIndex = 0; planeIndex < planeCount; ++planeIndex) {
        const PlaneCollider plane = planes[planeIndex];
        const float normalLengthSquared = plane.normal.x * plane.normal.x +
            plane.normal.y * plane.normal.y + plane.normal.z * plane.normal.z;

        if (normalLengthSquared == 0.0f)
            continue;

        const float normalScale = 1.0f / sqrtf(normalLengthSquared);
        const float3 normal = make_float3(
            plane.normal.x * normalScale,
            plane.normal.y * normalScale,
            plane.normal.z * normalScale
        );
        const float distance = (position.x - plane.point.x) * normal.x +
            (position.y - plane.point.y) * normal.y +
            (position.z - plane.point.z) * normal.z;

        if (distance < particleRadius) {
            if (particleRadius - distance <=
                projectionTolerance(position, plane.point, particleRadius))
                continue;
            markCorrection(correctionFlag);
            const float correction = particleRadius - distance;
            position.x += normal.x * correction;
            position.y += normal.y * correction;
            position.z += normal.z * correction;
        }
    }

    predictedPositions[index] = position;
}

__global__
void resolveVelocitiesKernel(const float4* positions, const float4* incomingVelocities,
    float4* velocities,
    std::size_t particleCount, const Container* container,
    const SphereCollider* spheres, std::size_t sphereCount,
    const BoxCollider* boxes, std::size_t boxCount,
    const PlaneCollider* planes, std::size_t planeCount,
    float particleRadius, float restitution, float friction) {

    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (index >= particleCount)
        return;

    const float4 position = positions[index];
    const float4 incomingVelocity = incomingVelocities[index];
    float4 velocity = velocities[index];
    const float tangentScale = 1.0f - friction;
    constexpr float collisionEpsilon = 1e-4f;

    const float minX = container->min.x + particleRadius;
    const float maxX = container->max.x - particleRadius;
    const float minY = container->min.y + particleRadius;
    const float minZ = container->min.z + particleRadius;
    const float maxZ = container->max.z - particleRadius;

    if (position.x <= minX + collisionEpsilon)
        resolveContactVelocity(make_float3(1.0f, 0.0f, 0.0f), incomingVelocity,
            restitution, tangentScale, velocity);
    else if (position.x >= maxX - collisionEpsilon)
        resolveContactVelocity(make_float3(-1.0f, 0.0f, 0.0f), incomingVelocity,
            restitution, tangentScale, velocity);

    if (position.y <= minY + collisionEpsilon)
        resolveContactVelocity(make_float3(0.0f, 1.0f, 0.0f), incomingVelocity,
            restitution, tangentScale, velocity);

    if (position.z <= minZ + collisionEpsilon)
        resolveContactVelocity(make_float3(0.0f, 0.0f, 1.0f), incomingVelocity,
            restitution, tangentScale, velocity);
    else if (position.z >= maxZ - collisionEpsilon)
        resolveContactVelocity(make_float3(0.0f, 0.0f, -1.0f), incomingVelocity,
            restitution, tangentScale, velocity);

    for (std::size_t sphereIndex = 0; sphereIndex < sphereCount; ++sphereIndex) {
        const SphereCollider sphere = spheres[sphereIndex];
        const float minDistance = sphere.radius + particleRadius;
        const float3 offset = make_float3(
            position.x - sphere.center.x,
            position.y - sphere.center.y,
            position.z - sphere.center.z
        );
        const float distanceSquared = offset.x * offset.x + offset.y * offset.y + offset.z * offset.z;

        const float contactDistance = minDistance + collisionEpsilon;

        if (distanceSquared > 0.0f && distanceSquared <= contactDistance * contactDistance) {
            const float scale = 1.0f / sqrtf(distanceSquared);
            const float3 normal = make_float3(offset.x * scale, offset.y * scale, offset.z * scale);
            resolveContactVelocity(normal, incomingVelocity, restitution, tangentScale, velocity);
        }
    }

    for (std::size_t boxIndex = 0; boxIndex < boxCount; ++boxIndex) {
        const BoxCollider box = boxes[boxIndex];
        const float3 min = make_float3(
            box.center.x - box.halfExtents.x,
            box.center.y - box.halfExtents.y,
            box.center.z - box.halfExtents.z
        );
        const float3 max = make_float3(
            box.center.x + box.halfExtents.x,
            box.center.y + box.halfExtents.y,
            box.center.z + box.halfExtents.z
        );
        const float3 closest = make_float3(
            fminf(fmaxf(position.x, min.x), max.x),
            fminf(fmaxf(position.y, min.y), max.y),
            fminf(fmaxf(position.z, min.z), max.z)
        );
        const float3 offset = make_float3(
            position.x - closest.x,
            position.y - closest.y,
            position.z - closest.z
        );
        
        const float distanceSquared = offset.x * offset.x + offset.y * offset.y + offset.z * offset.z;
        float3 normal;

        if (distanceSquared > 0.0f && distanceSquared <= (particleRadius + collisionEpsilon) * (particleRadius + collisionEpsilon)) {
            const float scale = 1.0f / sqrtf(distanceSquared);
            normal = make_float3(offset.x * scale, offset.y * scale, offset.z * scale);
        }
        else if (distanceSquared == 0.0f) {
            const float distanceX = fminf(position.x - min.x, max.x - position.x);
            const float distanceY = fminf(position.y - min.y, max.y - position.y);
            const float distanceZ = fminf(position.z - min.z, max.z - position.z);

            if (distanceX <= distanceY && distanceX <= distanceZ)
                normal = make_float3(position.x - min.x < max.x - position.x ? -1.0f : 1.0f, 0.0f, 0.0f);
            else if (distanceY <= distanceZ)
                normal = make_float3(0.0f, position.y - min.y < max.y - position.y ? -1.0f : 1.0f, 0.0f);
            else
                normal = make_float3(0.0f, 0.0f, position.z - min.z < max.z - position.z ? -1.0f : 1.0f);
        } 
        else
            continue;

        resolveContactVelocity(normal, incomingVelocity, restitution, tangentScale, velocity);
    }

    for (std::size_t planeIndex = 0; planeIndex < planeCount; ++planeIndex) {
        const PlaneCollider plane = planes[planeIndex];
        const float normalLengthSquared = plane.normal.x * plane.normal.x +
            plane.normal.y * plane.normal.y + plane.normal.z * plane.normal.z;

        if (normalLengthSquared == 0.0f)
            continue;

        const float normalScale = 1.0f / sqrtf(normalLengthSquared);
        const float3 normal = make_float3(
            plane.normal.x * normalScale,
            plane.normal.y * normalScale,
            plane.normal.z * normalScale
        );
        
        const float distance = (position.x - plane.point.x) * normal.x +
            (position.y - plane.point.y) * normal.y +
            (position.z - plane.point.z) * normal.z;

        if (distance <= particleRadius + collisionEpsilon)
            resolveContactVelocity(normal, incomingVelocity, restitution, tangentScale, velocity);
    }

    velocities[index] = velocity;
}
