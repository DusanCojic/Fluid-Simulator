#ifndef COLLIDERS_H
#define COLLIDERS_H


#include <cuda_runtime.h>

struct Container {
    float3 min;
    float3 max;
};

struct SphereCollider {
    float3 center;
    float radius;
};

struct BoxCollider {
    float3 center;
    float3 halfExtents;
};

struct PlaneCollider {
    float3 point;
    float3 normal;
};

#endif