#ifndef INTEGRATION_H
#define INTEGRATION_H


// kernel predicts new position per particle based on initial position and velocity, influenced by gravity
__global__
void integrate(const float4* position, float4* predictedPosition, float4* velocity,
    int particleCount, float dt, float3 gravity
);


#endif