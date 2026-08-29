#include "pbf/particle_data.hpp"
#include "pbf/simulation_params.hpp"
#include "pbf/integration.cuh"

#include <cuda_runtime.h>

__global__
void integrate(const float4* position, float4* predictedPosition, float4* velocity,
    int particleCount, float dt, float3 gravity
) {
    // calculate particle index
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= particleCount) return;

    float4 v = velocity[index];
    float4 p = position[index];

    // calculate new velocity
    v.x = v.x + gravity.x * dt;
    v.y = v.y + gravity.y * dt;
    v.z = v.z + gravity.z * dt;

    // predict position
    float4 predicted;
    predicted.x = p.x + v.x * dt;
    predicted.y = p.y + v.y * dt;
    predicted.z = p.z + v.z * dt;

    // update global velocity and predictedPosition
    velocity[index] = v;
    predictedPosition[index] = predicted;
}