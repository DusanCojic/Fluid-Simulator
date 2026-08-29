#ifndef PARTICLE_DATA_H
#define PARTICLE_DATA_H


#include "pbf/cuda_buffer.hpp"
#include <cuda_runtime.h>

struct ParticleData {
    CudaBuffer<float4> position;
    CudaBuffer<float4> predictedPosition;
    CudaBuffer<float4> velocity;

    CudaBuffer<float> density;
    CudaBuffer<float> lambda;

    CudaBuffer<float4> deltaPosition;
};


#endif