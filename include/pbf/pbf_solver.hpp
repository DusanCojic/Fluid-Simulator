#ifndef PBF_SOLVER_H
#define PBF_SOLVER_H


#include "pbf/pbf_solver_kernels.cuh"
#include "pbf/particle_data.hpp"
#include "pbf/simulation_params.hpp"
#include "pbf/integration.cuh"
#include "pbf/neighbors.cuh"
#include "pbf/cuda_buffer.hpp"
#include "pbf/spatial_grid.cuh"
#include "pbf/collision_system.hpp"

class PBFSolver {
public:
    void initialize(size_t maxParticles, float3 minBounds, float3 maxBounds,
                    const SimulationParams& params);

    void setParticles(const float4* positions, const float4* velocities,
                      size_t particleCount);

    void setSpheres(const std::vector<SphereCollider>& spheres);
    void setBoxes(const std::vector<BoxCollider>& boxes);
    void setPlanes(const std::vector<PlaneCollider>& planes);

    void step();

    // Advances one simulation frame.  A frame consists of params.substeps
    // substeps, each with params.solverIterations constraint iterations.
    void run();
    void run(std::size_t frameCount);

    void copyPositionsToHost(float4* positions, size_t particleCount) const;
    void copyVelocitiesToHost(float4* velocities, size_t particleCount) const;

private:
    size_t maxParticles_ = 0;
    size_t particleCount_ = 0;

    SimulationParams params_{};

    SpatialGrid spatialGrid_;

    CudaBuffer<float4> positions_;
    CudaBuffer<float4> predictedPositions_;
    CudaBuffer<float4> velocities_;
    CudaBuffer<float4> xsphVelocities_;
    CudaBuffer<float4> vorticity_;

    CudaBuffer<uint32_t> neighbors_;
    CudaBuffer<int> neighborsCount_;

    CudaBuffer<float> density_;
    CudaBuffer<float> constraints_;
    CudaBuffer<float> lambda_;
    CudaBuffer<float4> deltaPosition_;

    CollisionSystem _collisionSystem;
};


#endif
