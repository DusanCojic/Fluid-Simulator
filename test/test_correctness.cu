#include "pbf/pbf_solver.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <vector>

namespace {

constexpr double pi = 3.14159265358979323846;
using Vector = std::array<double, 3>;

Vector difference(float4 a, float4 b) {
    return {double(a.x) - b.x, double(a.y) - b.y, double(a.z) - b.z};
}

double squared(Vector a) {
    return a[0]*a[0] + a[1]*a[1] + a[2]*a[2];
}

double weight(Vector d, double h) {
    const double r2 = squared(d);
    return r2 >= h*h ? 0.0 : 315.0 / (64.0*pi*std::pow(h, 9)) * std::pow(h*h-r2, 3);
}

Vector gradient(Vector d, double h) {
    const double r = std::sqrt(squared(d));
    if (r == 0 || r >= h) return {};
    const double scale = -45.0 / (pi*std::pow(h, 6)) * (h-r)*(h-r) / r;
    return {scale*d[0], scale*d[1], scale*d[2]};
}

template<class T>
std::vector<T> readDevice(const T* data, std::size_t count) {
    std::vector<T> result(count);
    const auto error = cudaMemcpy(result.data(), data, count*sizeof(T), cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
    return result;
}

// An independent all-pairs oracle also checks every grid range and sorted index.
void checkNeighborhoods(const std::vector<float4>& positions, float h,
                        SpatialGrid& grid, CudaBuffer<float4>& devicePositions,
                        CudaBuffer<uint32_t>& neighbors, CudaBuffer<int>& counts,
                        int capacity) {
    const std::size_t n = positions.size();
    devicePositions.copyFromHostToDevice(positions.data(), n);
    grid.build(devicePositions.data(), n);
    const auto keys = readDevice(grid.sortedKeys(), n);
    const auto indices = readDevice(grid.sortedIndices(), n);
    std::vector<float4> sortedPositions(n);
    for (std::size_t i = 0; i < n; ++i)
        sortedPositions[i] = positions[indices[i]];
    CudaBuffer<float4> sortedDevicePositions(n);
    CudaBuffer<uint32_t> sortedNeighbors(n * capacity);
    CudaBuffer<int> sortedCounts(n);
    sortedDevicePositions.copyFromHostToDevice(sortedPositions.data(), n);
    findNeighbors<<<(n+255)/256, 256>>>(sortedDevicePositions.data(),
        grid.cellStart(), grid.cellEnd(), grid.gridSize(), grid.minBounds(),
        grid.cellSize(), n, h, sortedNeighbors.data(), sortedCounts.data(),
        capacity, n);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    const auto starts = readDevice(grid.cellStart(), grid.numCells());
    const auto ends = readDevice(grid.cellEnd(), grid.numCells());
    const auto sortedHostCounts = readDevice(sortedCounts.data(), n);
    const auto sortedHostNeighbors = readDevice(sortedNeighbors.data(), n*capacity);
    std::vector<int> hostCounts(n);
    std::vector<uint32_t> hostNeighbors(n * capacity);
    for (std::size_t sortedParticle = 0; sortedParticle < n; ++sortedParticle) {
        ASSERT_LE(sortedHostCounts[sortedParticle], capacity);
        const std::size_t originalParticle = indices[sortedParticle];
        hostCounts[originalParticle] = sortedHostCounts[sortedParticle];
        for (int offset = 0; offset < sortedHostCounts[sortedParticle]; ++offset) {
            const uint32_t sortedNeighbor =
                sortedHostNeighbors[offset * n + sortedParticle];
            hostNeighbors[offset * n + originalParticle] = indices[sortedNeighbor];
        }
    }
    counts.copyFromHostToDevice(hostCounts.data(), n);
    neighbors.copyFromHostToDevice(hostNeighbors.data(), n * capacity);
    std::vector<bool> seen(n, false);
    for (std::size_t i = 0; i < n; ++i) {
        ASSERT_LT(keys[i], grid.numCells());
        ASSERT_LT(indices[i], n);
        ASSERT_FALSE(seen[indices[i]]);
        seen[indices[i]] = true;
        if (i) ASSERT_LE(keys[i-1], keys[i]);
        ASSERT_LE(starts[keys[i]], int(i));
        ASSERT_GT(ends[keys[i]], int(i));
        ASSERT_LE(ends[keys[i]], int(n));
        ASSERT_GE(hostCounts[i], 0);
        ASSERT_LE(hostCounts[i], capacity);
        std::vector<uint32_t> expected;
        for (std::size_t j = 0; j < n; ++j) {
            const float x = positions[i].x - positions[j].x;
            const float y = positions[i].y - positions[j].y;
            const float z = positions[i].z - positions[j].z;
            if (i != j && x*x+y*y+z*z <= h*h) expected.push_back(uint32_t(j));
        }
        std::vector<uint32_t> actual;
        for (int offset = 0; offset < hostCounts[i]; ++offset)
            actual.push_back(hostNeighbors[offset * n + i]);
        std::sort(actual.begin(), actual.end());
        ASSERT_EQ(actual, expected) << "particle " << i;
    }
    for (std::size_t cell = 0; cell < grid.numCells(); ++cell) {
        if (starts[cell] == -1) { ASSERT_EQ(ends[cell], -1); continue; }
        ASSERT_GE(starts[cell], 0);
        ASSERT_GT(ends[cell], starts[cell]);
        ASSERT_LE(ends[cell], int(n));
        for (int i = starts[cell]; i < ends[cell]; ++i) ASSERT_EQ(keys[i], cell);
    }
}

} // namespace

TEST(CorrectnessReferenceTest, CoupledDensityLambdaAndCorrectionMatchDoublePrecisionOracle) {
    // Multiple non-collinear gradients, a coincident pair, and an isolated particle.
    const std::vector<float4> positions = {
        {0.49f,0.49f,0.49f,1}, {0.51f,0.51f,0.51f,2},
        {0.25f,0.5f,0.5f,3}, {0.75f,0.5f,0.5f,4},
        {0.5f,0.25f,0.5f,5}, {0.5f,0.75f,0.5f,6},
        {0.5f,0.5f,0.25f,7}, {0.5f,0.5f,0.75f,8},
        {0.49f,0.49f,0.49f,9}, {3,3,3,10}
    };
    const std::size_t n = positions.size();
    constexpr int capacity = 16;
    constexpr float h = 0.7f, mass = 0.4f, rho0 = 2.0f, epsilon = 0.03f;
    constexpr float k = 0.002f, dq = 0.2f;
    SpatialGrid grid;
    grid.initialize(n, {0,0,0}, {4,4,4}, 0.25f);
    CudaBuffer<float4> dp(n), delta(n);
    CudaBuffer<uint32_t> neighbors(n*capacity);
    CudaBuffer<int> counts(n);
    CudaBuffer<float> lambda(n);
    checkNeighborhoods(positions, h, grid, dp, neighbors, counts, capacity);
    ASSERT_FALSE(HasFatalFailure());
    computeLambda<<<1,32>>>(dp.data(), neighbors.data(), counts.data(), capacity,
        n, h, mass, rho0, epsilon, lambda.data(), n);
    computeDeltaPosition<<<1,32>>>(dp.data(), neighbors.data(), counts.data(), capacity,
        lambda.data(), n, h, mass, rho0, k, 4, dq, delta.data(), n);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    const auto lambdas = readDevice(lambda.data(), n);
    const auto deltas = readDevice(delta.data(), n);
    std::vector<double> referenceLambda(n);
    for (std::size_t i = 0; i < n; ++i) {
        double rho = 0, denominator = epsilon;
        Vector center{};
        for (std::size_t j = 0; j < n; ++j) {
            const auto d = difference(positions[i], positions[j]);
            rho += mass * weight(d,h);
            auto g = gradient(d,h);
            for (int a = 0; a < 3; ++a) { g[a] *= double(mass)/rho0; center[a] += g[a]; }
            denominator += squared(g);
        }
        denominator += squared(center);
        const double c = rho/rho0-1;
        referenceLambda[i] = -c/denominator;
        EXPECT_NEAR(lambdas[i], referenceLambda[i], 2e-6*std::max(1.0,std::abs(referenceLambda[i])));
    }
    for (std::size_t i = 0; i < n; ++i) {
        Vector correction{};
        for (std::size_t j = 0; j < n; ++j) {
            if (i == j) continue;
            const auto d = difference(positions[i],positions[j]);
            const auto g = gradient(d,h);
            const double scorr = -k*std::pow(weight(d,h)/weight({dq,0,0},h),4);
            for (int a = 0; a < 3; ++a)
                correction[a] += double(mass)/rho0*(referenceLambda[i]+referenceLambda[j]+scorr)*g[a];
        }
        EXPECT_NEAR(deltas[i].x, correction[0], 2e-5);
        EXPECT_NEAR(deltas[i].y, correction[1], 2e-5);
        EXPECT_NEAR(deltas[i].z, correction[2], 2e-5);
        EXPECT_EQ(deltas[i].w, 0.0f);
    }
}

TEST(CorrectnessStabilityTest, DenseFluidWithSphereAndBoxFor1200Frames) {
    constexpr int side = 10, frames = 1200, capacity = 256;
    constexpr std::size_t n = side*side*side;
    SimulationParams params{};
    params.dt = 1.0f/60.0f;
    params.substeps = 2;
    params.solverIterations = 4;
    params.restDensity = 1000;
    params.particleMass = 1000*0.12f*0.12f*0.12f;
    params.particleRadius = 0.05f;
    params.smoothingRadius = 0.24f;
    params.lambdaEpsilon = 100;
    params.gravity = {0,-9.81f,0};
    params.collisionFriction = 0.03f;
    params.scorrK = 0.0001f;
    params.scorrN = 4;
    params.scorrDeltaQ = 0.072f;
    params.xsphViscosity = 0.00001f;
    params.vorticityStrength = 0.00001f;
    const SphereCollider sphere{{0.8f,0.45f,1.0f},0.22f};
    const BoxCollider box{{1.9f,0.35f,1.0f},{0.22f,0.35f,0.4f}};
    PBFSolver solver;
    solver.initialize(n, {0,0,0}, {3,3,2}, params);
    solver.setSpheres({sphere});
    solver.setBoxes({box});
    std::vector<float4> positions, velocities(n, {1,0,0.15f,0});
    for (int z=0; z<side; ++z) for (int y=0; y<side; ++y) for (int x=0; x<side; ++x)
        positions.push_back({0.35f+x*0.12f,1.1f+y*0.12f,0.4f+z*0.12f,1});
    solver.setParticles(positions.data(), velocities.data(), n);
    SpatialGrid grid;
    grid.initialize(n, {0,0,0}, {3,3,2}, params.smoothingRadius);
    CudaBuffer<float4> dp(n);
    CudaBuffer<uint32_t> neighbors(n*capacity);
    CudaBuffer<int> counts(n);
    double peakSpeed = 0, finalRmsSpeed = 0, finalMeanHeight = 0;
    int sphereContacts = 0, boxContacts = 0;
    constexpr float tolerance = 4e-6f;
    for (int frame=0; frame<frames; ++frame) {
        SCOPED_TRACE(frame);
        ASSERT_NO_THROW(solver.step());
        solver.copyPositionsToHost(positions.data(), n);
        solver.copyVelocitiesToHost(velocities.data(), n);
        double speedSquaredSum = 0, heightSum = 0;
        for (std::size_t i=0; i<n; ++i) {
            const auto p=positions[i], v=velocities[i];
            ASSERT_TRUE(std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z));
            ASSERT_TRUE(std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z));
            ASSERT_GE(p.x, params.particleRadius-tolerance);
            ASSERT_LE(p.x, 3-params.particleRadius+tolerance);
            ASSERT_GE(p.y, params.particleRadius-tolerance);
            ASSERT_LT(p.y, 20.0f); // catastrophic launch, not an artificial closed top
            ASSERT_GE(p.z, params.particleRadius-tolerance);
            ASSERT_LE(p.z, 2-params.particleRadius+tolerance);
            const double distance = std::sqrt(squared(difference(p,
                {sphere.center.x,sphere.center.y,sphere.center.z,0})));
            ASSERT_GE(distance, sphere.radius+params.particleRadius-tolerance);
            if (distance < sphere.radius+params.particleRadius+1e-4f) ++sphereContacts;
            const double dx=std::max(0.0, std::abs(double(p.x)-box.center.x)-box.halfExtents.x);
            const double dy=std::max(0.0, std::abs(double(p.y)-box.center.y)-box.halfExtents.y);
            const double dz=std::max(0.0, std::abs(double(p.z)-box.center.z)-box.halfExtents.z);
            const double boxDistance=std::sqrt(dx*dx+dy*dy+dz*dz);
            ASSERT_GE(boxDistance, params.particleRadius-tolerance);
            if (boxDistance < params.particleRadius+1e-4f) ++boxContacts;
            const double speed2=squared({v.x,v.y,v.z});
            peakSpeed=std::max(peakSpeed,std::sqrt(speed2));
            ASSERT_LT(speed2, 50.0*50.0);
            speedSquaredSum+=speed2;
            heightSum+=p.y;
        }
        finalRmsSpeed=std::sqrt(speedSquaredSum/n);
        finalMeanHeight=heightSum/n;
        if (frame%100 == 0 || frame == frames-1) {
            checkNeighborhoods(positions,params.smoothingRadius,grid,dp,neighbors,counts,capacity);
            ASSERT_FALSE(HasFatalFailure());
        }
    }
    EXPECT_GT(sphereContacts,0);
    EXPECT_GT(boxContacts,0);
    EXPECT_LT(finalMeanHeight,1.0);
    EXPECT_LT(finalRmsSpeed,1.0);
    std::cout << "Stability: particles=" << n << " frames=" << frames
        << " substeps=" << frames*params.substeps << " peak_speed=" << peakSpeed
        << " final_rms_speed=" << finalRmsSpeed << " final_mean_height=" << finalMeanHeight
        << " sphere_contacts=" << sphereContacts << " box_contacts=" << boxContacts << '\n';
}
