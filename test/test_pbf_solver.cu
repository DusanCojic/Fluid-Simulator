#include "pbf/pbf_solver.cuh"

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

constexpr float pi = 3.14159265358979323846f;

SimulationParams validParams() {
    SimulationParams params{};
    params.dt = 0.1f;
    params.restDensity = 315.0f / (64.0f * pi);
    params.particleMass = 1.0f;
    params.smoothingRadius = 1.0f;
    params.lambdaEpsilon = 0.01f;
    params.solverIterations = 1;
    params.substeps = 1;
    params.gravity = make_float3(0.0f, -9.8f, 0.0f);
    params.scorrK = 0.0f;
    params.scorrN = 0.0f;
    params.scorrDeltaQ = 0.0f;
    params.xsphViscosity = 0.0f;
    params.vorticityStrength = 0.0f;
    return params;
}

float selfDensity(float particleMass, float smoothingRadius) {
    return particleMass * 315.0f /
        (64.0f * pi * std::pow(smoothingRadius, 3.0f));
}

void initializeSolver(PBFSolver& solver, std::size_t capacity,
                      const SimulationParams& params) {
    solver.initialize(
        capacity,
        make_float3(-2.0f, -2.0f, -2.0f),
        make_float3(12.0f, 12.0f, 12.0f),
        params
    );
}

void expectFloat4Near(const float4& actual, const float4& expected,
                      float tolerance, std::size_t particleIndex) {
    EXPECT_NEAR(actual.x, expected.x, tolerance) << "particle " << particleIndex;
    EXPECT_NEAR(actual.y, expected.y, tolerance) << "particle " << particleIndex;
    EXPECT_NEAR(actual.z, expected.z, tolerance) << "particle " << particleIndex;
    EXPECT_NEAR(actual.w, expected.w, tolerance) << "particle " << particleIndex;
}

void integrateReference(float4& position, float4& velocity, float dt,
                        float3 gravity, int substeps) {
    const float substepDt = dt / static_cast<float>(substeps);

    for (int substep = 0; substep < substeps; ++substep) {
        velocity.x += gravity.x * substepDt;
        velocity.y += gravity.y * substepDt;
        velocity.z += gravity.z * substepDt;

        position.x += velocity.x * substepDt;
        position.y += velocity.y * substepDt;
        position.z += velocity.z * substepDt;
    }
}

} // namespace

TEST(PBFSolverInitializationTest, RejectsCallsThatRequireInitialization) {
    PBFSolver solver;
    const float4 particle = make_float4(0.0f, 0.0f, 0.0f, 1.0f);

    EXPECT_THROW(solver.setParticles(&particle, &particle, 1), std::logic_error);
    EXPECT_THROW(solver.step(), std::logic_error);
    EXPECT_THROW(solver.run(), std::logic_error);
    EXPECT_THROW(solver.copyPositionsToHost(nullptr, 1), std::out_of_range);
    EXPECT_THROW(solver.copyVelocitiesToHost(nullptr, 1), std::out_of_range);
}

TEST(PBFSolverInitializationTest, RejectsInvalidCapacityAndBounds) {
    const SimulationParams params = validParams();

    PBFSolver zeroCapacitySolver;
    EXPECT_THROW(
        zeroCapacitySolver.initialize(
            0,
            make_float3(0.0f, 0.0f, 0.0f),
            make_float3(1.0f, 1.0f, 1.0f),
            params
        ),
        std::invalid_argument
    );

    PBFSolver excessiveCapacitySolver;
    const std::size_t excessiveCapacity =
        static_cast<std::size_t>(std::numeric_limits<int>::max()) + 1U;
    EXPECT_THROW(
        excessiveCapacitySolver.initialize(
            excessiveCapacity,
            make_float3(0.0f, 0.0f, 0.0f),
            make_float3(1.0f, 1.0f, 1.0f),
            params
        ),
        std::length_error
    );

    PBFSolver invalidBoundsSolver;
    EXPECT_THROW(
        invalidBoundsSolver.initialize(
            8,
            make_float3(1.0f, 0.0f, 0.0f),
            make_float3(1.0f, 2.0f, 2.0f),
            params
        ),
        std::invalid_argument
    );
}

TEST(PBFSolverInitializationTest, RejectsInvalidSimulationParameters) {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float infinity = std::numeric_limits<float>::infinity();

    auto expectInvalid = [](const SimulationParams& params) {
        PBFSolver solver;
        EXPECT_THROW(
            solver.initialize(
                8,
                make_float3(0.0f, 0.0f, 0.0f),
                make_float3(2.0f, 2.0f, 2.0f),
                params
            ),
            std::invalid_argument
        );
    };

    SimulationParams params = validParams();
    params.dt = 0.0f;
    expectInvalid(params);
    params = validParams();
    params.dt = nan;
    expectInvalid(params);

    params = validParams();
    params.restDensity = -1.0f;
    expectInvalid(params);
    params = validParams();
    params.restDensity = infinity;
    expectInvalid(params);

    params = validParams();
    params.particleMass = 0.0f;
    expectInvalid(params);
    params = validParams();
    params.particleMass = nan;
    expectInvalid(params);

    params = validParams();
    params.smoothingRadius = -1.0f;
    expectInvalid(params);
    params = validParams();
    params.smoothingRadius = infinity;
    expectInvalid(params);

    params = validParams();
    params.lambdaEpsilon = -0.01f;
    expectInvalid(params);
    params = validParams();
    params.lambdaEpsilon = nan;
    expectInvalid(params);

    params = validParams();
    params.gravity.x = infinity;
    expectInvalid(params);
    params = validParams();
    params.gravity.y = nan;
    expectInvalid(params);
    params = validParams();
    params.gravity.z = -infinity;
    expectInvalid(params);

    params = validParams();
    params.solverIterations = 0;
    expectInvalid(params);
    params = validParams();
    params.solverIterations = -1;
    expectInvalid(params);

    params = validParams();
    params.substeps = 0;
    expectInvalid(params);
    params = validParams();
    params.substeps = -1;
    expectInvalid(params);
}

TEST(PBFSolverParticleDataTest, RejectsInvalidParticleInput) {
    PBFSolver solver;
    const SimulationParams params = validParams();
    initializeSolver(solver, 2, params);

    const float4 position = make_float4(1.0f, 2.0f, 3.0f, 1.0f);
    const float4 velocity = make_float4(0.0f, 0.0f, 0.0f, 7.0f);

    EXPECT_THROW(solver.setParticles(&position, &velocity, 3), std::out_of_range);
    EXPECT_THROW(solver.setParticles(nullptr, &velocity, 1), std::invalid_argument);
    EXPECT_THROW(solver.setParticles(&position, nullptr, 1), std::invalid_argument);
}

TEST(PBFSolverParticleDataTest, RejectsInvalidOutputRequests) {
    PBFSolver solver;
    const SimulationParams params = validParams();
    initializeSolver(solver, 4, params);

    const std::vector<float4> positions = {
        make_float4(1.0f, 2.0f, 3.0f, 1.0f),
        make_float4(2.0f, 3.0f, 4.0f, 2.0f)
    };
    const std::vector<float4> velocities = {
        make_float4(0.0f, 0.0f, 0.0f, 3.0f),
        make_float4(1.0f, 1.0f, 1.0f, 4.0f)
    };
    solver.setParticles(positions.data(), velocities.data(), positions.size());

    std::vector<float4> output(3);
    EXPECT_THROW(solver.copyPositionsToHost(output.data(), 1), std::out_of_range);
    EXPECT_THROW(solver.copyPositionsToHost(output.data(), 3), std::out_of_range);
    EXPECT_THROW(solver.copyVelocitiesToHost(output.data(), 1), std::out_of_range);
    EXPECT_THROW(solver.copyVelocitiesToHost(output.data(), 3), std::out_of_range);
    EXPECT_THROW(solver.copyPositionsToHost(nullptr, 2), std::invalid_argument);
    EXPECT_THROW(solver.copyVelocitiesToHost(nullptr, 2), std::invalid_argument);
}

TEST(PBFSolverParticleDataTest, AcceptsEmptyParticleSet) {
    PBFSolver solver;
    SimulationParams params = validParams();
    params.solverIterations = 3;
    initializeSolver(solver, 8, params);

    EXPECT_NO_THROW(solver.setParticles(nullptr, nullptr, 0));
    EXPECT_NO_THROW(solver.step());
    EXPECT_NO_THROW(solver.run());
    EXPECT_NO_THROW(solver.copyPositionsToHost(nullptr, 0));
    EXPECT_NO_THROW(solver.copyVelocitiesToHost(nullptr, 0));
}

TEST(PBFSolverParticleDataTest, UploadsAndDownloadsParticlesWithoutChangingThem) {
    PBFSolver solver;
    const SimulationParams params = validParams();
    initializeSolver(solver, 8, params);

    const std::vector<float4> positions = {
        make_float4(1.0f, 2.0f, 3.0f, 4.0f),
        make_float4(-1.0f, -2.0f, -3.0f, -4.0f),
        make_float4(5.5f, 6.5f, 7.5f, 8.5f)
    };
    const std::vector<float4> velocities = {
        make_float4(0.1f, 0.2f, 0.3f, 9.0f),
        make_float4(-0.1f, -0.2f, -0.3f, 10.0f),
        make_float4(1.0f, 2.0f, 3.0f, 11.0f)
    };

    solver.setParticles(positions.data(), velocities.data(), positions.size());

    std::vector<float4> resultPositions(positions.size());
    std::vector<float4> resultVelocities(velocities.size());
    solver.copyPositionsToHost(resultPositions.data(), resultPositions.size());
    solver.copyVelocitiesToHost(resultVelocities.data(), resultVelocities.size());

    for (std::size_t i = 0; i < positions.size(); ++i) {
        expectFloat4Near(resultPositions[i], positions[i], 0.0f, i);
        expectFloat4Near(resultVelocities[i], velocities[i], 0.0f, i);
    }
}

TEST(PBFSolverParticleDataTest, ReplacesActiveParticlesWithASmallerSet) {
    PBFSolver solver;
    const SimulationParams params = validParams();
    initializeSolver(solver, 8, params);

    const std::vector<float4> firstPositions(6, make_float4(1.0f, 1.0f, 1.0f, 1.0f));
    const std::vector<float4> firstVelocities(6, make_float4(2.0f, 2.0f, 2.0f, 2.0f));
    solver.setParticles(firstPositions.data(), firstVelocities.data(), firstPositions.size());

    const std::vector<float4> secondPositions = {
        make_float4(3.0f, 4.0f, 5.0f, 6.0f),
        make_float4(7.0f, 8.0f, 9.0f, 10.0f)
    };
    const std::vector<float4> secondVelocities = {
        make_float4(-1.0f, -2.0f, -3.0f, 11.0f),
        make_float4(-4.0f, -5.0f, -6.0f, 12.0f)
    };
    solver.setParticles(secondPositions.data(), secondVelocities.data(), secondPositions.size());

    std::vector<float4> resultPositions(secondPositions.size());
    std::vector<float4> resultVelocities(secondVelocities.size());
    solver.copyPositionsToHost(resultPositions.data(), resultPositions.size());
    solver.copyVelocitiesToHost(resultVelocities.data(), resultVelocities.size());

    for (std::size_t i = 0; i < secondPositions.size(); ++i) {
        expectFloat4Near(resultPositions[i], secondPositions[i], 0.0f, i);
        expectFloat4Near(resultVelocities[i], secondVelocities[i], 0.0f, i);
    }
}

TEST(PBFSolverStepTest, AdvancesSingleParticleWithGravity) {
    PBFSolver solver;
    SimulationParams params = validParams();
    params.dt = 0.2f;
    params.substeps = 4;
    params.gravity = make_float3(1.0f, -10.0f, 2.0f);
    initializeSolver(solver, 1, params);

    float4 expectedPosition = make_float4(1.0f, 5.0f, 2.0f, 17.0f);
    float4 expectedVelocity = make_float4(2.0f, 3.0f, -1.0f, 23.0f);
    const float4 initialPosition = expectedPosition;
    const float4 initialVelocity = expectedVelocity;
    solver.setParticles(&initialPosition, &initialVelocity, 1);

    integrateReference(
        expectedPosition, expectedVelocity, params.dt, params.gravity, params.substeps
    );
    solver.step();

    float4 resultPosition{};
    float4 resultVelocity{};
    solver.copyPositionsToHost(&resultPosition, 1);
    solver.copyVelocitiesToHost(&resultVelocity, 1);

    expectFloat4Near(resultPosition, expectedPosition, 1e-5f, 0);
    expectFloat4Near(resultVelocity, expectedVelocity, 1e-5f, 0);
}

TEST(PBFSolverStepTest, SeparatesTwoNeighboringParticlesSymmetrically) {
    PBFSolver solver;
    SimulationParams params = validParams();
    params.dt = 0.1f;
    params.gravity = make_float3(0.0f, 0.0f, 0.0f);
    params.solverIterations = 3;
    params.substeps = 1;
    params.restDensity = selfDensity(params.particleMass, params.smoothingRadius);
    initializeSolver(solver, 2, params);

    const std::vector<float4> positions = {
        make_float4(4.75f, 5.0f, 5.0f, 3.0f),
        make_float4(5.25f, 5.0f, 5.0f, 4.0f)
    };
    const std::vector<float4> velocities = {
        make_float4(0.0f, 0.0f, 0.0f, 7.0f),
        make_float4(0.0f, 0.0f, 0.0f, 8.0f)
    };
    solver.setParticles(positions.data(), velocities.data(), positions.size());

    solver.step();

    std::vector<float4> resultPositions(2);
    std::vector<float4> resultVelocities(2);
    solver.copyPositionsToHost(resultPositions.data(), resultPositions.size());
    solver.copyVelocitiesToHost(resultVelocities.data(), resultVelocities.size());

    EXPECT_LT(resultPositions[0].x, positions[0].x);
    EXPECT_GT(resultPositions[1].x, positions[1].x);
    EXPECT_NEAR(resultPositions[0].x + resultPositions[1].x, 10.0f, 1e-5f);
    EXPECT_NEAR(resultPositions[0].y, 5.0f, 1e-6f);
    EXPECT_NEAR(resultPositions[1].y, 5.0f, 1e-6f);
    EXPECT_NEAR(resultPositions[0].z, 5.0f, 1e-6f);
    EXPECT_NEAR(resultPositions[1].z, 5.0f, 1e-6f);
    EXPECT_FLOAT_EQ(resultPositions[0].w, positions[0].w);
    EXPECT_FLOAT_EQ(resultPositions[1].w, positions[1].w);

    EXPECT_LT(resultVelocities[0].x, 0.0f);
    EXPECT_GT(resultVelocities[1].x, 0.0f);
    EXPECT_NEAR(resultVelocities[0].x + resultVelocities[1].x, 0.0f, 1e-4f);
    EXPECT_NEAR(resultVelocities[0].y, 0.0f, 1e-6f);
    EXPECT_NEAR(resultVelocities[1].y, 0.0f, 1e-6f);
    EXPECT_NEAR(resultVelocities[0].z, 0.0f, 1e-6f);
    EXPECT_NEAR(resultVelocities[1].z, 0.0f, 1e-6f);
    EXPECT_FLOAT_EQ(resultVelocities[0].w, velocities[0].w);
    EXPECT_FLOAT_EQ(resultVelocities[1].w, velocities[1].w);
}

TEST(PBFSolverRunTest, ExecutesConfiguredNumberOfSteps) {
    PBFSolver solver;
    SimulationParams params = validParams();
    params.dt = 0.1f;
    params.gravity = make_float3(0.0f, 0.0f, 0.0f);
    params.solverIterations = 3;
    params.substeps = 1;
    initializeSolver(solver, 1, params);

    const float4 position = make_float4(1.0f, 2.0f, 3.0f, 5.0f);
    const float4 velocity = make_float4(2.0f, -1.0f, 0.5f, 6.0f);
    solver.setParticles(&position, &velocity, 1);

    solver.run();

    float4 resultPosition{};
    float4 resultVelocity{};
    solver.copyPositionsToHost(&resultPosition, 1);
    solver.copyVelocitiesToHost(&resultVelocity, 1);

    const float elapsedTime = params.dt * static_cast<float>(params.solverIterations);
    const float4 expectedPosition = make_float4(
        position.x + velocity.x * elapsedTime,
        position.y + velocity.y * elapsedTime,
        position.z + velocity.z * elapsedTime,
        position.w
    );

    expectFloat4Near(resultPosition, expectedPosition, 1e-5f, 0);
    expectFloat4Near(resultVelocity, velocity, 1e-5f, 0);
}

TEST(PBFSolverLargeParticleTest, AdvancesParticlesAcrossSeveralCudaBlocks) {
    constexpr std::size_t particleCount = 1025;
    constexpr int latticeWidth = 11;
    constexpr float spacing = 0.75f;

    PBFSolver solver;
    SimulationParams params = validParams();
    params.dt = 0.02f;
    params.smoothingRadius = 0.5f;
    params.particleMass = 0.25f;
    params.restDensity = selfDensity(params.particleMass, params.smoothingRadius);
    params.lambdaEpsilon = 0.1f;
    params.solverIterations = 2;
    params.substeps = 2;
    params.gravity = make_float3(0.5f, -9.8f, -0.25f);
    initializeSolver(solver, particleCount + 100, params);

    std::vector<float4> positions(particleCount);
    std::vector<float4> velocities(particleCount);
    for (std::size_t i = 0; i < particleCount; ++i) {
        const int x = static_cast<int>(i % latticeWidth);
        const int y = static_cast<int>((i / latticeWidth) % latticeWidth);
        const int z = static_cast<int>(i / (latticeWidth * latticeWidth));
        positions[i] = make_float4(
            0.25f + spacing * x,
            0.25f + spacing * y,
            0.25f + spacing * z,
            1.0f + static_cast<float>(i % 5)
        );
        velocities[i] = make_float4(
            0.1f + 0.00001f * static_cast<float>(x),
            0.2f + 0.00001f * static_cast<float>(y),
            -0.3f + 0.00001f * static_cast<float>(z),
            10.0f + static_cast<float>(i % 7)
        );
    }

    std::vector<float4> expectedPositions = positions;
    std::vector<float4> expectedVelocities = velocities;
    for (std::size_t i = 0; i < particleCount; ++i) {
        integrateReference(
            expectedPositions[i], expectedVelocities[i], params.dt,
            params.gravity, params.substeps
        );
    }

    solver.setParticles(positions.data(), velocities.data(), particleCount);
    solver.step();

    std::vector<float4> resultPositions(particleCount);
    std::vector<float4> resultVelocities(particleCount);
    solver.copyPositionsToHost(resultPositions.data(), particleCount);
    solver.copyVelocitiesToHost(resultVelocities.data(), particleCount);

    for (std::size_t i = 0; i < particleCount; ++i) {
        expectFloat4Near(resultPositions[i], expectedPositions[i], 2e-4f, i);
        expectFloat4Near(resultVelocities[i], expectedVelocities[i], 2e-4f, i);
        EXPECT_TRUE(std::isfinite(resultPositions[i].x)) << "particle " << i;
        EXPECT_TRUE(std::isfinite(resultPositions[i].y)) << "particle " << i;
        EXPECT_TRUE(std::isfinite(resultPositions[i].z)) << "particle " << i;
        EXPECT_TRUE(std::isfinite(resultVelocities[i].x)) << "particle " << i;
        EXPECT_TRUE(std::isfinite(resultVelocities[i].y)) << "particle " << i;
        EXPECT_TRUE(std::isfinite(resultVelocities[i].z)) << "particle " << i;
    }
}
