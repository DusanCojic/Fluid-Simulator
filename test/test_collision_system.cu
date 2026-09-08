#include "pbf/collision_system.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

Container testContainer() {
    return {
        make_float3(-10.0f, -10.0f, -10.0f),
        make_float3(10.0f, 10.0f, 10.0f)
    };
}

float4 solveOne(CollisionSystem& collisionSystem, float4 position, float particleRadius = 0.5f) {
    CudaBuffer<float4> positions(1);
    positions.copyFromHostToDevice(&position, 1);
    collisionSystem.solve(positions.data(), 1, particleRadius);
    positions.copyFromDeviceToHost(&position, 1);
    return position;
}

} // namespace

TEST(CollisionSystemTest, FloorStopsNormalVelocityAndDampsTangentVelocity) {
    const Container container{
        make_float3(0.0f, 0.0f, 0.0f),
        make_float3(10.0f, 10.0f, 10.0f)
    };
    CollisionSystem collisionSystem(container);
    CudaBuffer<float4> positions(1);
    CudaBuffer<float4> velocities(1);
    float4 position = make_float4(2.0f, 0.2f, 2.0f, 1.0f);
    float4 velocity = make_float4(1.0f, -2.0f, 0.0f, 0.0f);

    positions.copyFromHostToDevice(&position, 1);
    velocities.copyFromHostToDevice(&velocity, 1);
    collisionSystem.solve(positions.data(), 1, 0.5f);
    collisionSystem.resolveVelocities(positions.data(), velocities.data(), 1, 0.5f, 0.0f, 0.1f);
    positions.copyFromDeviceToHost(&position, 1);
    velocities.copyFromDeviceToHost(&velocity, 1);

    EXPECT_NEAR(position.y, 0.5f, 1e-5f);
    EXPECT_NEAR(velocity.x, 0.9f, 1e-5f);
    EXPECT_NEAR(velocity.y, 0.0f, 1e-5f);
}

TEST(CollisionSystemTest, SphereIsImpenetrable) {
    const Container container{
        make_float3(-10.0f, -10.0f, -10.0f),
        make_float3(10.0f, 10.0f, 10.0f)
    };
    CollisionSystem collisionSystem(container);
    collisionSystem.setSpheres({ { make_float3(0.0f, 0.0f, 0.0f), 1.0f } });
    CudaBuffer<float4> positions(1);
    CudaBuffer<float4> velocities(1);
    float4 position = make_float4(0.2f, 0.0f, 0.0f, 1.0f);
    float4 velocity = make_float4(-3.0f, 0.0f, 0.0f, 0.0f);

    positions.copyFromHostToDevice(&position, 1);
    velocities.copyFromHostToDevice(&velocity, 1);
    collisionSystem.solve(positions.data(), 1, 0.5f);
    collisionSystem.resolveVelocities(positions.data(), velocities.data(), 1, 0.5f, 0.0f, 0.0f);
    positions.copyFromDeviceToHost(&position, 1);
    velocities.copyFromDeviceToHost(&velocity, 1);

    EXPECT_NEAR(position.x, 1.5f, 1e-5f);
    EXPECT_NEAR(velocity.x, 0.0f, 1e-5f);
}

TEST(CollisionSystemTest, BoxEdgeUsesRoundedContact) {
    const Container container{
        make_float3(-10.0f, -10.0f, -10.0f),
        make_float3(10.0f, 10.0f, 10.0f)
    };
    CollisionSystem collisionSystem(container);
    collisionSystem.setBoxes({ { make_float3(0.0f, 0.0f, 0.0f), make_float3(1.0f, 1.0f, 1.0f) } });
    CudaBuffer<float4> positions(1);
    float4 position = make_float4(1.1f, 1.1f, 0.0f, 1.0f);

    positions.copyFromHostToDevice(&position, 1);
    collisionSystem.solve(positions.data(), 1, 0.5f);
    positions.copyFromDeviceToHost(&position, 1);

    const float distanceX = position.x - 1.0f;
    const float distanceY = position.y - 1.0f;
    EXPECT_NEAR(std::sqrt(distanceX * distanceX + distanceY * distanceY), 0.5f, 1e-5f);
}

TEST(CollisionSystemTest, PlaneStopsNormalVelocity) {
    const Container container{
        make_float3(-10.0f, -10.0f, -10.0f),
        make_float3(10.0f, 10.0f, 10.0f)
    };
    CollisionSystem collisionSystem(container);
    collisionSystem.setPlanes({ { make_float3(0.0f, 0.0f, 0.0f), make_float3(0.0f, 1.0f, 0.0f) } });
    CudaBuffer<float4> positions(1);
    CudaBuffer<float4> velocities(1);
    float4 position = make_float4(0.0f, 0.1f, 0.0f, 1.0f);
    float4 velocity = make_float4(0.0f, -2.0f, 0.0f, 0.0f);

    positions.copyFromHostToDevice(&position, 1);
    velocities.copyFromHostToDevice(&velocity, 1);
    collisionSystem.solve(positions.data(), 1, 0.5f);
    collisionSystem.resolveVelocities(positions.data(), velocities.data(), 1, 0.5f, 0.0f, 0.0f);
    positions.copyFromDeviceToHost(&position, 1);
    velocities.copyFromDeviceToHost(&velocity, 1);

    EXPECT_NEAR(position.y, 0.5f, 1e-5f);
    EXPECT_NEAR(velocity.y, 0.0f, 1e-5f);
}

TEST(CollisionSystemValidationTest, RejectsInvalidSolveArguments) {
    CollisionSystem collisionSystem(testContainer());
    CudaBuffer<float4> positions(1);
    const float nan = std::numeric_limits<float>::quiet_NaN();

    EXPECT_THROW(collisionSystem.solve(nullptr, 1, 0.5f), std::invalid_argument);
    EXPECT_THROW(collisionSystem.solve(positions.data(), 1, 0.0f), std::invalid_argument);
    EXPECT_THROW(collisionSystem.solve(positions.data(), 1, -0.5f), std::invalid_argument);
    EXPECT_THROW(collisionSystem.solve(positions.data(), 1, nan), std::invalid_argument);
    EXPECT_NO_THROW(collisionSystem.solve(positions.data(), 0, 0.5f));

    CollisionSystem narrowContainer({
        make_float3(0.0f, 0.0f, 0.0f),
        make_float3(0.5f, 1.0f, 1.0f)
    });
    EXPECT_THROW(narrowContainer.solve(positions.data(), 1, 0.3f), std::invalid_argument);
}

TEST(CollisionSystemValidationTest, RejectsInvalidContainerBoundsBeforeAllocation) {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    CollisionSystem collisionSystem;

    EXPECT_THROW(
        collisionSystem.setContainer({
            make_float3(0.0f, 0.0f, 0.0f),
            make_float3(0.0f, 1.0f, 1.0f)
        }),
        std::invalid_argument
    );
    EXPECT_THROW(
        collisionSystem.setContainer({
            make_float3(0.0f, 0.0f, 0.0f),
            make_float3(1.0f, nan, 1.0f)
        }),
        std::invalid_argument
    );
}

TEST(CollisionSystemValidationTest, RejectsInvalidVelocityArguments) {
    CollisionSystem collisionSystem(testContainer());
    CudaBuffer<float4> positions(1);
    CudaBuffer<float4> velocities(1);
    const float nan = std::numeric_limits<float>::quiet_NaN();

    EXPECT_THROW(
        collisionSystem.resolveVelocities(nullptr, velocities.data(), 1, 0.5f, 0.0f, 0.0f),
        std::invalid_argument
    );
    EXPECT_THROW(
        collisionSystem.resolveVelocities(positions.data(), nullptr, 1, 0.5f, 0.0f, 0.0f),
        std::invalid_argument
    );
    EXPECT_THROW(
        collisionSystem.resolveVelocities(positions.data(), velocities.data(), 1, 0.0f, 0.0f, 0.0f),
        std::invalid_argument
    );
    EXPECT_THROW(
        collisionSystem.resolveVelocities(positions.data(), velocities.data(), 1, nan, 0.0f, 0.0f),
        std::invalid_argument
    );
    EXPECT_THROW(
        collisionSystem.resolveVelocities(positions.data(), velocities.data(), 1, 0.5f, -0.1f, 0.0f),
        std::invalid_argument
    );
    EXPECT_THROW(
        collisionSystem.resolveVelocities(positions.data(), velocities.data(), 1, 0.5f, 1.1f, 0.0f),
        std::invalid_argument
    );
    EXPECT_THROW(
        collisionSystem.resolveVelocities(positions.data(), velocities.data(), 1, 0.5f, nan, 0.0f),
        std::invalid_argument
    );
    EXPECT_THROW(
        collisionSystem.resolveVelocities(positions.data(), velocities.data(), 1, 0.5f, 0.0f, -0.1f),
        std::invalid_argument
    );
    EXPECT_THROW(
        collisionSystem.resolveVelocities(positions.data(), velocities.data(), 1, 0.5f, 0.0f, 1.1f),
        std::invalid_argument
    );
    EXPECT_THROW(
        collisionSystem.resolveVelocities(positions.data(), velocities.data(), 1, 0.5f, 0.0f, nan),
        std::invalid_argument
    );
    EXPECT_NO_THROW(
        collisionSystem.resolveVelocities(
            positions.data(), velocities.data(), 0, 0.5f, 0.0f, 0.0f
        )
    );

    CollisionSystem narrowContainer({
        make_float3(0.0f, 0.0f, 0.0f),
        make_float3(0.5f, 1.0f, 1.0f)
    });
    EXPECT_THROW(
        narrowContainer.resolveVelocities(
            positions.data(), velocities.data(), 1, 0.3f, 0.0f, 0.0f
        ),
        std::invalid_argument
    );
}

TEST(CollisionSystemValidationTest, RejectsInvalidSphereRadii) {
    CollisionSystem collisionSystem(testContainer());
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float infinity = std::numeric_limits<float>::infinity();

    EXPECT_THROW(
        collisionSystem.setSpheres({ { make_float3(0.0f, 0.0f, 0.0f), 0.0f } }),
        std::invalid_argument
    );
    EXPECT_THROW(
        collisionSystem.setSpheres({ { make_float3(0.0f, 0.0f, 0.0f), -1.0f } }),
        std::invalid_argument
    );
    EXPECT_THROW(
        collisionSystem.setSpheres({ { make_float3(0.0f, 0.0f, 0.0f), nan } }),
        std::invalid_argument
    );
    EXPECT_THROW(
        collisionSystem.setSpheres({ { make_float3(0.0f, 0.0f, 0.0f), infinity } }),
        std::invalid_argument
    );
    EXPECT_THROW(
        collisionSystem.setSpheres({ { make_float3(nan, 0.0f, 0.0f), 1.0f } }),
        std::invalid_argument
    );
}

TEST(CollisionSystemValidationTest, RejectsInvalidBoxHalfExtents) {
    CollisionSystem collisionSystem(testContainer());
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const std::vector<float3> invalidExtents = {
        make_float3(0.0f, 1.0f, 1.0f),
        make_float3(1.0f, -1.0f, 1.0f),
        make_float3(1.0f, 1.0f, nan)
    };

    for (const float3 halfExtents : invalidExtents) {
        EXPECT_THROW(
            collisionSystem.setBoxes({ { make_float3(0.0f, 0.0f, 0.0f), halfExtents } }),
            std::invalid_argument
        );
    }

    EXPECT_THROW(
        collisionSystem.setBoxes({
            { make_float3(0.0f, nan, 0.0f), make_float3(1.0f, 1.0f, 1.0f) }
        }),
        std::invalid_argument
    );
}

TEST(CollisionSystemValidationTest, RejectsInvalidPlaneNormals) {
    CollisionSystem collisionSystem(testContainer());
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float infinity = std::numeric_limits<float>::infinity();

    EXPECT_THROW(
        collisionSystem.setPlanes({
            { make_float3(0.0f, 0.0f, 0.0f), make_float3(0.0f, 0.0f, 0.0f) }
        }),
        std::invalid_argument
    );
    EXPECT_THROW(
        collisionSystem.setPlanes({
            { make_float3(0.0f, 0.0f, 0.0f), make_float3(infinity, 0.0f, 0.0f) }
        }),
        std::invalid_argument
    );
    EXPECT_THROW(
        collisionSystem.setPlanes({
            { make_float3(0.0f, 0.0f, nan), make_float3(0.0f, 1.0f, 0.0f) }
        }),
        std::invalid_argument
    );
}

TEST(CollisionSystemConfigurationTest, DefaultConstructionAndContainerUpdateWork) {
    CollisionSystem collisionSystem;
    collisionSystem.setContainer({
        make_float3(0.0f, 0.0f, 0.0f),
        make_float3(10.0f, 10.0f, 10.0f)
    });
    EXPECT_NEAR(solveOne(collisionSystem, make_float4(-1.0f, 5.0f, 5.0f, 7.0f)).x, 0.5f, 1e-5f);

    collisionSystem.setContainer({
        make_float3(-5.0f, -5.0f, -5.0f),
        make_float3(5.0f, 5.0f, 5.0f)
    });
    EXPECT_NEAR(solveOne(collisionSystem, make_float4(-6.0f, 0.0f, 0.0f, 8.0f)).x, -4.5f, 1e-5f);
}

TEST(CollisionSystemConfigurationTest, RejectsUseBeforeContainerConfiguration) {
    CollisionSystem collisionSystem;
    CudaBuffer<float4> positions(1);
    CudaBuffer<float4> velocities(1);

    EXPECT_THROW(collisionSystem.solve(positions.data(), 1, 0.5f), std::logic_error);
    EXPECT_THROW(
        collisionSystem.resolveVelocities(
            positions.data(), velocities.data(), 1, 0.5f, 0.0f, 0.0f
        ),
        std::logic_error
    );
}

TEST(CollisionSystemTest, ReportsNonConvergentColliderConstraints) {
    CollisionSystem collisionSystem({
        make_float3(-2.0f, -2.0f, -2.0f),
        make_float3(2.0f, 2.0f, 2.0f)
    });
    collisionSystem.setSpheres({
        {make_float3(-1.5f, 0.0f, 0.0f), 0.5f}
    });
    CudaBuffer<float4> positions(1);
    const float4 position = make_float4(-1.6f, 0.0f, 0.0f, 1.0f);
    positions.copyFromHostToDevice(&position, 1);

    EXPECT_THROW(collisionSystem.solve(positions.data(), 1, 0.25f), std::runtime_error);
}

TEST(CollisionSystemConfigurationTest, EmptySettersAndClearMethodsRemoveColliders) {
    CollisionSystem collisionSystem(testContainer());
    collisionSystem.setSpheres({ { make_float3(0.0f, 0.0f, 0.0f), 1.0f } });
    collisionSystem.setBoxes({
        { make_float3(3.0f, 0.0f, 0.0f), make_float3(1.0f, 1.0f, 1.0f) }
    });
    collisionSystem.setPlanes({
        { make_float3(0.0f, 3.0f, 0.0f), make_float3(0.0f, 1.0f, 0.0f) }
    });

    collisionSystem.setSpheres({});
    collisionSystem.clearBoxes();
    collisionSystem.setPlanes({});

    const float4 position = make_float4(0.0f, 0.0f, 0.0f, 4.0f);
    const float4 result = solveOne(collisionSystem, position);
    EXPECT_FLOAT_EQ(result.x, position.x);
    EXPECT_FLOAT_EQ(result.y, position.y);
    EXPECT_FLOAT_EQ(result.z, position.z);
    EXPECT_FLOAT_EQ(result.w, position.w);
    EXPECT_NO_THROW(collisionSystem.clearSpheres());
    EXPECT_NO_THROW(collisionSystem.clearBoxes());
    EXPECT_NO_THROW(collisionSystem.clearPlanes());
}

TEST(CollisionSystemConfigurationTest, SettersReplacePreviousColliders) {
    CollisionSystem collisionSystem(testContainer());
    collisionSystem.setSpheres({ { make_float3(0.0f, 0.0f, 0.0f), 1.0f } });
    collisionSystem.setSpheres({ { make_float3(5.0f, 0.0f, 0.0f), 1.0f } });

    const float4 position = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
    const float4 result = solveOne(collisionSystem, position);
    EXPECT_FLOAT_EQ(result.x, position.x);
    EXPECT_FLOAT_EQ(result.y, position.y);
    EXPECT_FLOAT_EQ(result.z, position.z);
}

TEST(CollisionSystemConfigurationTest, FailedSetterPreservesPreviousColliders) {
    CollisionSystem collisionSystem(testContainer());
    collisionSystem.setSpheres({ { make_float3(0.0f, 0.0f, 0.0f), 1.0f } });
    EXPECT_THROW(
        collisionSystem.setSpheres({ { make_float3(5.0f, 0.0f, 0.0f), -1.0f } }),
        std::invalid_argument
    );

    const float4 result = solveOne(
        collisionSystem, make_float4(0.0f, 0.0f, 0.0f, 1.0f)
    );
    EXPECT_NEAR(result.x, 1.5f, 1e-5f);
}

TEST(CollisionSystemIntegrationTest, ResolvesMultipleColliderTypesAndVelocities) {
    CollisionSystem collisionSystem(testContainer());
    collisionSystem.setSpheres({ { make_float3(-5.0f, 5.0f, 0.0f), 1.0f } });
    collisionSystem.setBoxes({
        { make_float3(0.0f, 5.0f, 0.0f), make_float3(1.0f, 1.0f, 1.0f) }
    });
    collisionSystem.setPlanes({
        { make_float3(0.0f, 3.0f, 0.0f), make_float3(0.0f, 1.0f, 0.0f) }
    });

    std::vector<float4> positions = {
        make_float4(-5.2f, 5.0f, 0.0f, 1.0f),
        make_float4(1.2f, 5.0f, 0.0f, 2.0f),
        make_float4(5.0f, 2.8f, 0.0f, 3.0f)
    };
    std::vector<float4> velocities = {
        make_float4(2.0f, 0.0f, 0.0f, 4.0f),
        make_float4(-2.0f, 0.0f, 0.0f, 5.0f),
        make_float4(0.0f, -2.0f, 0.0f, 6.0f)
    };
    CudaBuffer<float4> devicePositions(positions.size());
    CudaBuffer<float4> deviceVelocities(velocities.size());
    devicePositions.copyFromHostToDevice(positions.data(), positions.size());
    deviceVelocities.copyFromHostToDevice(velocities.data(), velocities.size());

    collisionSystem.solve(devicePositions.data(), positions.size(), 0.5f);
    collisionSystem.resolveVelocities(
        devicePositions.data(), deviceVelocities.data(), positions.size(),
        0.5f, 0.0f, 0.0f
    );
    devicePositions.copyFromDeviceToHost(positions.data(), positions.size());
    deviceVelocities.copyFromDeviceToHost(velocities.data(), velocities.size());

    EXPECT_NEAR(positions[0].x, -6.5f, 1e-5f);
    EXPECT_NEAR(positions[1].x, 1.5f, 1e-5f);
    EXPECT_NEAR(positions[2].y, 3.5f, 1e-5f);
    EXPECT_NEAR(velocities[0].x, 0.0f, 1e-5f);
    EXPECT_NEAR(velocities[1].x, 0.0f, 1e-5f);
    EXPECT_NEAR(velocities[2].y, 0.0f, 1e-5f);
}
