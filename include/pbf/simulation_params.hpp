#ifndef SIMULATION_PARAMS_H
#define SIMULATION_PARAMS_H

#include <vector_types.h>

struct SimulationParams {
    float dt;                 // time step

    float restDensity;        // target fluid density
    float particleMass;       // mass of one particle
    float smoothingRadius;    // particle interaction radius

    float lambdaEpsilon;      // constraint solver stabilization

    int solverIterations;     // PBF iterations per substep
    int substeps;             // simulation substeps per frame

    float3 gravity;           // gravitational acceleration vector

    float scorrK;             // artificial pressure strength
    float scorrN;             // artificial pressure exponent
    float scorrDeltaQ;        // artificial pressure reference distance

    float xsphViscosity;      // velocity smoothing strength
    float vorticityStrength;  // vorticity confinement strength
};


#endif