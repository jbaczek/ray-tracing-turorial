#ifndef RAND_VEC_H
#define RAND_VEC_H
#include "vec3.h"
#include <curand_kernel.h>

__device__ vec3 random_in_unit_sphere(curandState* rand_state)
{
    vec3 p;
    do
    {
        p = 2.0*vec3(curand_uniform(rand_state), curand_uniform(rand_state), curand_uniform(rand_state)) - vec3(1,1,1);
    } while (p.squared_length() >= 1.0);

    return p;
}

__device__ vec3 random_in_unit_circle(curandState* rand_state)
{
    vec3 p;
    do
    {
        p = 2.0*vec3(curand_uniform(rand_state), curand_uniform(rand_state),0) - vec3(1,1,0);
    } while (dot(p,p) >= 1);

    return p;
}

#endif
