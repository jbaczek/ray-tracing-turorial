#ifndef RAND_VEC_H
#define RAND_VEC_H
#include "vec3.h"

float rand_float()
{
    return float(rand())/(float(RAND_MAX)+1.0);
}

vec3 random_in_unit_sphere()
{
    vec3 p;
    do
    {
        p = 2.0*vec3(rand_float(), rand_float(), rand_float()) - vec3(1,1,1);
    } while (p.squared_length() >= 1.0);

    return p;
}

vec3 random_in_unit_circle()
{
    vec3 p;
    do
    {
        p = 2.0*vec3(rand_float(), rand_float(),0) - vec3(1,1,0);
    } while (dot(p,p) >= 1);

    return p;
}

#endif
