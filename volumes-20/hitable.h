#ifndef HITABLE_H
#define HITABLE_H

#include "ray.h"
#include "aabb.h"
#include <curand_kernel.h>

class material;

struct hit_record
{
    float t;
    float u;
    float v;
    vec3 p;
    vec3 normal;
    material *mat_ptr;
    curandState* rand_state;
};

class hitable
{
    public:
        __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const = 0;
        __device__ virtual bool bounding_box(float t0, float t1, aabb& box) const =0;
};

#endif
