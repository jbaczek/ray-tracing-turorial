#include "vec3.h"
#ifndef AABB_H
#define AABB_H

__device__ inline float ffmin(float a, float b) { return a < b ? a : b; }
__device__ inline float ffmax(float a, float b) { return a > b ? a : b; }

class aabb
{
    public:
        __device__ aabb() {}
        __device__ aabb(const vec3& a, const vec3& b) { _min = a; _max = b; }

        vec3 min() const { return _min; }
        vec3 max() const { return _max; }

        __device__ bool hit(const ray& r, float tmin, float tmax) const;
        vec3 _min;
        vec3 _max;
};

__device__ inline bool aabb::hit(const ray& r, float tmin, float tmax) const
{
    for(int i=0; i<3; i++)
    {
        float invD = 1.0f / r.direction()[i];
        float t0 = (min()[i] - r.origin()[i]) * invD;
        float t1 = (max()[i] - r.origin()[i]) * invD;
        if(invD < 0.0f)
            std::swap(t0, t1);
        tmin = t0 > tmin ? t0 : tmin;
        tmax = t1 < tmax ? t1 : tmax;
        if (tmax <= tmin)
            return false;
    }
    return true;
}

__device__ aabb surrounding_box(aabb box0, aabb box1)
{
    vec3 small(fmin(box0.min().x(), box1.min().x()),
               fmin(box0.min().y(), box1.min().y()),
               fmin(box0.min().z(), box1.min().z()));
    vec3 big  (fmin(box0.max().x(), box1.max().x()),
               fmin(box0.max().y(), box1.max().y()),
               fmin(box0.max().z(), box1.max().z()));

    return aabb(small, big);
}
#endif
