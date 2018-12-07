#include "hitable.h"

#ifndef BVH_H
#define BVH_H
class bvh_node : public hitable
{
    public:
        __device__ bvh_node() {}
        __device__ bvh_node(hitable** l, int n, float time0, float time1);
        __device__ virtual bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const;
        __device__ virtual bool bounding_box(float t0, float t1, abb& box) const;
        hitable* left;
        hitable* right;
        aabb box;
}

__device__ bool bvh_node::bounding_box(float t0, float t1, aabb& b) const
{
    b = box;
    return true;
}

#endif
