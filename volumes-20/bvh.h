#include "hitable.h"
#include <curand_kernel.h>
#ifndef BVH_H
#define BVH_H
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
struct box_x_cmp
{
    __device__ bool operator()(const hitable* a, const hitable* b)
    {
        aabb box_left, box_right;
        a->bounding_box(0,0,box_left);
        b->bounding_box(0,0,box_right);
        return box_left.min().x() > box_right.min().x();
    }
};
struct box_y_cmp
{
    __device__ bool operator()(const hitable* a, const hitable* b)
    {
        aabb box_left, box_right;
        a->bounding_box(0,0,box_left);
        b->bounding_box(0,0,box_right);
        return box_left.min().y() > box_right.min().y();
    }
};
struct box_z_cmp
{
    __device__ bool operator()(const hitable* a, const hitable* b)
    {
        aabb box_left, box_right;
        a->bounding_box(0,0,box_left);
        b->bounding_box(0,0,box_right);
        return box_left.min().z() > box_right.min().z();
    }
};

class bvh_node : public hitable
{
    public:
        __device__ bvh_node() {}
        __device__ bvh_node(hitable** l, int n, float time0, float time1, curandState* rand_state);
        __device__ virtual bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const;
        __device__ virtual bool bounding_box(float t0, float t1, aabb& box) const;
        hitable* left;
        hitable* right;
        aabb box;
};

__device__ bvh_node::bvh_node(hitable** l, int n, float time0, float time1, curandState* rand_state)
{
    int axis = int(3*curand_uniform(rand_state));
    if (axis == 0)
        thrust::sort(thrust::seq, l, l+n, box_x_cmp());
    else if(axis == 1)
        thrust::sort(thrust::seq, l, l+n, box_y_cmp());
    else
        thrust::sort(thrust::seq, l, l+n, box_z_cmp());
    if(n == 1)
    {
        left = right = l[0];
    }
    else if(n ==2 )
    {
        left = l[0];
        right = l[1];
    }
    else
    {
        left = new bvh_node(l, n/2, time0, time1, rand_state);
        right = new bvh_node(l + n/2, n - n/2, time0, time1, rand_state);
    }
    aabb box_left, box_right;
    left->bounding_box(time0, time1, box_left);
    right->bounding_box(time0, time1, box_right);
    box = surrounding_box(box_left, box_right);

}

__device__ bool bvh_node::bounding_box(float t0, float t1, aabb& b) const
{
    b = box;
    return true;
}

__device__ bool bvh_node::hit(const ray& r, float t_min, float t_max, hit_record& rec) const
{
    if(box.hit(r, t_min, t_max))
    {
        hit_record left_rec, right_rec;
        left_rec.rand_state = rec.rand_state;
        right_rec.rand_state = rec.rand_state;
        bool hit_left = left->hit(r, t_min, t_max, left_rec);
        bool hit_right = right->hit(r, t_min, t_max, right_rec);
        if(hit_left && hit_right)
        {
            if(left_rec.t < right_rec.t)
                rec = left_rec;
            else
                rec = right_rec;
            return true;
        }
        else if(hit_left)
        {
            rec = left_rec;
            return true;
        }
        else if(hit_right)
        {
            rec = right_rec;
            return true;
        }
        else
            return false;
    }
    else return false;
}
#endif
