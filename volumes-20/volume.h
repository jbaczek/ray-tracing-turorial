#ifndef VOLUME_H
#define VOLUME_H
#include "hitable.h"
#include "material.h"
#include <curand_kernel.h>

class constant_medium : public hitable
{
    public:
        __device__ constant_medium(hitable* b, float d, e_texture* a) : boundary(b), density(d)
        {
            phase_function = new isotropic(a);
        }
        __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const;
        __device__ virtual bool bounding_box(float t0, float t1, aabb& box) const
        {
            return boundary->bounding_box(t0, t1, box);
        }
        hitable* boundary;
        float density;
        material* phase_function;
};


__device__ bool constant_medium::hit(const ray& r, float t_min, float t_max, hit_record& rec) const
{
    hit_record rec1, rec2;
    rec1.rand_state = rec.rand_state;
    rec2.rand_state = rec.rand_state;
    if(boundary->hit(r, -FLT_MAX, FLT_MAX, rec1))
        if(boundary->hit(r, rec1.t+0.0001, FLT_MAX, rec2))
        {
            if(rec1.t < t_min)
                rec1.t=t_min;
            if(rec2.t > t_max)
                rec2.t = t_max;
            if(rec1.t >= rec2.t)
                return false;
            if(rec1.t < 0)
                rec1.t = 0;
            float distance_inside_boundary = (rec2.t-rec1.t)*r.direction().length();
            float hit_distance = -(1/density)*log(curand_uniform(rec.rand_state));
            if(hit_distance < distance_inside_boundary)
            {
                rec.t = rec1.t + hit_distance / r.direction().length();
                rec.p = r.point_at_parameter(rec.t);
                rec.normal = vec3(1,0,0);
                rec.mat_ptr = phase_function;
                return true;
            }
        }
   return false;
}

#endif
