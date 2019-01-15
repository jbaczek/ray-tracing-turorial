#ifndef RECTANGLE_H
#define RECTANGLE_H
#include "hitable.h"
#include "vec3.h"
#include "material.h"
#include "aabb.h"
#include "bvh.h"

class xy_rect : public hitable
{
    public:
        __device__ xy_rect() {}
        __device__ xy_rect(float _x0, float _x1, float _y0, float _y1, float _k, material* m):
            x0(_x0), x1(_x1), y0(_y0), y1(_y1), k(_k), mat(m) {};
        __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const;
        __device__ virtual bool bounding_box(float t0, float t1, aabb& box) const
        {
            box = aabb(vec3(x0,y0, k-0.0001), vec3(x1,y1, k+0.0001));
            return true;
        }

        material* mat;
        float x0, x1, y0, y1, k;
};

__device__ bool xy_rect::hit(const ray& r, float t0, float t1, hit_record& rec) const
{
    float t = (k-r.origin().z()) / r.direction().z();
    if(t < t0 || t > t1)
        return false;
    float x = r.origin().x() + t*r.direction().x();
    float y = r.origin().y() + t*r.direction().y();
    if(x < x0 || x > x1 || y < y0 || y > y1)
        return false;
    rec.u = (x-x0)/(x1-x0);
    rec.v = (y-y0)/(y1-y0);
    rec.t = t;
    rec.mat_ptr = mat;
    rec.p = r.point_at_parameter(t);
    rec.normal = vec3(0,0,1);
    return true;
}

#endif
