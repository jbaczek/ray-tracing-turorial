#ifndef SPHERE_H
#define SPHERE_H

#include "hitable.h"
#include "aabb.h"
#include <cmath>

__device__ void polar_to_kart(const vec3& p, float& u, float& v);

class sphere: public hitable
{
    public:
    __device__ sphere() {}
    __device__ sphere(vec3 cen, float r, material* m) : center0(cen), center1(cen), radius(r), mat_ptr(m){ time0 = 0.0; time1 = 0.1; }
	__device__ sphere(vec3 cen0, vec3 cen1, float t0, float t1, float r, material* m) :
		center0(cen0), center1(cen1), time0(t0), time1(t1), radius(r), mat_ptr(m) {};
	__device__ virtual bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const;
    __device__ virtual bool bounding_box(float t0, float t1, aabb& box) const;
	__device__ vec3 center(float time) const;
	vec3 center0, center1;
	float radius;
	float time0, time1;
	material* mat_ptr;
};

__device__ vec3 sphere::center(float time) const
{
	return center0 + ((time-time0) / (time1-time0))*(center1 - center0);
}

__device__ bool sphere::hit(const ray& r, float t_min, float t_max, hit_record& rec) const
{
    vec3 oc = r.origin() - center(r.time());    
    float a = dot(r.direction(), r.direction());
    float b = dot(oc, r.direction());
    float c = dot(oc, oc) - radius*radius;
    float discriminant = b*b -a*c;
    polar_to_kart((rec.p-center(r.time()))/radius, rec.u, rec.v);

    if (discriminant > 0)
    {
        float temp = (-b - sqrt(discriminant))/a;
        if (temp < t_max && temp > t_min)
        {
            rec.t = temp;
            rec.p = r.point_at_parameter(rec.t);
            rec.normal = (rec.p-center(r.time()))/radius;
            rec.mat_ptr = mat_ptr;
            return true;
        }
        temp = (-b+sqrt(discriminant))/a;
        if (temp < t_max && temp > t_min)
        {
            rec.t = temp;
            rec.p = r.point_at_parameter(rec.t);
            rec.normal = (rec.p-center(r.time()))/radius;
            rec.mat_ptr = mat_ptr;
            return true;
        } 
    }
    return false;
}

__device__ bool sphere::bounding_box(float t0, float t1, aabb& box) const
{
    aabb box0 = aabb(center(time0) - vec3(radius, radius, radius), center(time0) + vec3(radius, radius, radius));
    aabb box1 = aabb(center(time1) - vec3(radius, radius, radius), center(time1) + vec3(radius, radius, radius));
    box = surrounding_box(box0, box1);
    return true;
}

// it assumes that p is a unit vector, u and v are image coordinates (origin is top left)
__device__ void polar_to_kart(const vec3& p, float& u, float& v)
{
    float phi = atan2(p.z(), p.x());
    float theta = asin(p.y());
    u = 1-(phi + M_PI)/(2*M_PI);
    v = (theta + M_PI/2)/M_PI;
}

#endif
