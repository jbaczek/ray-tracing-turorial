#ifndef MATERIAL_H
#define MATERIAL_H
#include "rand_vec.h"
#include "ray.h"
#include "hitable.h"
#include "texture.h"
#include <curand_kernel.h>

struct hit_record;

__device__ float schlick(float cosine, float ref_idx) 
{
    float r0 = (1-ref_idx) / (1+ref_idx);
    r0 = r0*r0;
    return r0 + (1-r0)*pow((1 - cosine),5);
}

__device__ bool refract(const vec3& v, const vec3& n, float ni_over_nt, vec3& refracted)
{
    vec3 uv = unit_vector(v);
    vec3 un = unit_vector(n);
    float cosI = -dot(uv, un);
    float sinT2 = ni_over_nt*ni_over_nt*( 1.0 - cosI * cosI);
    if (sinT2 < 1.0)
    {
        float cosT = sqrt(1.0 - sinT2);
        refracted = ni_over_nt*uv + (ni_over_nt*cosI - cosT)*un;
        return true;
    }
    else
        return false;
}

__device__ inline vec3 reflect(const vec3& v, const vec3& n)
{
    return v - 2*dot(v,n)*n;
}

class material
{
    public:
        __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, curandState* rand_state) const = 0;
};

class lambertian : public material
{
    public:
        __device__ lambertian(e_texture* a) : albedo(a) {}
        __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, curandState* rand_state) const
        {
            vec3 target = rec.normal + random_in_unit_sphere(rand_state);
            scattered = ray(rec.p, target, r_in.time());
            attenuation = albedo->value(rec.u, rec.v, rec.p);
            return true;
        }

        e_texture* albedo;
};

class metal : public material
{
    public:
        __device__ metal(e_texture* a, float f) : albedo(a) { if (f<1) fuzz = f; else fuzz = 1; }
        __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, curandState* rand_state) const
        {
            vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
            scattered = ray(rec.p, reflected + fuzz*random_in_unit_sphere(rand_state), r_in.time());
            attenuation = albedo->value(0, 0, rec.p);
            return dot(scattered.direction(), rec.normal) > 0;
        }

        e_texture* albedo;
        float fuzz;
};

class dielectric : public material
{
    public:
        __device__ dielectric(float ri) : ref_idx(ri) {}
        __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, curandState* rand_state) const
        {
            vec3 outward_normal;
            vec3 reflected = reflect(r_in.direction(), rec.normal);
            float ni_over_nt;
            attenuation = vec3(1.0, 1.0, 1.0);
            vec3 refracted;
            float reflection_prob;
            float ang; // it represents sin/cos of an angle
            if(dot(r_in.direction(), rec.normal) > 0)
            {
                outward_normal = -rec.normal;
                ni_over_nt = ref_idx;
                ang = dot(r_in.direction(), rec.normal) / (r_in.direction().length()*rec.normal.length()); //cosine
                ang = sqrt(1.0 - ref_idx*ref_idx*(1.0-ang*ang)); // sine
            }
            else
            {
                outward_normal = rec.normal;
                ni_over_nt = 1.0 / ref_idx;
                ang = -dot(r_in.direction(), rec.normal) / (r_in.direction().length()*rec.normal.length()); //cosine
            }

            if (refract(r_in.direction(), outward_normal, ni_over_nt, refracted))
                reflection_prob = schlick(ang, ref_idx);
            else
                reflection_prob = 1.0;
            if (curand_uniform(rand_state) < reflection_prob)
                scattered = ray(rec.p, reflected, r_in.time());
            else
                scattered = ray(rec.p, refracted, r_in.time());
            return true;
        }
        
        float ref_idx;
};

#endif
