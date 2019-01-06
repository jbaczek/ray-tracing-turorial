#include "vec3.h"
#include "perlin.h"
#ifndef TEXTURE_H
#define TEXTURE_H

class e_texture
{
    public:
        __device__ virtual vec3 value(float u, float v, const vec3& p) const = 0;
};

class constant_texture : public e_texture
{
    public:
        __device__ constant_texture() {}
        __device__ constant_texture(vec3 c) : color(c) {}
        __device__ virtual vec3 value(float u, float v, const vec3& p) const
        {
            return color;
        }

        vec3 color;
};

class checker_texture : public e_texture
{
    public:
        __device__ checker_texture() {}
        __device__ checker_texture(e_texture* o, e_texture* e) : odd(o), even(e) {}
        __device__ virtual vec3 value(float u, float v, const vec3& p) const
        {
            float sines = sin(10*p.x())*sin(10*p.y())*sin(10*p.z());
            if(sines < 0)
                return odd->value(u, v, p);
            else
                return even->value(u, v, p);
        }

        e_texture* odd;
        e_texture* even;
};

class noise_texture : public e_texture
{
    public:
        __device__ noise_texture() {}
        __device__ noise_texture(perlin* n) : noise(n) {}
        __device__ virtual vec3 value(float u, float v, const vec3& p) const
        {
            return vec3(1,1,1)*noise->noise(p);
        }

        perlin* noise;
};

#endif
