#include "vec3.h"
#include "perlin.h"
#ifndef TEXTURE_H
#define TEXTURE_H

struct tex_info
{
    char* path;
    unsigned char* tex_data;
    int nx, ny, nn;
};

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
        __device__ noise_texture(perlin* n, float s) : noise(n), scale(s) {}
        __device__ virtual vec3 value(float u, float v, const vec3& p) const
        {
            //return vec3(1,1,1)*noise->noise(scale*p);
            //return vec3(1,1,1)*noise->turb(scale*p);
            return vec3(1,1,1)*0.5*(1+sin(scale*p.z() + 10*noise->turb(p)));
        }

        perlin* noise;
        float scale;
};

class image_texture : public e_texture
{
    public:
        __device__ image_texture() {}
        __device__ image_texture(unsigned char* pixels, int A, int B) : data(pixels), nx(A), ny(B) {}
        __device__ virtual vec3 value(float u, float v, const vec3& p) const;
        unsigned char* data;
        int nx, ny;
};

__device__ vec3 image_texture::value(float u, float v, const vec3& p) const
{
    int i = u*nx;
    int j = (1-v)*ny-0.001;
    if(i<0) i=0;
    if(j<0) j=0;
    if(i>nx-1) i=nx-1;
    if(j>ny-1) j=ny-1;
    float r = int(data[3*i + 3*nx*j])/ 255.0;
    float g = int(data[3*i + 3*nx*j+1])/ 255.0;
    float b = int(data[3*i + 3*nx*j+2])/ 255.0;

    return vec3(r,g,b);
};
#endif
