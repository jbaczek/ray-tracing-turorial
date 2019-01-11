#ifndef PERLIN_H
#define PERLIN_H

#include "rand_vec.h"
#include "vec3.h"
#ifndef RND
#define RND curand_uniform(rand_state)
#endif

__device__ vec3* perlin_generate(curandState* rand_state);
__device__ void permute(int* p, int n, curandState* rand_state);
__device__ int* perlin_generate_perm(curandState* rand_state);
__device__ float trilin_interpolation(float c[2][2][2], float u, float v, float w);
__device__ inline float perlin_interpolation(vec3 c[2][2][2], float u, float v, float w);

class perlin
{
    public:
        __device__ perlin(curandState* rand_state)
        {
            ranvec = perlin_generate(rand_state);
            perm_x = perlin_generate_perm(rand_state);
            perm_y = perlin_generate_perm(rand_state);
            perm_z = perlin_generate_perm(rand_state);
        }
        __device__ float noise(const vec3& p) const
        {
            float u = p.x() - floor(p.x());
            float v = p.y() - floor(p.y());
            float w = p.z() - floor(p.z());
            int i = floor(p.x());
            int j = floor(p.y());
            int k = floor(p.z());
            vec3 c[2][2][2];
            for(int di=0; di<2; di++)
                for(int dj=0; dj<2; dj++)
                    for(int dk=0; dk<2; dk++)
                        c[di][dj][dk] = ranvec[
                                perm_x[(i+di) & 255] ^
                                perm_y[(j+dj) & 255] ^
                                perm_z[(k+dk) & 255]
                        ];

            return perlin_interpolation(c, u, v, w);
        }

        __device__ float turb(const vec3& p, int depth=7) const
        {
            float accum = 0;
            float freq = 1;
            float amp = 1;
            float max_val = 0;
            for(int i=0; i<depth; i++)
            {
                accum += amp*noise(freq * p);
                max_val += amp;
                amp *= 0.5;
                freq *= 2;
            }

            return accum/max_val;
        }

        vec3* ranvec;
        int* perm_x;
        int* perm_y;
        int* perm_z;
};

__device__ vec3* perlin_generate(curandState* rand_state)
{
    vec3* p = new vec3[256];
    for(int i=0; i< 256; i++)
    {
        p[i] = random_in_unit_sphere(rand_state);
        p[i].make_unit_vector();
    }
    return p;
}

//Thist implements Knuth shuffles algorithm
__device__ void permute(int* p, int n, curandState* rand_state)
{
    for(int i=n-1; i>0; i--)
    {
        int target = int(RND*(i+1));
        int tmp = p[i];
        p[i] = p[target];
        p[target] = tmp;    
    }
}

__device__ int* perlin_generate_perm(curandState* rand_state)
{
    int* p = new int[256];
    for(int i=0; i<256; i++)
        p[i] = i;
    permute(p, 256, rand_state);
    return p;
}

__device__ float trilin_interpolation(float c[2][2][2], float u, float v, float w)
{
    float accum = 0;
    for(int i=0; i<2; i++)
        for(int j=0; j<2; j++)
            for(int k=0; k<2; k++)
                accum += (i*u + (1-i)*(1-u))*
                         (j*v + (1-j)*(1-v))*
                         (k*w + (1-k)*(1-w))*
                         c[i][j][k];

    return accum;
}

__device__ inline float perlin_interpolation(vec3 c[2][2][2], float u, float v, float w)
{
    float uu = u*u*(3-2*u);
    float vv = v*v*(3-2*v);
    float ww = w*w*(3-2*w);
    float accum = 0;
    for(int i=0; i<2; i++)
        for(int j=0; j<2; j++)
            for(int k=0; k<2; k++)
            {
                vec3 weight_v(u-i, v-j, w-k);
                accum += (i*uu + (1-i)*(1-uu))*
                         (j*vv + (1-j)*(1-vv))*
                         (k*ww + (1-k)*(1-ww))*
                         (dot(weight_v, c[i][j][k])+1)/2;
            }
    return accum;
}
#endif
