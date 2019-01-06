#ifndef PERLIN_H
#define PERLIN_H

#ifndef RND
#define RND curand_uniform(rand_state)
#endif

__device__ float* perlin_generate(curandState* rand_state);
__device__ void permute(int* p, int n, curandState* rand_state);
__device__ int* perlin_generate_perm(curandState* rand_state);

class perlin
{
    public:
        __device__ perlin(curandState* rand_state)
        {
            ranfloat = perlin_generate(rand_state);
            perm_x = perlin_generate_perm(rand_state);
            perm_y = perlin_generate_perm(rand_state);
            perm_z = perlin_generate_perm(rand_state);
        }
        __device__ float noise(const vec3& p) const
        {
            float u = p.x() - floor(p.x());
            float v = p.y() - floor(p.y());
            float w = p.z() - floor(p.z());
            int i = int(4*p.x()) & 255;
            int j = int(4*p.y()) & 255;
            int k = int(4*p.z()) & 255;

            return ranfloat[perm_x[i] ^ perm_y[j] ^ perm_z[k]];
        }

        float* ranfloat;
        int* perm_x;
        int* perm_y;
        int* perm_z;
};

__device__ float* perlin_generate(curandState* rand_state)
{
    float* p = new float[256];
    for(int i=0; i< 256; i++)
        p[i] = RND;
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
#endif
