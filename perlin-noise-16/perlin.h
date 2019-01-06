#ifndef PERLIN_H
#define PERLIN_H

#ifndef RND
#define RND curand_uniform(rand_state)
#endif

__device__ float* perlin_generate(curandState* rand_state);
__device__ void permute(int* p, int n, curandState* rand_state);
__device__ int* perlin_generate_perm(curandState* rand_state);
__device__ float trilin_interpolation(float c[2][2][2], float u, float v, float w);

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
            int i = floor(p.x());
            int j = floor(p.y());
            int k = floor(p.z());
            float c[2][2][2];
            for(int di=0; di<2; di++)
                for(int dj=0; dj<2; dj++)
                    for(int dk=0; dk<2; dk++)
                        c[di][dj][dk] = ranfloat[
                                perm_x[(i+di) & 255] ^
                                perm_y[(j+dj) & 255] ^
                                perm_z[(k+dk) & 255]
                        ];

            return trilin_interpolation(c, u, v, w);
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
#endif
