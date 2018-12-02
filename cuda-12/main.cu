#include <iostream>
#include <time.h>
#include <float.h>
#include <curand_kernel.h>
#include "vec3.h"
#include "ray.h"
#include "hitable.h"
#include "hitablelist.h"
#include "sphere.h"
#include "camera.h"
#include "material.h"

#define RND curand_uniform(rand_state)
// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}
__global__ void create_world(hitable** d_list, hitable** d_world, curandState* rand_state)
{
    if(threadIdx.x==0 && blockIdx.x==0)
    {
        d_list[0] = new sphere(vec3(0,-1000,-1), 1000, new lambertian(vec3(0.5, 0.5, 0.5)));
        int i = 1;
        for(int a=-11; a<11; a++)
        {
            for(int b=-11; b<11; b++)
            {
                float choose_mat = RND;
                vec3 center(a+RND, 0.2, b+RND);
                if(choose_mat < 0.8f)
                {
                    d_list[i++] = new sphere(center, 0.2,
                            new lambertian(vec3(RND*RND, RND*RND, RND*RND)));
                }
                else if(choose_mat < 0.95f)
                {
                    d_list[i++] = new sphere(center, 0.2,
                            new metal(vec3(0.5f*(1.0f+RND), 0.5f*(1.0f+RND), 0.5f*(1.0f+RND)), 0.5f*RND));
                }
                else
                {
                    d_list[i++] = new sphere(center, 0.2, new dielectric(1.5));
                }
            }
        }
        d_list[i++] = new sphere(vec3(0,1,0), 1.0, new dielectric(1.5));
        d_list[i++] = new sphere(vec3(-4,1,0), 1.0, new lambertian(vec3(0.4, 0.2, 0.1)));
        d_list[i++] = new sphere(vec3(4,1,0), 1.0, new metal(vec3(0.7, 0.6, 0.5), 0.0));
        *d_world = new hitable_list(d_list, 22*22+4);
    }
}

__global__ void free_world(hitable **d_list, hitable **d_world)
{
    for(int i=0; i<22*22+4; i++)
    {
        delete((sphere*)d_list[i])->mat_ptr;
        delete d_list[i];
    }
    delete *d_world;
}

__global__ void create_camera(camera** cam, int nx, int ny)
{
    if(threadIdx.x==0 && blockIdx.x==0)
    {
        vec3 lookfrom(13, 2, 3);
        vec3 lookat(0.0, 0.0, 0);
        vec3 vup(0, 1, 0);
        float vfov = 30.0;
        float aspect = float(nx)/float(ny);
        float aperture = 0.0;
        float focus_dist = (lookat-lookfrom).length();
        *cam = new camera(lookfrom, lookat, vup, vfov, aspect, aperture, focus_dist);
    }
}

__global__ void free_camera(camera** cam)
{
    if(threadIdx.x==0 && blockIdx.x==0)
        delete *(cam);
}
__device__ vec3 color(const ray& r, hitable** world, curandState* rand_state)
{
    ray cur_ray = r;
    vec3 cur_attenuation = vec3(1.0, 1.0, 1.0);
    for(int i=0; i<50; i++)
    {
        hit_record rec;
        if((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec))
        {
            ray scattered;
            vec3 attenuation;
            if(rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, rand_state))
            {
                cur_attenuation *= attenuation;
                cur_ray = scattered;
            }
            else
            {
                return vec3(0,0,0);
            }
        }
        else
        {
            vec3 unit_direction = unit_vector(cur_ray.direction());
            float t = 0.5f*(unit_direction.y() + 1.0f);
            vec3 c = (1.0f-t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
            return cur_attenuation*c;
        }
    }
    return vec3(0,0,0);
}

__global__ void rand_init(curandState *rand_state) 
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        curand_init(1984, 0, 0, rand_state);
    }
}

__global__ void render_init(int max_x, int max_y, curandState* rand_state)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j*max_x + i;
    curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
}

__global__ void render(vec3 *fb, int max_x, int max_y, int ns, camera ** cam, hitable** world, curandState *rand_state) 
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j*max_x + i;
    curandState local_rand_state = rand_state[pixel_index];
    vec3 col(0,0,0);
    for(int s=0; s<ns; s++)
    {
        float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
        float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
        ray r = (*cam)->get_ray(u,v, &local_rand_state);
        col += color(r, world, &local_rand_state);
    }
    fb[pixel_index] = col/float(ns);
}


int main() {
    int nx = 2000;
    int ny = 1000;
    int ns = 100;
    int tx = 8;
    int ty = 8;

    dim3 blocks(nx/tx+1,ny/ty+1);
    dim3 threads(tx,ty);

    std::cerr << "Rendering a " << nx << "x" << ny << " image ";
    std::cerr << "in " << tx << "x" << ty << " blocks.\n";

    int num_pixels = nx*ny;
    size_t fb_size = num_pixels*sizeof(vec3);

    // allocate FB
    vec3 *fb;
    checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

    // curand init
    curandState* d_rand_state;
    checkCudaErrors(cudaMalloc((void**)&d_rand_state, num_pixels*sizeof(curandState)));
    curandState* d_rand_state2;
    checkCudaErrors(cudaMalloc((void**)&d_rand_state2, sizeof(curandState)));
    rand_init<<<1,1>>>(d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // create world
    hitable **d_list;
    int num_hitables =22*22+4;
    checkCudaErrors(cudaMalloc((void**) &d_list, num_hitables*sizeof(hitable*))); 
    hitable **d_world;
    checkCudaErrors(cudaMalloc((void**) &d_world, sizeof(hitable*)));
    create_world<<<1,1>>>(d_list,d_world, d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    //create camera
    camera **d_cam;
    checkCudaErrors(cudaMalloc((void**) &d_cam, sizeof(camera*)));
    create_camera<<<1,1>>>(d_cam, nx, ny);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    clock_t start, stop;
    start = clock();
    // Render our buffer
    render_init<<<blocks, threads>>>(nx, ny, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    render<<<blocks, threads>>>(fb, nx, ny, ns, d_cam, d_world, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "took " << timer_seconds << " seconds.\n";

    // Output FB as Image
    std::cout << "P3\n" << nx << " " << ny << "\n255\n";
    for (int j = ny-1; j >= 0; j--) {
        for (int i = 0; i < nx; i++) {
            size_t pixel_index = j*nx + i;
            int ir = int(255.99*fb[pixel_index].r());
            int ig = int(255.99*fb[pixel_index].g());
            int ib = int(255.99*fb[pixel_index].b());
            std::cout << ir << " " << ig << " " << ib << "\n";
        }
    }

    checkCudaErrors(cudaDeviceSynchronize());
    free_world<<<1,1>>>(d_list, d_world);
    checkCudaErrors(cudaGetLastError());
    free_camera<<<1,1>>>(d_cam);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_cam));
    checkCudaErrors(cudaFree(fb));
}
