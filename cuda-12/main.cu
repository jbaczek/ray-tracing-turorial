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
        d_list[0] = new sphere(vec3(0,-1000,0), 1000, new lambertian(vec3(0.5, 0.5, 0.5)));
        int i = 1;
        for(int a=-11; a<11; a++)
        {
            for(int b=-11; b<11; b++)
            {
                float choose_mat = curand_uniform(rand_state);
                vec3 center(a+0.9f*curand_uniform(rand_state), 0.2f, b+0.9f*curand_uniform(rand_state));
                if ((center - vec3(4, 0.2, 0)).length() > 0.9)
                {
                    if(choose_mat < 0.8f)
                    {
                        d_list[i++] = new sphere(center, 0.2, new lambertian(vec3(curand_uniform(rand_state)*curand_uniform(rand_state), curand_uniform(rand_state)*curand_uniform(rand_state), curand_uniform(rand_state)*curand_uniform(rand_state))));
                    }
                    else if(choose_mat < 0.95f)
                    {
                        d_list[i++] = new sphere(center, 0.2,
                                new metal(vec3(0.5*(1+curand_uniform(rand_state)), 0.5*(1+curand_uniform(rand_state)), 0.5*(1+curand_uniform(rand_state))), 0.5*curand_uniform(rand_state)));
                    }
                    else
                    {
                        d_list[i++] = new sphere(center, 0.2, new dielectric(1.5));
                    }
                }
            }
        }
        d_list[i++] = new sphere(vec3(0,1,0), 1.0, new dielectric(1.5));
        d_list[i++] = new sphere(vec3(-4,1,0), 1.0, new lambertian(vec3(0.4, 0.2, 0.1)));
        d_list[i++] = new sphere(vec3(4,1,0), 1.0, new metal(vec3(0.7, 0.6, 0.5), 0.0));
    }
}

__global__ void free_world(hitable **d_list, hitable **d_world)
{
    delete *(d_list);
    delete *(d_list+1);
    delete *d_world;
}

__global__ void create_camera(camera** cam)
{
    if(threadIdx.x==0 && blockIdx.x==0)
    {
        vec3 lookfrom(0.0, 1.0, 0.0);
        vec3 lookat(0.0, 0.0, -1.0);
        vec3 vup(0.0, 1.0, 0.0);
        float vfov = 90.0;
        float aspect = 2.0;
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
__device__ vec3 color(const ray& r, hitable** world, curandState* local_rand_state)
{
    ray cur_ray = r;
    ray scattered;
    vec3 attenuation(1.0, 1.0, 1.0);
    vec3 cur_attenuation;
    for(int i=0; i<50; i++)
    {
        hit_record rec;
        if((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec) &&
                rec.mat_ptr->scatter(cur_ray, rec, cur_attenuation, scattered, local_rand_state))
        {
            attenuation *= cur_attenuation;
            cur_ray = scattered;
        }
        else
        {
            vec3 unit_direction = unit_vector(r.direction());
            float t = 0.5f*(unit_direction.y() + 1.0f);
            vec3 c = (1.0f-t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
            return attenuation*c;
        }
    }
    return vec3(0,0,0);
}

__global__ void render_init(int max_x, int max_y, curandState *rand_state)
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
    int nx = 1200;
    int ny = 600;
    int ns = 100;
    int tx = 8;
    int ty = 8;

    dim3 blocks(nx/tx+1,ny/ty+1);
    dim3 threads(tx,ty);

    vec3 lower_left_corner(-2.0, -1.0, -1.0);
    vec3 horizontal(4.0, 0.0, 0.0);
    vec3 vertical(0.0, 2.0, 0.0);
    vec3 origin(0.0, 0.0, 0.0);

    std::cerr << "Rendering a " << nx << "x" << ny << " image ";
    std::cerr << "in " << tx << "x" << ty << " blocks.\n";

    int num_pixels = nx*ny;
    size_t fb_size = num_pixels*sizeof(vec3);

    // curand init
    curandState * d_rand_state;
    checkCudaErrors(cudaMalloc((void**)&d_rand_state, num_pixels*sizeof(curandState)));

    // allocate FB
    vec3 *fb;
    checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

    // create world
    hitable **d_list;
    checkCudaErrors(cudaMalloc((void**) &d_list, 200*sizeof(hitable*))); //allocate sufficient amount of memory to hold whole world
    hitable **d_world;
    checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(hitable *)));
    create_world<<<1,1>>>(d_list,d_world, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    //create camera
    camera **d_cam;
    checkCudaErrors(cudaMalloc((void**) &d_cam, sizeof(camera*)));
    create_camera<<<1,1>>>(d_cam);
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
    //checkCudaErrors(cudaDeviceSynchronize());
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
