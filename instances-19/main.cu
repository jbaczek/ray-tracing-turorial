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
#include "bvh.h"
#include "scenes.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
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

__global__ void create_camera(camera** cam, int nx, int ny)
{
    if(threadIdx.x==0 && blockIdx.x==0)
    {
        vec3 lookfrom(278, 278, -800);
        vec3 lookat(278, 278, 0);
        vec3 vup(0, 1, 0);
        float vfov = 40.0;
        float aspect = float(nx)/float(ny);
        float aperture = 0.0;
        float focus_dist = (lookat-lookfrom).length();
        *cam = new camera(lookfrom, lookat, vup, vfov, aspect, aperture, focus_dist, 0.0, 1.0);
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
    vec3 cur_emitted = vec3(0,0,0);
    for(int i=0; i<50; i++)
    {
        hit_record rec;
        if((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec))
        {
            ray scattered;
            vec3 attenuation;
            vec3 emitted = rec.mat_ptr->emitted(rec.u, rec.v, rec.p);
            cur_emitted += attenuation*emitted;
            if(rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, rand_state))
            {
                cur_attenuation *= attenuation;
                cur_ray = scattered;
            }
            else
            {
                return cur_attenuation*emitted;
            }
        }
        else
        {
            return vec3(0,0,0);
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
    int nx = 1000;
    int ny = 500;
    int ns = 3000;
    int tx = 8;
    int ty = 8;

    dim3 blocks(nx/tx+1,ny/ty+1);
    dim3 threads(tx,ty);

    std::cerr << "Rendering a " << nx << "x" << ny << " image ";
    std::cerr << "in " << tx << "x" << ty << " blocks.\n";

    int num_pixels = nx*ny;
    size_t fb_size = num_pixels*sizeof(vec3);

    //Set bigger heap size
    size_t heapSize = 0;
    checkCudaErrors(cudaDeviceGetLimit(&heapSize, cudaLimitMallocHeapSize));
    std::cerr << "Initial heap size: " << heapSize/(1024*1024) << "MB\n";
    size_t newHeapSize = 1024 * 1024 * 200;
    checkCudaErrors(cudaDeviceSetLimit(cudaLimitMallocHeapSize, newHeapSize));
    checkCudaErrors(cudaDeviceGetLimit(&heapSize, cudaLimitMallocHeapSize));
    std::cerr << "Modified heap size: " << heapSize/(1024*1024) << "MB\n";
    size_t stackSize = 0;
    checkCudaErrors(cudaDeviceGetLimit(&stackSize, cudaLimitStackSize));
    std::cerr << "Initial stack size: " << stackSize/1024 << "kB\n";
    size_t newStackSize = 1024 * 10;
    checkCudaErrors(cudaDeviceSetLimit(cudaLimitStackSize, newStackSize));
    checkCudaErrors(cudaDeviceGetLimit(&stackSize, cudaLimitStackSize));
    std::cerr << "Modified stack size: " << stackSize/1024 << "kB\n";


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

    //load textures (to be refactored)
    int num_tex = 1;
    tex_info* textures = new tex_info[num_tex];
    tex_info info1;
    info1.path=(char*)std::string("earth.jpg").c_str();
    textures[0] = info1;
    tex_info* d_textures;
    checkCudaErrors(cudaMalloc((void**) &d_textures, num_tex*sizeof(tex_info)));
    for(int i=0; i<num_tex; i++)
    {
       textures[i].tex_data = stbi_load(textures[i].path,
               &textures[i].nx, &textures[i].ny, &textures[i].nn, 0);
       char* d_path;
       int path_len = strlen(textures[i].path);
       checkCudaErrors(cudaMalloc((void**) &d_path, path_len));
       checkCudaErrors(cudaMemcpy(d_path, textures[i].path, path_len, cudaMemcpyHostToDevice));
       unsigned char* d_tex_data;
       int data_len = textures[i].nx * textures[i].ny * textures[i].nn;
       checkCudaErrors(cudaMalloc((void**) &d_tex_data, data_len));
       checkCudaErrors(cudaMemcpy(d_tex_data, textures[i].tex_data, data_len, cudaMemcpyHostToDevice));
       free(textures[i].tex_data);
       textures[i].path = d_path;
       textures[i].tex_data = d_tex_data;
    }
       checkCudaErrors(cudaMemcpy(d_textures, textures, num_tex*sizeof(tex_info), cudaMemcpyHostToDevice));


    // create world
    hitable **d_list;
    int num_hitables = 6;
    checkCudaErrors(cudaMalloc((void**) &d_list, num_hitables*sizeof(hitable*))); 
    hitable **d_world;
    checkCudaErrors(cudaMalloc((void**) &d_world, sizeof(hitable*)));
    create_scene6<<<1,1>>>(d_list,d_world, d_rand_state2);
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
            vec3 col = vec3(sqrt(fb[pixel_index].r()), sqrt(fb[pixel_index].g()), sqrt(fb[pixel_index].b()));
            int ir = int(255.99*col.r());
            int ig = int(255.99*col.g());
            int ib = int(255.99*col.b());
            std::cout << ir << " " << ig << " " << ib << "\n";
        }
    }

    checkCudaErrors(cudaDeviceSynchronize());
    free_scene6<<<1,1>>>(d_list, d_world);
    checkCudaErrors(cudaGetLastError());
    free_camera<<<1,1>>>(d_cam);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_cam));
    checkCudaErrors(cudaFree(fb));
}
