#ifndef SCENES_H
#define SCENES_H
#define RND curand_uniform(rand_state)
#include "rectangle.h"
#include "box.h"

__global__ void create_scene1(hitable** d_list, hitable** d_world, curandState* rand_state)
{
    if(threadIdx.x==0 && blockIdx.x==0)
    {
        e_texture* checker = new checker_texture( new constant_texture(vec3(0.2, 0.3, 0.1)),
                new constant_texture(vec3(0.9, 0.9, 0.9)));
        d_list[0] = new sphere(vec3(0,-1000,-1), 1000, new lambertian(checker));
        int i = 1;
        for(int a=-11; a<11; a++)
        {
            for(int b=-11; b<11; b++)
            {
                float choose_mat = RND;
                vec3 center(a+RND, 0.2, b+RND);
                if(choose_mat < 0.8f)
                {
                    e_texture* tex = new constant_texture(vec3(RND*RND, RND*RND, RND*RND));
                    d_list[i++] = new sphere(center, center+vec3(0,0.5*RND,0), 0.0, 1.0, 0.2,
                            new lambertian(tex));
                }
                else if(choose_mat < 0.95f)
                {
                    e_texture* tex = new constant_texture(vec3(0.5f*(1.0f+RND), 0.5f*(1.0f+RND), 0.5f*(1.0f+RND)));
                    d_list[i++] = new sphere(center, 0.2,
                            new metal(tex, 0.5f*RND));
                }
                else
                {
                    d_list[i++] = new sphere(center, center, 0.0, 1.0, 0.2, new dielectric(1.5));
                }
            }
        }
        d_list[i++] = new sphere(vec3(0,1,0), 1.0, new dielectric(1.5));
        d_list[i++] = new sphere(vec3(-4,1,0), 1.0, new lambertian(new constant_texture(vec3(0.4, 0.2, 0.1))));
        d_list[i++] = new sphere(vec3(4,1,0), 1.0, new metal(new constant_texture(vec3(0.7, 0.6, 0.5)), 0.0));

        //*d_world = new hitable_list(d_list, 22*22+4);
        *d_world = new bvh_node(d_list, 22*22+4, 0.0, 0.1, rand_state);
    }
}

__global__ void free_scene1(hitable **d_list, hitable **d_world)
{
    for(int i=0; i<22*22+4; i++)
    {
        delete((sphere*)d_list[i])->mat_ptr;
        delete d_list[i];
    }
    delete *d_world;
}

__global__ void create_scene2(hitable** d_list, hitable** d_world, curandState* rand_state)
{
    if(threadIdx.x==0 && blockIdx.x==0)
    {
        e_texture* checker = new checker_texture( new constant_texture(vec3(0.2, 0.3, 0.1)),
                new constant_texture(vec3(0.9, 0.9, 0.9)));
        d_list[0] = new sphere(vec3(0,-10,0), 10, new lambertian(checker));
        d_list[1] = new sphere(vec3(0, 10,0), 10, new lambertian(checker));

        *d_world = new hitable_list(d_list, 2);
    }
}

__global__ void free_scene2(hitable **d_list, hitable **d_world)
{
    for(int i=0; i<2; i++)
    {
        delete((sphere*)d_list[i])->mat_ptr;
        delete d_list[i];
    }
    delete *d_world;
}

__global__ void create_scene3(hitable** d_list, hitable** d_world, curandState* rand_state)
{
    if(threadIdx.x==0 && blockIdx.x==0)
    {
        perlin* noise = new perlin(rand_state);
        e_texture* pertext = new noise_texture(noise, 4);
        d_list[0] = new sphere(vec3(0,-1000,0), 1000, new lambertian(pertext));
        d_list[1] = new sphere(vec3(0, 2, 0), 2, new lambertian(pertext));

        *d_world = new hitable_list(d_list, 2);
    }
}

__global__ void free_scene3(hitable **d_list, hitable **d_world)
{
    for(int i=0; i<2; i++)
    {
        delete((sphere*)d_list[i])->mat_ptr;
        delete d_list[i];
    }
    delete *d_world;
}

__global__ void create_scene4(hitable** d_list, hitable** d_world, curandState* rand_state, tex_info* textures)
{
    if(threadIdx.x==0 && blockIdx.x==0)
    {
        e_texture* imtex = new image_texture(textures[0].tex_data, textures[0].nx, textures[0].ny);
        d_list[0] = new sphere(vec3(0,0,0), 3, new lambertian(imtex));

        *d_world = new hitable_list(d_list, 1);
    }
}

__global__ void free_scene4(hitable **d_list, hitable **d_world)
{
    delete((sphere*)d_list[0])->mat_ptr;
    delete d_list[0];
    delete *d_world;
}
__global__ void create_scene5(hitable** d_list, hitable** d_world, curandState* rand_state)
{
    if(threadIdx.x==0 && blockIdx.x==0)
    {
        perlin* noise = new perlin(rand_state);
        e_texture* pertext = new noise_texture(noise, 4);
        d_list[0] = new sphere(vec3(0,-1000,0), 1000, new lambertian(pertext));
        d_list[1] = new sphere(vec3(0, 2, 0), 2, new lambertian(pertext));
        d_list[2] = new sphere(vec3(0, 7, 0), 2, new diffuse_light(new constant_texture(vec3(4,4,4))));
        d_list[3] = new xy_rect(3, 5, 1, 3, -2, new diffuse_light(new constant_texture(vec3(4,4,4))));

        *d_world = new hitable_list(d_list, 4);
    }
}

__global__ void free_scene5(hitable** d_list, hitable** d_world)
{
    for(int i=0; i<3; i++)
    {
        delete((sphere*)d_list[i])->mat_ptr;
        delete d_list[i];
    }
    delete((xy_rect*)d_list[3])->mat_ptr;
    delete d_list[3];
    delete *d_world;
}

__global__ void create_scene6(hitable** d_list, hitable** d_world, curandState* rand_state)
{
    if(threadIdx.x==0 && blockIdx.x==0)
    {
        int i=0;
        material* red   = new lambertian(new constant_texture(vec3(0.65, 0.05, 0.05)));
        material* white = new lambertian(new constant_texture(vec3(0.73, 0.73, 0.73)));
        material* green = new lambertian(new constant_texture(vec3(0.12, 0.45, 0.15)));
        material* light = new diffuse_light(new constant_texture(vec3(30, 30, 30)));
        d_list[i++] = new flip_normals(new yz_rect(0, 555, 0, 555, 555, green));
        d_list[i++] = new yz_rect(0, 555, 0, 555, 0,   red);
        d_list[i++] = new xz_rect(213, 343, 227, 332, 554, light);
        d_list[i++] = new flip_normals(new xz_rect(0, 555, 0, 555, 555, white));
        d_list[i++] = new xz_rect(0, 555, 0, 555, 0, white);
        d_list[i++] = new flip_normals(new xy_rect(0, 555, 0, 555, 555, white));
        d_list[i++] = new translate(new rotate_y(new box(vec3(0,0,0), vec3(165,165,165), white), -18), vec3(130,0,65));
        d_list[i++] = new translate(new rotate_y(new box(vec3(0,0,0), vec3(165,330,165), white), 15), vec3(265,0,265));
       
        *d_world = new hitable_list(d_list, i);
    }
}

__global__ void free_scene6(hitable** d_list, hitable** d_world)
{
    delete ((yz_rect*)((flip_normals*)d_list[0])->ptr)->mat_ptr;
    delete ((flip_normals*)d_list[0])->ptr;
    delete ((yz_rect*)d_list[1])->mat_ptr;
    delete ((xz_rect*)d_list[2])->mat_ptr;
    delete ((xz_rect*)((flip_normals*)d_list[3])->ptr)->mat_ptr;
    delete ((flip_normals*)d_list[3])->ptr;
    delete ((flip_normals*)d_list[5])->ptr;
    for(int i=0; i<8; i++)
    {
        delete d_list[i];
    }
    delete *d_world;
    
}
#endif
