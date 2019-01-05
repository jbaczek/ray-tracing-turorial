#ifndef SCENES_H
#define SCENES_H
#define RND curand_uniform(rand_state)

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
#endif
