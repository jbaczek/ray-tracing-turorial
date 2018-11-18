## Normal vectors
In order to shade a bit the image we can use normal vectors. This is a vector perpendicular to the surface at given point. Having point `P` on a sphere centered at `C` normal vector at poin `P` has value `P-C`. Now we can assign color to a pixel based on direction of the normal vector at point of intersection of a ray and the sphere.
Simultaneously we introduce abstraction for a collection of objects that can be hit by a ray.
