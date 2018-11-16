## Adding a sphere

Sphere is a set of points in form of `p=(x,y,z)` which for given radius `R` and center point `c=(cx,cy,cz)` follow the equation `(x-cx)*(x-cx)+(y-cy)*(y-cy)+(z-cz)*(z-cz)=R*R`. We can reformulate the equation to use dot product: `dot(p-c, p-c)=R*R`.
Recall that points or our ray are in form of `r(t) = A + t*B`. That means if we are given the ray and the sphere we can calculate if they intersect. Lets substitute `(x,y,z)` with `(r(t)[0], r(t)[1], r(t)[2])` and rewrite first equation.
```
dot(A+t*B-c, A+t*B-c) = R*R
```
Hence
```
t*t*dot(B, B) + 2*t*dot(B,A-c) + dot(A-c, A-c) = R*R
```
This is an simple quadratic equation. Ray hits the sphere it determinant of this quation is greater than 0.
