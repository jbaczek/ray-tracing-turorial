##Antialiasing

We can see that images we produced so far havevery sharp edges. This is the effect of quantization of an image and assigning every pixel to specific object. On photos pixels making edges of an object are blend of the color of the object andthe color of the background. We will simulate this in our code by sending multiple rays through one pixel and then averaging their colors. We will randomly jiggle every ray with uniform distribution so every pixel will have a mixed color if rays will pass "near" the object edge.
