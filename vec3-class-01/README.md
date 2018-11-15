## vec3 class
In this lesson we write simple C++ vector class for holding spatial position or RGB values. In graphics libraries for this purposes 4D vectors are used (for sake of geometry or additional alpha channel for RGB). To write as little code as possible we will use vec3 class for all the purposes we can but this approach will not prevent us from doing silly things such as adding a color to a locaion in space. Thus we should be very careful when doing operations. Steps to compile:
```
g++ -c vec3.cpp
g++ -c main.cpp
g++ -o main main.o vec3.o
```
