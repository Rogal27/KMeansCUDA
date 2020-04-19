# KMeansCUDA
CUDA program for k-means algorithm. Includes WPF GUI.
Program contains 3 gpu k-means implementations and 2 cpu implementations.
GPU methods:
 - Scatter
 - Gather
 - Reduce by key

CPU methods are:
 - Scatter
 - Multi-thread Scatter

Selection of centroids is randomized.

GUI is written in C#, connection between C# and C++ was achieved by C++/CLI wrapper.

GUI also allows to change color space of an image, and change chromaticities of the primary colors and the white point for source and destination color space.

KMeans algorithm is applied to loaded image. Images are converted to L* a* b* (so distances between colors are defined by Euclidean metrics). GPU algorithms can work also with points in 3D space.
