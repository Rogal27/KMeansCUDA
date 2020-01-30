// KMeans.Cpp/Logic.h
#pragma once

namespace KMeans
{
    namespace Cpp
    {
        // This is our native implementation
        // It's marked with __declspec(dllexport) 
        // to be visible from outside the DLL boundaries
        class __declspec(dllexport) Logic
        {
        public:
            int Get() const; // That's where our code goes
            int Sum(int[], int) const;
            int* addParallelVectors(int* vector1, int* vector2, int length);
            int KMeansGather(
                float* vector_x_h,
                float* vector_y_h,
                float* vector_z_h,
                int length,
                int k_param);
            int KMeansImageGather(
                int* colors,
                int length,
                int k_param);
        private:
            float Dist(const float& x1, const float& y1, const float& z1, const float& x2, const float& y2, const float& z2);
        };
    }
}