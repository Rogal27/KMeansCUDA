// KMeans.Cpp.CLI/Logic.h
#pragma once
using namespace System;

namespace KMeans
{
    namespace Cpp
    {
        // First a Forward Declaration to Cpp::Logic class:
        class Logic; // This allows us to mention it in this header file
        // without actually including the native version of Logic.h

        namespace CLI
        {
            // Next is the managed wrapper of Logic:
            public ref class Logic
            {
            public:
                // Managed wrappers are generally less concerned 
                // with copy constructors and operators, since .NET will
                // not call them most of the time.
                // The methods that do actually matter are:
                // The constructor, the "destructor" and the finalizer
                Logic();
                ~Logic();
                !Logic();

                int Get();

                int Sum(array<int>^, int);
                array<int>^ addParallelVectors(array<int>^ vector1, array<int>^ vector2, int length);
                int KMeansGather(
                    array<float>^ vector_x_h,
                    array<float>^ vector_y_h,
                    array<float>^ vector_z_h,
                    int length,
                    int k_param);

                void Destroy(); // Helper function
            private:
                // Pointer to our implementation
                Cpp::Logic* _impl;
            };
        }
    }
}