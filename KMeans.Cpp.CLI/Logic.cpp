// KMeans.Cpp.CLI/Logic.cpp
#include "Logic.h"
#include "..\KMeans.Cpp\Logic.h"

using namespace std;

KMeans::Cpp::CLI::Logic::Logic() : _impl(new Cpp::Logic())
    // Allocate some memory for the native implementation
{
}

int KMeans::Cpp::CLI::Logic::Get()
{
    return _impl->Get(); // Call native Get
}

int KMeans::Cpp::CLI::Logic::Sum(array<int>^ tab, int length)
{
    //int* unmanaged_tab = new int[length];
    //System::Runtime::InteropServices::Marshal::Copy(tab, 0, unmanaged_tab, length);
    pin_ptr<int> tab_ptr = &tab[0];
    return _impl->Sum(tab_ptr, length);
}

array<int>^ KMeans::Cpp::CLI::Logic::addParallelVectors(array<int>^ vector1, array<int>^ vector2, int length)
{
    pin_ptr<int> vector1_ptr = &vector1[0];
    pin_ptr<int> vector2_ptr = &vector2[0];
    int* result_ptr = _impl->addParallelVectors(vector1_ptr, vector2_ptr, length);
    if (result_ptr == nullptr)
    {
        return nullptr;
    }
    array<int>^ result = gcnew array<int>(length);
    System::Runtime::InteropServices::Marshal::Copy((IntPtr)result_ptr, result, 0, length);
    delete result_ptr;
    return result;
}

int KMeans::Cpp::CLI::Logic::KMeansGather(
    array<float>^ vector_x_h,
    array<float>^ vector_y_h,
    array<float>^ vector_z_h,
    int length,
    int k_param)
{
    pin_ptr<float> vector_x_ptr = &vector_x_h[0];
    pin_ptr<float> vector_y_ptr = &vector_y_h[0];
    pin_ptr<float> vector_z_ptr = &vector_z_h[0];
    int iters = _impl->KMeansGather(vector_x_ptr, vector_y_ptr, vector_z_ptr, length, k_param);
    return iters;
}

int KMeans::Cpp::CLI::Logic::KMeansImageGather(
    array<int>^ colors,
    int length,
    float XR,
    float YR,
    float ZR,
    float gamma,
    array<float>^ RGBtoXYZMatrix,
    array<float>^ XYZtoRGBMatrix,
    int k_param)
{
    pin_ptr<int> colors_ptr = &colors[0];
    pin_ptr<float> RGBtoXYZMatrix_ptr = &RGBtoXYZMatrix[0];
    pin_ptr<float> XYZtoRGBMatrix_ptr = &XYZtoRGBMatrix[0];
    int iters = _impl->KMeansImageGather(colors_ptr, length, XR, YR, ZR, gamma, RGBtoXYZMatrix_ptr, XYZtoRGBMatrix_ptr, k_param);
    return iters;
}

void KMeans::Cpp::CLI::Logic::Destroy()
{
    if (_impl != nullptr)
    {
        delete _impl;
        _impl = nullptr;
    }
}

KMeans::Cpp::CLI::Logic::~Logic()
{
    // C++ CLI compiler will automaticly make all ref classes implement IDisposable.
    // The default implementation will invoke this method + call GC.SuspendFinalize.
    Destroy(); // Clean-up any native resources 
}

KMeans::Cpp::CLI::Logic::!Logic()
{
    // This is the finalizer
    // It's essentially a fail-safe, and will get called
    // in case Logic was not used inside a using block.
    Destroy(); // Clean-up any native resources 
}