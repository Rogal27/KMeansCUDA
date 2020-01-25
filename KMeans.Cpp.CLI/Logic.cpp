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