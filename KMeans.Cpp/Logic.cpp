// KMeans.Cpp/Logic.cpp
#include "Logic.h"

#include "kernel.cuh"
#include <cuda_runtime.h>
#include <stdio.h>

int KMeans::Cpp::Logic::Get() const
{
	return 42; // Really, what else did you expect?
}

int KMeans::Cpp::Logic::Sum(int tab[], int length) const
{
	int sum = 0;
	for (int i = 0; i < length; i++)
	{
		sum += tab[i];
	}
	return sum;
}

int* KMeans::Cpp::Logic::addParallelVectors(int* vector1, int* vector2, int length)
{
	//return invokeParallelSumCUDA(vector1, vector2, length);
	main();
	return nullptr;
}