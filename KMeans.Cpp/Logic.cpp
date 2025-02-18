// KMeans.Cpp/Logic.cpp
#include "Logic.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "kernel.cuh"
#include "kmeans.cuh"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

constexpr float EPS = 1e-4f;
constexpr int RGBConvertionMatrixSize = 9;

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
    return invokeParallelSumCUDA(vector1, vector2, length);
	//return nullptr;
}

float KMeans::Cpp::Logic::Dist(const float& x1, const float& y1, const float& z1, const float& x2, const float& y2, const float& z2)
{
	float dx = x1 - x2;
	float dy = y1 - y2;
	float dz = z1 - z2;
	return dx * dx + dy * dy + dz * dz;
}

int KMeans::Cpp::Logic::KMeansGather(
	float* vector_x_h,
	float* vector_y_h,
	float* vector_z_h,
	int length,
	int k_param,
	int max_iter)
{
	const unsigned int memSizeFloat = sizeof(float) * length;
	const unsigned int memSizeInt = sizeof(int) * length;
	const unsigned int centroidSize = sizeof(float) * k_param;
	float* vector_x_d;
	float* vector_y_d;
	float* vector_z_d;
	float* centroid_x_h = new float[k_param];
	float* centroid_y_h = new float[k_param];
	float* centroid_z_h = new float[k_param];
	float* centroid_x_d;
	float* centroid_y_d;
	float* centroid_z_d;
	int* cluster;
	allocateArray((void**)&vector_x_d, memSizeFloat);
	allocateArray((void**)&vector_y_d, memSizeFloat);
	allocateArray((void**)&vector_z_d, memSizeFloat);
	allocateArray((void**)&centroid_x_d, centroidSize);
	allocateArray((void**)&centroid_y_d, centroidSize);
	allocateArray((void**)&centroid_z_d, centroidSize);
	allocateArray((void**)&cluster, memSizeInt);

	srand(time(NULL));

	for (size_t i = 0; i < k_param; i++)
	{
		int ind = rand() % length;
		bool recalculate = true;
		while (recalculate == true)
		{
			recalculate = false;
			centroid_x_h[i] = vector_x_h[ind];
			centroid_y_h[i] = vector_y_h[ind];
			centroid_z_h[i] = vector_z_h[ind];
			bool recalculate = false;
			for (size_t j = 0; j < i; j++)
			{
				float dist = Dist(centroid_x_h[i], centroid_y_h[i], centroid_z_h[i], centroid_x_h[j], centroid_y_h[j], centroid_z_h[j]);
				if (dist < EPS)
				{
					recalculate = true;
				}
			}
		}
	}

	copyArrayToDevice(vector_x_d, vector_x_h, 0, memSizeFloat);
	copyArrayToDevice(vector_y_d, vector_y_h, 0, memSizeFloat);
	copyArrayToDevice(vector_z_d, vector_z_h, 0, memSizeFloat);
	copyArrayToDevice(centroid_x_d, centroid_x_h, 0, centroidSize);
	copyArrayToDevice(centroid_y_d, centroid_y_h, 0, centroidSize);
	copyArrayToDevice(centroid_z_d, centroid_z_h, 0, centroidSize);

	int iterations = KMeansGatherCuda(
		vector_x_d,
		vector_y_d,
		vector_z_d,
		length,
		k_param,
		max_iter,
		cluster,
		centroid_x_d,
		centroid_y_d,
		centroid_z_d);

	copyArrayFromDevice(vector_x_h, vector_x_d, memSizeFloat);
	copyArrayFromDevice(vector_y_h, vector_y_d, memSizeFloat);
	copyArrayFromDevice(vector_z_h, vector_z_d, memSizeFloat);

	freeArray(vector_x_d);
	freeArray(vector_y_d);
	freeArray(vector_z_d);
	freeArray(centroid_x_d);
	freeArray(centroid_y_d);
	freeArray(centroid_z_d);
	freeArray(cluster);
	delete [] centroid_x_h;
	delete [] centroid_z_h;
	delete [] centroid_y_h;

	return iterations;
}

int KMeans::Cpp::Logic::KMeansImage(
	int* colors,
	int length,
	float XR,
	float YR,
	float ZR,
	float gamma,
	float* RGBtoXYZMatrix,
	float* XYZtoRGBMatrix,
	int k_param,
	int max_iter,
	int param)
{
	const unsigned int memSizeFloat = sizeof(float) * length;
	const unsigned int memSizeInt = sizeof(int) * length;
	const unsigned int conversionArraySize = sizeof(int) * RGBConvertionMatrixSize;
	const unsigned int centroidSize = sizeof(float) * k_param;
	int* colors_d;
	float* RGBtoXYZMatrix_d;
	float* XYZtoRGBMatrix_d;
	float* vector_x_d;
	float* vector_y_d;
	float* vector_z_d;
	float* centroid_x_h = new float[k_param];
	float* centroid_y_h = new float[k_param];
	float* centroid_z_h = new float[k_param];
	float* centroid_x_d;
	float* centroid_y_d;
	float* centroid_z_d;
	int* cluster;
	allocateArray((void**)&colors_d, memSizeInt);
	allocateArray((void**)&RGBtoXYZMatrix_d, conversionArraySize);
	allocateArray((void**)&XYZtoRGBMatrix_d, conversionArraySize);
	allocateArray((void**)&vector_x_d, memSizeFloat);
	allocateArray((void**)&vector_y_d, memSizeFloat);
	allocateArray((void**)&vector_z_d, memSizeFloat);
	allocateArray((void**)&centroid_x_d, centroidSize);
	allocateArray((void**)&centroid_y_d, centroidSize);
	allocateArray((void**)&centroid_z_d, centroidSize);
	allocateArray((void**)&cluster, memSizeInt);

	srand(time(NULL));

	for (size_t i = 0; i < k_param; i++)
	{
		int ind = rand() % length;
		bool recalculate = true;
		while (recalculate == true)
		{
			recalculate = false;
			centroid_x_h[i] = (float)(rand() % 101) / 100.0f;
			centroid_y_h[i] = (float)(rand() % 201 - 100) / 100.0f;
			centroid_z_h[i] = (float)(rand() % 201 - 100) / 100.0f;
			bool recalculate = false;
			for (size_t j = 0; j < i; j++)
			{
				float dist = Dist(centroid_x_h[i], centroid_y_h[i], centroid_z_h[i], centroid_x_h[j], centroid_y_h[j], centroid_z_h[j]);
				if (dist < EPS)
				{
					recalculate = true;
				}
			}
		}
	}

	copyArrayToDevice(colors_d, colors, 0, memSizeInt);
	copyArrayToDevice(RGBtoXYZMatrix_d, RGBtoXYZMatrix, 0, conversionArraySize);
	copyArrayToDevice(XYZtoRGBMatrix_d, XYZtoRGBMatrix, 0, conversionArraySize);
	copyArrayToDevice(centroid_x_d, centroid_x_h, 0, centroidSize);
	copyArrayToDevice(centroid_y_d, centroid_y_h, 0, centroidSize);
	copyArrayToDevice(centroid_z_d, centroid_z_h, 0, centroidSize);


	ConvertToLABCuda(
		colors_d,
		length,
		XR,
		YR,
		ZR,
		gamma,
		RGBtoXYZMatrix_d,
		vector_x_d,
		vector_y_d,
		vector_z_d);

	cudaDeviceSynchronize();

	int iterations = -1;

	if (param == 0)
	{
		iterations = KMeansGatherCuda(
			vector_x_d,
			vector_y_d,
			vector_z_d,
			length,
			k_param,
			max_iter,
			cluster,
			centroid_x_d,
			centroid_y_d,
			centroid_z_d);
	}
	else if (param == 1)
	{
		iterations = KMeansScatterCuda(
			vector_x_d,
			vector_y_d,
			vector_z_d,
			length,
			k_param,
			max_iter,
			cluster,
			centroid_x_d,
			centroid_y_d,
			centroid_z_d);
	}
	else if (param == 2)
	{
		iterations = KMeansReduceByKeyCuda(
			vector_x_d,
			vector_y_d,
			vector_z_d,
			length,
			k_param,
			max_iter,
			cluster,
			centroid_x_d,
			centroid_y_d,
			centroid_z_d);
	}

	cudaDeviceSynchronize();

	ConvertFromLABCuda(
		colors_d,
		length,
		XR,
		YR,
		ZR,
		gamma,
		XYZtoRGBMatrix_d,
		vector_x_d,
		vector_y_d,
		vector_z_d);

	copyArrayFromDevice(colors, colors_d, memSizeInt);

	freeArray(colors_d);
	freeArray(RGBtoXYZMatrix_d);
	freeArray(XYZtoRGBMatrix_d);


	freeArray(vector_x_d);
	freeArray(vector_y_d);
	freeArray(vector_z_d);
	freeArray(centroid_x_d);
	freeArray(centroid_y_d);
	freeArray(centroid_z_d);
	freeArray(cluster);
	delete[] centroid_x_h;
	delete[] centroid_z_h;
	delete[] centroid_y_h;

	return iterations;
}