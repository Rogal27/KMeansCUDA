#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "kmeans_kernel_impl.cuh";

#include <stdio.h>

#include <helper_cuda.h>

#include <helper_functions.h>

constexpr int THREADS = 256;


void allocateArray(void** devPtr, size_t size)
{
	checkCudaErrors(cudaMalloc(devPtr, size));
}

void freeArray(void* devPtr)
{
	checkCudaErrors(cudaFree(devPtr));
}

void threadSync()
{
	checkCudaErrors(cudaDeviceSynchronize());
}

void copyArrayToDevice(void* device, const void* host, int offset, int size)
{
	checkCudaErrors(cudaMemcpy((char*)device + offset, host, size, cudaMemcpyHostToDevice));
}

void copyArrayFromDevice(void* host, const void* device, int size)
{
	checkCudaErrors(cudaMemcpy(host, device, size, cudaMemcpyDeviceToHost));
}

int CalculateBlockNumber(int length, int blockSize)
{
	return (length + blockSize - 1) / blockSize;
}

int KMeansGatherCuda(
	float* vector_x, 
	float* vector_y, 
	float* vector_z, 
	int length, 
	int k_param,
	int max_iter,
	int* cluster,
	float* centroid_x, 
	float* centroid_y, 
	float* centroid_z)
{
	//select centroids
	dim3 block_first(THREADS, 1, 1);
	dim3 grid_first(CalculateBlockNumber(length, block_first.x), 1, 1);
	dim3 block_second(THREADS, 1, 1);
	dim3 grid_second(CalculateBlockNumber(k_param, block_second.x), 1, 1);

	bool* hasCentroidChanged_h = new bool[1];
	hasCentroidChanged_h[0] = true;
	bool* hasCentroidChanged_d;
	allocateArray((void**)&hasCentroidChanged_d, sizeof(bool));
	int iterations = max_iter;

	for (size_t i = 0; i < max_iter; i++)
	{
		hasCentroidChanged_h[0] = false;
		copyArrayToDevice(hasCentroidChanged_d, hasCentroidChanged_h, 0, sizeof(bool));
		CalculateDistancesGather_kernel << <grid_first, block_first >> > (
			vector_x,
			vector_y,
			vector_z,
			length,
			k_param,
			cluster,
			centroid_x,
			centroid_y,
			centroid_z);
		CalculateNewCentroidsGather_kernel << <grid_second, block_second >> > (
			vector_x,
			vector_y,
			vector_z,
			length,
			k_param,
			cluster,
			centroid_x,
			centroid_y,
			centroid_z,
			hasCentroidChanged_d);
		;
		//threadSync();
		int a = 5;
		;
		copyArrayFromDevice(hasCentroidChanged_h, hasCentroidChanged_d, sizeof(bool));
		if (hasCentroidChanged_h[0] == false)
		{
			iterations = i;
			break;
		}
	}

	CalculateNewVectors_kernel << <grid_first, block_first >> > (
		vector_x,
		vector_y,
		vector_z,
		length,
		k_param,
		cluster,
		centroid_x,
		centroid_y,
		centroid_z);

	freeArray(hasCentroidChanged_d);
	delete [] hasCentroidChanged_h;

	return iterations;
}

int KMeansReduceByKeyCuda(
	float* vector_x,
	float* vector_y,
	float* vector_z,
	int length,
	int k_param,
	int max_iter,
	int* cluster,
	float* centroid_x,
	float* centroid_y,
	float* centroid_z)
{
	return 0;
}

int KMeansScatterCuda(
	float* vector_x,
	float* vector_y,
	float* vector_z,
	int length,
	int k_param,
	int max_iter,
	int* cluster,
	float* centroid_x,
	float* centroid_y,
	float* centroid_z)
{
	dim3 block_first(THREADS, 1, 1);
	dim3 grid_first(CalculateBlockNumber(length, block_first.x), 1, 1);
	dim3 block_second(THREADS, 1, 1);
	dim3 grid_second(CalculateBlockNumber(k_param, block_second.x), 1, 1);
	bool* hasCentroidChanged_h = new bool[1];
	hasCentroidChanged_h[0] = true;
	bool* hasCentroidChanged_d;
	allocateArray((void**)&hasCentroidChanged_d, sizeof(bool));
	int iterations = max_iter;

	for (size_t i = 0; i < max_iter; i++)
	{
		hasCentroidChanged_h[0] = false;
		copyArrayToDevice(hasCentroidChanged_d, hasCentroidChanged_h, 0, sizeof(bool));
		CalculateDistancesGather_kernel << <grid_first, block_first >> > (
			vector_x,
			vector_y,
			vector_z,
			length,
			k_param,
			cluster,
			centroid_x,
			centroid_y,
			centroid_z);
		CalculateNewCentroidsGather_kernel << <grid_second, block_second >> > (
			vector_x,
			vector_y,
			vector_z,
			length,
			k_param,
			cluster,
			centroid_x,
			centroid_y,
			centroid_z,
			hasCentroidChanged_d);
		;
		//threadSync();
		;
		copyArrayFromDevice(hasCentroidChanged_h, hasCentroidChanged_d, sizeof(bool));
		if (hasCentroidChanged_h[0] == false)
		{
			iterations = i;
			break;
		}
	}

	CalculateNewVectors_kernel << <grid_first, block_first >> > (
		vector_x,
		vector_y,
		vector_z,
		length,
		k_param,
		cluster,
		centroid_x,
		centroid_y,
		centroid_z);

	freeArray(hasCentroidChanged_d);
	delete[] hasCentroidChanged_h;

	return iterations;
}

void ConvertToLABCuda(
	int* colors,
	int length,
	float XR,
	float YR,
	float ZR,
	float gamma,
	float* RGBtoXYZMatrix,
	float* vector_x,
	float* vector_y,
	float* vector_z)
{
	dim3 block(THREADS, 1, 1);
	dim3 grid(CalculateBlockNumber(length, block.x), 1, 1);

	ConvertToLAB_kernel << <grid, block >> > (
		colors, 
		length, 
		XR,
		YR,
		ZR, 
		gamma,
		RGBtoXYZMatrix,
		vector_x, 
		vector_y,
		vector_z);

}

void ConvertFromLABCuda(
	int* colors,
	int length,
	float XR,
	float YR,
	float ZR,
	float gamma,
	float* XYZtoRGBMatrix,
	float* vector_x,
	float* vector_y,
	float* vector_z)
{
	dim3 block(THREADS, 1, 1);
	dim3 grid(CalculateBlockNumber(length, block.x), 1, 1);

	ConvertFromLAB_kernel << <grid, block >> > (
		colors, 
		length,
		XR,
		YR,
		ZR,
		gamma,
		XYZtoRGBMatrix,
		vector_x,
		vector_y, 
		vector_z);

}