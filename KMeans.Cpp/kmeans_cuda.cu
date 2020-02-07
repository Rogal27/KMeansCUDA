#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "kmeans_kernel_impl.cuh"

#include <stdio.h>

#include <helper_cuda.h>

#include <helper_functions.h>

#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/sequence.h>

constexpr int THREADS = 256;
constexpr int SCATTER_THREADS = 1024;

typedef thrust::tuple<float, float, float, int> tuple;

struct TupleSum : thrust::binary_function<tuple, tuple, tuple>
{

	__host__ __device__ tuple operator()(const tuple& a, const tuple& b)
	{
		return tuple(
			thrust::get<0>(a) + thrust::get<0>(b),
			thrust::get<1>(a) + thrust::get<1>(b),
			thrust::get<2>(a) + thrust::get<2>(b),
			thrust::get<3>(a) + thrust::get<3>(b));
	}
};

void allocateArray(void** devPtr, size_t size)
{
	checkCudaErrors(cudaMalloc(devPtr, size));
}

void freeArray(void* devPtr)
{
	checkCudaErrors(cudaFree(devPtr));
}

void memsetArray(void* devPtr, int value, size_t size)
{
	checkCudaErrors(cudaMemset(devPtr,value,size));
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

void copyArrayFromDeviceToDevice(const void* src, void* dest, int size)
{
	checkCudaErrors(cudaMemcpy(dest, src, size, cudaMemcpyDeviceToDevice));
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

	bool hasCentroidChanged_h[1] = { true };
	bool* hasCentroidChanged_d;
	allocateArray((void**)&hasCentroidChanged_d, sizeof(bool));
	int iterations = max_iter;

	for (size_t i = 0; i < max_iter; i++)
	{
		hasCentroidChanged_h[0] = false;
		copyArrayToDevice(hasCentroidChanged_d, hasCentroidChanged_h, 0, sizeof(bool));
		CalculateDistances_kernel << <grid_first, block_first >> > (
			vector_x,
			vector_y,
			vector_z,
			length,
			k_param,
			cluster,
			centroid_x,
			centroid_y,
			centroid_z);

		threadSync();

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
	//delete [] hasCentroidChanged_h;

	return iterations;
}

int KMeansReduceByKeyCuda(
	float* vector_x_orig,
	float* vector_y_orig,
	float* vector_z_orig,
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

	bool hasCentroidChanged_h[1] = { true };
	bool* hasCentroidChanged_d;
	allocateArray((void**)&hasCentroidChanged_d, sizeof(bool));
	int iterations = max_iter;

	thrust::equal_to<int> binary_pred;

	float* vector_x;
	float* vector_y;
	float* vector_z;
	int memVectorSize = length * sizeof(float);
	allocateArray((void**)&vector_x, memVectorSize);
	allocateArray((void**)&vector_y, memVectorSize);
	allocateArray((void**)&vector_z, memVectorSize);

	copyArrayFromDeviceToDevice(vector_x_orig, vector_x, memVectorSize);
	copyArrayFromDeviceToDevice(vector_y_orig, vector_y, memVectorSize);
	copyArrayFromDeviceToDevice(vector_z_orig, vector_z, memVectorSize);

	float* vector_x_sum_d;
	float* vector_y_sum_d;
	float* vector_z_sum_d;
	int* cluster_sum_d;
	int* cluster_values_d;
	int* vector_indexes;
	int memFloatSize = k_param * sizeof(float);
	int memIntSize = k_param * sizeof(int);
	int memIntVectorSize = length * sizeof(int);

	allocateArray((void**)&vector_x_sum_d, memFloatSize);
	allocateArray((void**)&vector_y_sum_d, memFloatSize);
	allocateArray((void**)&vector_z_sum_d, memFloatSize);
	allocateArray((void**)&cluster_sum_d, memIntSize);
	allocateArray((void**)&cluster_values_d, memIntSize);
	allocateArray((void**)&vector_indexes, memIntVectorSize);

	thrust::sequence(thrust::device, vector_indexes, vector_indexes + length, 0, 1);

	auto vector_iterator_sort_first = thrust::make_zip_iterator(thrust::make_tuple(vector_x, vector_y, vector_z, thrust::make_constant_iterator(1), vector_indexes));
	auto vector_iterator_first = thrust::make_zip_iterator(thrust::make_tuple(vector_x, vector_y, vector_z, thrust::make_constant_iterator(1)));
	//auto vector_iterator_first = thrust::make_zip_iterator(thrust::make_tuple(vector_x, vector_y, vector_z));
	auto output_iterator_first = thrust::make_zip_iterator(thrust::make_tuple(vector_x_sum_d, vector_y_sum_d, vector_z_sum_d, cluster_sum_d));
	//auto output_iterator_first = thrust::make_zip_iterator(thrust::make_tuple(vector_x_sum_d, vector_y_sum_d, vector_z_sum_d));

	for (size_t i = 0; i < max_iter; i++)
	{
		hasCentroidChanged_h[0] = false;
		copyArrayToDevice(hasCentroidChanged_d, hasCentroidChanged_h, 0, sizeof(bool));
		CalculateDistances_kernel << <grid_first, block_first >> > (
			vector_x,
			vector_y,
			vector_z,
			length,
			k_param,
			cluster,
			centroid_x,
			centroid_y,
			centroid_z);

		threadSync();

		thrust::sort_by_key(thrust::device, cluster, cluster + length, vector_iterator_first);

		auto result = thrust::reduce_by_key(thrust::device, cluster, cluster + length, vector_iterator_first, cluster_values_d, output_iterator_first, binary_pred, TupleSum());

		int output_length = result.first - cluster_values_d;

		dim3 block_thrust(THREADS, 1, 1);
		dim3 grid_thrust(CalculateBlockNumber(output_length, block_second.x), 1, 1);

		CalculateNewCentroidsReduceByKey_kernel << <grid_thrust, block_thrust >> > (
			centroid_x,
			centroid_y,
			centroid_z,
			hasCentroidChanged_d,
			vector_x_sum_d,
			vector_y_sum_d,
			vector_z_sum_d,
			cluster_sum_d,
			cluster_values_d,
			output_length);

		

		copyArrayFromDevice(hasCentroidChanged_h, hasCentroidChanged_d, sizeof(bool));

		if (hasCentroidChanged_h[0] == false)
		{
			iterations = i;
			break;
		}
	}

	CalculateNewVectorsReduceByKey_kernel << <grid_first, block_first >> > (
		vector_x_orig,
		vector_y_orig,
		vector_z_orig,
		length,
		k_param,
		cluster,
		centroid_x,
		centroid_y,
		centroid_z,
		vector_indexes);

	freeArray(hasCentroidChanged_d);
	freeArray(vector_x_sum_d);
	freeArray(vector_y_sum_d);
	freeArray(vector_z_sum_d);
	freeArray(cluster_sum_d);
	freeArray(cluster_values_d);
	freeArray(vector_x);
	freeArray(vector_y);
	freeArray(vector_z);
	freeArray(vector_indexes);
	//delete [] hasCentroidChanged_h;

	return iterations;
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
	dim3 block_scatter(THREADS, 1, 1);
	dim3 grid_scatter(CalculateBlockNumber(SCATTER_THREADS, block_scatter.x), 1, 1);
	bool hasCentroidChanged_h[1] = { true };
	bool* hasCentroidChanged_d;
	allocateArray((void**)&hasCentroidChanged_d, sizeof(bool));
	int iterations = max_iter;

	int memFloatSize = SCATTER_THREADS * k_param * sizeof(float);
	int memIntSize = SCATTER_THREADS * k_param * sizeof(int);

	float* scatter_array_x_d;
	float* scatter_array_y_d;
	float* scatter_array_z_d;
	int* scatter_array_count_d;

	allocateArray((void**)&scatter_array_x_d, memFloatSize);
	allocateArray((void**)&scatter_array_y_d, memFloatSize);
	allocateArray((void**)&scatter_array_z_d, memFloatSize);
	allocateArray((void**)&scatter_array_count_d, memIntSize);
	

	for (size_t i = 0; i < max_iter; i++)
	{
		hasCentroidChanged_h[0] = false;
		copyArrayToDevice(hasCentroidChanged_d, hasCentroidChanged_h, 0, sizeof(bool));
		memsetArray(scatter_array_x_d, 0, memFloatSize);
		memsetArray(scatter_array_y_d, 0, memFloatSize);
		memsetArray(scatter_array_z_d, 0, memFloatSize);
		memsetArray(scatter_array_count_d, 0, memIntSize);

		CalculateDistances_kernel << <grid_first, block_first >> > (
			vector_x,
			vector_y,
			vector_z,
			length,
			k_param,
			cluster,
			centroid_x,
			centroid_y,
			centroid_z);

		threadSync();

		CalculatePartialSumsScatter_kernel << <grid_scatter, block_scatter >> > (
			vector_x,
			vector_y,
			vector_z,
			length,
			k_param,
			cluster,
			scatter_array_x_d,
			scatter_array_y_d,
			scatter_array_z_d,
			scatter_array_count_d,
			SCATTER_THREADS);

		threadSync();

		CalculateNewCentroidsScatter_kernel << <grid_second, block_second >> > (
			k_param,
			centroid_x,
			centroid_y,
			centroid_z,
			hasCentroidChanged_d,
			scatter_array_x_d,
			scatter_array_y_d,
			scatter_array_z_d,
			scatter_array_count_d,
			SCATTER_THREADS);

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
	freeArray(scatter_array_x_d);
	freeArray(scatter_array_y_d);
	freeArray(scatter_array_z_d);
	freeArray(scatter_array_count_d);

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

