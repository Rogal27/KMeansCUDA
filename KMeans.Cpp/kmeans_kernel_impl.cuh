#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

constexpr float EPS = 1e-4f;

__device__ float Dist(const float& x1, const float& y1, const float& z1, const float& x2, const float& y2, const float& z2)
{
	float dx = x1 - x2;
	float dy = y1 - y2;
	float dz = z1 - z2;
	return dx * dx + dy * dy + dz * dz;
}

__global__ void CalculateDistancesGather_kernel(
	float* vector_x,
	float* vector_y,
	float* vector_z,
	int length,
	int k_param,
	int* cluster,
	float* centroid_x,
	float* centroid_y,
	float* centroid_z)
{
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < length)
	{
		float minDist = Dist(vector_x[index], vector_y[index], vector_z[index], centroid_x[0], centroid_y[0], centroid_z[0]);
		int minIndex = 0;
		float dist;
		for (size_t i = 1; i < k_param; i++)
		{
			dist = Dist(vector_x[index], vector_y[index], vector_z[index], centroid_x[i], centroid_y[i], centroid_z[i]);
			if (minDist > dist)
			{
				minDist = dist;
				minIndex = i;
			}
		}
		cluster[index] = minIndex;
	}
}

__global__ void CalculateNewCentroidsGather_kernel(
	float* vector_x,
	float* vector_y,
	float* vector_z,
	int length,
	int k_param,
	int* cluster,
	float* centroid_x,
	float* centroid_y,
	float* centroid_z,
	bool* changed)
{
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	bool hasChanged = false;
	if (index < k_param)
	{
		float dist_sum_x = 0.0f;
		float dist_sum_y = 0.0f;
		float dist_sum_z = 0.0f;
		int sum = 0;
		for (size_t i = 0; i < length; i++)
		{
			if (cluster[i] == index)
			{
				dist_sum_x += vector_x[i];
				dist_sum_y += vector_y[i];
				dist_sum_z += vector_z[i];
				sum++;
			}
		}
		
		if (sum != 0)
		{
			dist_sum_x /= sum;
			dist_sum_y /= sum;
			dist_sum_z /= sum;
			float dist = Dist(dist_sum_x, dist_sum_y, dist_sum_z, centroid_x[index], centroid_y[index], centroid_z[index]);
			if (dist > EPS)
			{
				centroid_x[index] = dist_sum_x;
				centroid_y[index] = dist_sum_y;
				centroid_z[index] = dist_sum_z;
				hasChanged = true;
				changed[0] = true;
			}
		}
	}
}

__global__ void CalculateNewVectors_kernel(
	float* vector_x,
	float* vector_y,
	float* vector_z,
	int length,
	int k_param,
	int* cluster,
	float* centroid_x,
	float* centroid_y,
	float* centroid_z)
{
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < length)
	{
		vector_x[index] = centroid_x[cluster[index]];
		vector_y[index] = centroid_y[cluster[index]];
		vector_z[index] = centroid_z[cluster[index]];
	}
}