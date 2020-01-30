#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

constexpr float EPS = 1e-4f;
constexpr float k_modifier = 903.3f;
constexpr float eps_modifier = 0.008856f;

__device__ float Dist(const float& x1, const float& y1, const float& z1, const float& x2, const float& y2, const float& z2)
{
	float dx = x1 - x2;
	float dy = y1 - y2;
	float dz = z1 - z2;
	return dx * dx + dy * dy + dz * dz;
}

__device__ void ConvertToLABOneColor(int& color, float& L, float& a, float& b, float* RGBtoXYZmatrix, float& XR, float& YR, float& ZR, float& gamma)
{
	unsigned char mask = 255;
	unsigned char R = (unsigned char)((color & (mask << 16)) >> 16);
	unsigned char G = (unsigned char)((color & (mask << 8)) >> 8);
	unsigned char B = (unsigned char)(color & mask);
	float x = (float)R / 255.0f;
	float y = (float)G / 255.0f;
	float z = (float)B / 255.0f;

	//inverse gamma correction
	x = powf(x, gamma);
	y = powf(y, gamma);
	z = powf(z, gamma);

	//to xyz (multiply by matrix)
	float xr = RGBtoXYZmatrix[0] * x + RGBtoXYZmatrix[1] * y + RGBtoXYZmatrix[2] * z;
	float yr = RGBtoXYZmatrix[3] * x + RGBtoXYZmatrix[4] * y + RGBtoXYZmatrix[5] * z;
	float zr = RGBtoXYZmatrix[6] * x + RGBtoXYZmatrix[7] * y + RGBtoXYZmatrix[8] * z;

	xr = xr / XR;
	yr = yr / YR;
	zr = zr / ZR;

	float fx;
	float fy;
	float fz;

	if (xr > eps_modifier)
	{
		fx = powf(xr, 1.0f / 3.0f);
	}
	else
	{
		fx = (k_modifier * xr + 16.0f) / 116.0f;
	}

	if (yr > eps_modifier)
	{
		fy = powf(yr, 1.0f / 3.0f);
	}
	else
	{
		fy = (k_modifier * yr + 16.0f) / 116.0f;
	}

	if (zr > eps_modifier)
	{
		fz = powf(zr, 1.0f / 3.0f);
	}
	else
	{
		fz = (k_modifier * zr + 16.0f) / 116.0f;
	}
	
	L = 116.0f * fy - 16.0f;
	a = 500.0f * (fx - fy);
	b = 200.0f * (fy - fz);
}

__device__ void ConvertFromLABOneColor(int& color, float& L, float& a, float& b, float* XYZtoRGBmatrix, float& XR, float& YR, float& ZR, float& gamma)
{
	float xr = L / XR;
	float yr = a / YR;
	float zr = b / ZR;

	float fx;
	float fy;
	float fz;

	if (xr > eps_modifier)
	{
		fx = powf(xr, 1.0f / 3.0f);
	}
	else
	{
		fx = (k_modifier * xr + 16.0f) / 116.0f;
	}

	if (yr > eps_modifier)
	{
		fy = powf(yr, 1.0f / 3.0f);
	}
	else
	{
		fy = (k_modifier * yr + 16.0f) / 116.0f;
	}

	if (zr > eps_modifier)
	{
		fz = powf(zr, 1.0f / 3.0f);
	}
	else
	{
		fz = (k_modifier * zr + 16.0f) / 116.0f;
	}

	float x = 116.0f * fy - 16.0f;
	float y = 500.0f * (fx - fy);
	float z = 200.0f * (fy - fz);


	L = XYZtoRGBmatrix[0] * x + XYZtoRGBmatrix[1] * y + XYZtoRGBmatrix[2] * z;
	a = XYZtoRGBmatrix[3] * x + XYZtoRGBmatrix[4] * y + XYZtoRGBmatrix[5] * z;
	b = XYZtoRGBmatrix[6] * x + XYZtoRGBmatrix[7] * y + XYZtoRGBmatrix[8] * z;


	float inv_gamma = 1.0f / gamma;

	L = powf(L, inv_gamma);
	a = powf(a, inv_gamma);
	b = powf(b, inv_gamma);
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
				//changed[0] = true;
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

__global__ void ConvertToLAB_kernel(
	int* colors,
	int length,
	float XR,
	float YR,
	float ZR,
	float* vector_x,
	float* vector_y,
	float* vector_z)
{
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < length)
	{

	}
}