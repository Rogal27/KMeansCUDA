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
	int mask = 255;
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
	float xr;
	float yr;
	float zr;

	float fy = (L + 16.0f) / 116.0f;
	float fx = a / 500.0f + fy;
	float fz = fy - b / 200.0f;


	xr = powf(fx, 3.0f);
	if (xr <= eps_modifier)
	{
		xr = (116.0f * fx - 16.0f) / k_modifier;
	}

	if (L > k_modifier* eps_modifier)
	{
		yr = powf((L + 16.0f) / 116.0f, 3.0f);
	}
	else
	{
		yr = L / k_modifier;
	}

	zr = powf(fz, 3.0f);
	if (zr <= eps_modifier)
	{
		zr = (116.0f * fz - 16.0f) / k_modifier;
	}

	float x = xr * XR;
	float y = yr * YR;
	float z = zr * ZR;


	L = XYZtoRGBmatrix[0] * x + XYZtoRGBmatrix[1] * y + XYZtoRGBmatrix[2] * z;
	a = XYZtoRGBmatrix[3] * x + XYZtoRGBmatrix[4] * y + XYZtoRGBmatrix[5] * z;
	b = XYZtoRGBmatrix[6] * x + XYZtoRGBmatrix[7] * y + XYZtoRGBmatrix[8] * z;
	

	float inv_gamma = 1.0f / gamma;

	L = powf(L, inv_gamma);
	a = powf(a, inv_gamma);
	b = powf(b, inv_gamma);

	L *= 255.0f;
	a *= 255.0f;
	b *= 255.0f;
	
	if (L < 0.0f)
		L = 0.0f;
	if (a < 0.0f)
		a = 0.0f;
	if (b < 0.0f)
		b = 0.0f;

	if (L > 255.0f)
		L = 255.0f;
	if (a > 255.0f)
		a = 255.0f;
	if (b > 255.0f)
		b = 255.0f;


	unsigned char R = (unsigned char)L;
	unsigned char G = (unsigned char)a;
	unsigned char B = (unsigned char)b;

	color = 0;
	color |= 255 << 24;
	color |= R << 16; // R
	color |= G << 8; // G
	color |= B; //B
}

__global__ void CalculateDistances_kernel(
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
	if (index < k_param)
	{
		float dist_sum_x = 0.0f;
		float dist_sum_y = 0.0f;
		float dist_sum_z = 0.0f;
		long sum = 0;
		for (long i = 0; i < length; i++)
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

__global__ void ConvertToLAB_kernel(
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
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < length)
	{
		ConvertToLABOneColor(colors[index], vector_x[index], vector_y[index], vector_z[index], RGBtoXYZMatrix, XR, YR, ZR, gamma);
	}
}

__global__ void ConvertFromLAB_kernel(
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
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < length)
	{
		ConvertFromLABOneColor(colors[index], vector_x[index], vector_y[index], vector_z[index], XYZtoRGBMatrix, XR, YR, ZR, gamma);
	}
}

__global__ void CalculatePartialSumsScatter_kernel(
	float* vector_x,
	float* vector_y,
	float* vector_z,
	int length,
	int k_param,
	int* cluster,
	float* scatter_array_x,
	float* scatter_array_y,
	float* scatter_array_z,
	int* scatter_array_count,
	int scatter_threads_count)
{
	int gap = blockDim.x * gridDim.x;
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	
	for (size_t i = index; i < length; i += gap)
	{
		int row = scatter_threads_count * cluster[i];
		scatter_array_x[index + row] += vector_x[i];
		scatter_array_y[index + row] += vector_y[i];
		scatter_array_z[index + row] += vector_z[i];
		scatter_array_count[index + row]++;
	}
}

__global__ void CalculateNewCentroidsScatter_kernel(
	int k_param,
	float* centroid_x,
	float* centroid_y,
	float* centroid_z,
	bool* changed,
	float* scatter_array_x,
	float* scatter_array_y,
	float* scatter_array_z,
	int* scatter_array_count,
	int scatter_threads_count)
{
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < k_param)
	{
		float dist_sum_x = 0.0f;
		float dist_sum_y = 0.0f;
		float dist_sum_z = 0.0f;
		int row = scatter_threads_count * index;
		int sum = 0;
		for (size_t i = 0; i < scatter_threads_count; i++)
		{
			dist_sum_x += scatter_array_x[row + i];
			dist_sum_y += scatter_array_y[row + i];
			dist_sum_z += scatter_array_z[row + i];
			sum += scatter_array_count[row + i];
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

				changed[0] = true;
			}
		}
	}
}

__global__ void CalculateNewCentroidsReduceByKey_kernel(
	float* centroid_x,
	float* centroid_y,
	float* centroid_z,
	bool* changed,
	float* vector_x_sum,
	float* vector_y_sum,
	float* vector_z_sum,
	int* cluster_sum,
	int* cluster_keys,
	int cluster_keys_length)
{
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < cluster_keys_length)
	{
		int cluster_nr = cluster_keys[index];

		if (cluster_sum[index] != 0)
		{
			float dist_sum_x = vector_x_sum[index] / cluster_sum[index];
			float dist_sum_y = vector_y_sum[index] / cluster_sum[index];
			float dist_sum_z = vector_z_sum[index] / cluster_sum[index];
			float dist = Dist(dist_sum_x, dist_sum_y, dist_sum_z, centroid_x[cluster_nr], centroid_y[cluster_nr], centroid_z[cluster_nr]);
			if (dist > EPS)
			{
				centroid_x[cluster_nr] = dist_sum_x;
				centroid_y[cluster_nr] = dist_sum_y;
				centroid_z[cluster_nr] = dist_sum_z;

				changed[0] = true;
			}
		}
	}
}

__global__ void CalculateNewVectorsReduceByKey_kernel(
	float* vector_x,
	float* vector_y,
	float* vector_z,
	int length,
	int k_param,
	int* cluster,
	float* centroid_x,
	float* centroid_y,
	float* centroid_z,
	int* vector_indexes)
{
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < length)
	{
		vector_x[index] = centroid_x[cluster[vector_indexes[index]]];
		vector_y[index] = centroid_y[cluster[vector_indexes[index]]];
		vector_z[index] = centroid_z[cluster[vector_indexes[index]]];
	}
}

//__global__ void CalculateNewVectorsReduceByKey_kernel(
//	float* vector_x,
//	float* vector_y,
//	float* vector_z,
//	int length,
//	int k_param,
//	int* cluster,
//	float* centroid_x,
//	float* centroid_y,
//	float* centroid_z)
//{
//	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
//	if (index < length)
//	{
//		float minDist = Dist(vector_x[index], vector_y[index], vector_z[index], centroid_x[0], centroid_y[0], centroid_z[0]);
//		int minIndex = 0;
//		float dist;
//		for (size_t i = 1; i < k_param; i++)
//		{
//			dist = Dist(vector_x[index], vector_y[index], vector_z[index], centroid_x[i], centroid_y[i], centroid_z[i]);
//			if (minDist > dist)
//			{
//				minDist = dist;
//				minIndex = i;
//			}
//		}
//		vector_x[index] = centroid_x[minIndex];
//		vector_y[index] = centroid_y[minIndex];
//		vector_z[index] = centroid_z[minIndex];
//	}
//}