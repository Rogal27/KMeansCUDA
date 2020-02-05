void allocateArray(void** devPtr, size_t size);
void freeArray(void* devPtr);
void threadSync();
void copyArrayToDevice(void* device, const void* host, int offset, int size);
void copyArrayFromDevice(void* host, const void* device, int size);
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
	float* centroid_z);
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
	float* vector_z);
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
	float* vector_z);