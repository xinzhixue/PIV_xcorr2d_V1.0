#include "gpuinfo.h"
using namespace std;

void gpusetup::printDevProp(cudaDeviceProp devProp)
{
	printf("Major revision number:         %d\n", devProp.major);
	printf("Minor revision number:         %d\n", devProp.minor);
	printf("Name:                          %s\n", devProp.name);
	printf("Total global memory:           %u Byte\n", devProp.totalGlobalMem);
	printf("Total shared memory per block: %u Byte\n", devProp.sharedMemPerBlock);
	printf("Total registers per block:     %d\n", devProp.regsPerBlock);
	printf("Warp size:                     %d\n", devProp.warpSize);
	printf("Maximum memory pitch:          %u\n", devProp.memPitch);
	printf("Maximum threads per block:     %d\n", devProp.maxThreadsPerBlock);
	printf("Maximum threads per multiprocessor:     %d\n", devProp.maxThreadsPerMultiProcessor);
	printf("Number of multiprocessors:     %d\n", devProp.multiProcessorCount);
	printf("Shared memory available per multiprocessor:     %d Byte\n", devProp.sharedMemPerMultiprocessor);
	for (int i = 0; i < 3; ++i)
		printf("Maximum dimension %d of block:  %d\n", i, devProp.maxThreadsDim[i]);
	for (int i = 0; i < 3; ++i)
		printf("Maximum dimension %d of grid:   %d\n", i, devProp.maxGridSize[i]);
	printf("Clock rate:                    %d kHz\n", devProp.clockRate);
	printf("Total constant memory:         %u\n", devProp.totalConstMem);
	printf("Texture alignment:             %u\n", devProp.textureAlignment);
	printf("Concurrent copy and execution: %s\n", (devProp.deviceOverlap ? "Yes" : "No"));
	printf("Kernel execution timeout:      %s\n", (devProp.kernelExecTimeoutEnabled ? "Yes" : "No"));
	return;
}

bool gpusetup::GetNofGpu()
{
	cudaGetDeviceCount(&N_device);
	if (N_device == 0)
	{
		cout << "There is No GPU on this computer!" << endl;
		cout << "Press Enter to Exit Program:";
		getchar();
		return false;
	}
	if (N_device == 1) cout << "There is ONE GPU on this computer." << endl;
	if (N_device >1) cout << "Note: there are MORE THAN ONE GPU on this computer. Choose the one for computation wisely." << endl;
	return true;
}

void gpusetup::PrintGpuProperties()
{
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	cudaGetDevice(&idx_device);
	for (int i = 0; i < N_device; i++)
	{
		printf("---------------------------------------------------------------\n");
		printf("Properties of Device %d:\n", i);
		cudaSetDevice(i);
		cudaGetDeviceProperties(&prop, 0);
		printDevProp(prop);
		printf("---------------------------------------------------------------\n");
	}
	cudaSetDevice(idx_device);
}

void gpusetup::AssignGpu(int device_id)
{
	idx_device = device_id;
	cudaSetDevice(device_id);
}

void gpusetup::meminfo()
{
	size_t free, total;
	float free_m, total_m, used_m;
	for (int i = 0; i < N_device; i++)
	{
		printf("---------------------------------------------------------------\n");
		printf("Global memory of Device %d:\n", i);
		cudaSetDevice(i);
		cudaMemGetInfo(&free, &total);
		free_m = (float)free / 1048576.0;
		total_m = (float)total / 1048576.0;
		used_m = total_m - free_m;
		printf("Free: %.1f MB\n", free_m);
		printf("Used: %.1f MB\n", used_m);
		printf("Total: %.1f MB\n", total_m);
		printf("---------------------------------------------------------------\n");
	}
	cudaSetDevice(idx_device);
}