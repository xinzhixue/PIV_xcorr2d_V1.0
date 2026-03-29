#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cufft.h>
#include <cuda_runtime.h>
#include <iostream>

class gpusetup
{
public:
	int  N_device;
	int idx_device;
public:

	void gpusetup::AssignGpu(int device_id);
	bool gpusetup::GetNofGpu();
	void gpusetup::PrintGpuProperties();
	void gpusetup::meminfo();
protected:
	void gpusetup::printDevProp(cudaDeviceProp devProp);
};