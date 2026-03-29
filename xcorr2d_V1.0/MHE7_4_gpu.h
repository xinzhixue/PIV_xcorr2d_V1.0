#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cufft.h>
#include <cuda_runtime.h>

class mhe_gpu
{
public:
	int NUMLEVELS = 256;
	int height;
	int width;
	int nr;
	int nc;
	int maxI;
	int maxJ;
	float thresh;
	bool ifdiv;
	int xs;
	int ys;
	int Nx;
	int Ny;
	int Nt_w;
	int Nt_pix;
	dim3 griddim;
	dim3 blockdim;
	int N_trash;
	int N_nontrash;
public:
	int* d_cnt_neigh;
	int* d_levels;
	int* d_transfer;
	int* d_cumsum;
	unsigned char* d_enhanced;
public:
	bool mhe_gpu::init(int Nx_org, int Ny_org, int m_mhe_ws, float m_mhe_thresh);
	void mhe_gpu::freemem();
	void mhe_gpu::setupgridblockdim(int dim1_grid, int dim2_grid, int dim3_grid, int dim1_block, int dim2_block, int dim3_block);
	bool mhe_gpu::processsing(unsigned char* d_img_org);
};