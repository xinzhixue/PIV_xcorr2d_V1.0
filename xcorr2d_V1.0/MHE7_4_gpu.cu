#include "MHE7_4_gpu.h"

using namespace std;

__global__ void prescan_kernel(int *g_idata, int *g_odata)
{
	extern __shared__ int temp[];// allocated on invocation
	int thid = threadIdx.x;
	int idx_st = blockIdx.x*blockDim.x*2;
	int offset = 1;
	int n = blockDim.x * 2;
	
	temp[2 * thid] = g_idata[2 * thid + idx_st]; // load input into shared memory
	temp[2 * thid + 1] = g_idata[2 * thid + 1 + idx_st];
	for (int d = n >> 1; d > 0; d >>= 1) // build sum in place up the tree
	{
		__syncthreads();
		if (thid < d)
		{
			int ai = offset*(2 * thid + 1) - 1;
			int bi = offset*(2 * thid + 2) - 1;
			temp[bi] += temp[ai];
		}
		offset *= 2;
	}
	if (thid == 0) { temp[n - 1] = 0; } // clear the last element
	for (int d = 1; d < n; d *= 2) // traverse down tree & build scan
	{
		offset >>= 1;
		__syncthreads();
		if (thid < d)
		{
			int ai = offset*(2 * thid + 1) - 1;
			int bi = offset*(2 * thid + 2) - 1;
			int t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}
	__syncthreads();
	g_odata[2 * thid + idx_st] = temp[2 * thid + 1]; // write results to device memory
	g_odata[2 * thid + 1 + idx_st] = temp[2 * thid + 2];
}

__global__ void Calc_cnt_neigh_kernel(int* d_cnt, int Nx, int Ny, int Nt)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	int stride = blockDim.x * gridDim.x;
	while (tid < Nt)
	{
		int j = tid / Nx;
		int i = tid - j*Nx;
		if (i >= 1 && i <= Nx - 2 && j >= 1 && j <= Ny - 2){
			d_cnt[tid] = 9;
		}
		else if ((i == 0 && j == 0) || (i == 0 && j == Ny - 1) || (i == Nx - 1 && j == 0) || (i == Nx - 1 && j == Ny - 1)){
			d_cnt[tid] = 4;
		}
		else{
			d_cnt[tid] = 6;
		}
		tid += stride;
	}
}

__global__ void MakeHist_kernel(int* d_levels, unsigned char* d_img_org, int Nx_img_org, int Nx_img, int ws, int Nx_w, int xs, int ys, int Nt)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	int stride = blockDim.x * gridDim.x;
	while (tid < Nt)
	{
		int idy = tid / Nx_img;
		int idx = tid - idy*Nx_img;
		int index_pix = (int)d_img_org[(idy + ys)*Nx_img_org + xs + idx];
		int idy_w = idy / ws;
		int idx_w = idx / ws;
		int index = (Nx_w+2)*(idy_w+1) + 1 + idx_w;

		atomicAdd(&d_levels[index*256+index_pix], 1);
		
		tid += stride;
	}
}

__global__ void CreateTransfer_kernel(int* d_levels, int* d_transfer, int Nx_w, int Nt)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	int stride = blockDim.x * gridDim.x;
	while (tid < Nt)
	{
		int idw = tid / 256;
		int id_pix = tid - idw * 256;
		int idy = idw / Nx_w;
		int idx = idw - idy*Nx_w;
		int index = (idy + 1)*(Nx_w + 2) + 1 + idx;
		d_transfer[tid] = d_levels[index*256+id_pix] + d_levels[(index - 1)*256+id_pix] + d_levels[(index + 1)*256+id_pix] + d_levels[(index - Nx_w - 2)*256+id_pix] +
		 d_levels[(index - Nx_w - 1)*256+id_pix] + d_levels[(index - Nx_w - 3)*256+id_pix] + d_levels[(index + Nx_w + 2)*256+id_pix] + d_levels[(index + Nx_w + 1)*256+id_pix] + 
		 d_levels[(index + Nx_w + 3)*256+id_pix];
		tid += stride;
	}
}

__global__ void Calc_Enhanced_value_kernel(int* d_cumsum, int* d_cnt_neigh, int Ntrash, int Nnontrash, int Nt)
{
	int thid = threadIdx.x;
	int tid = thid + blockIdx.x*blockDim.x;
	int stride = blockDim.x * gridDim.x;
	while (tid < Nt)
	{
		int idw = tid / 256;
		int neighcnt = d_cnt_neigh[idw];
		d_cumsum[tid] = d_cumsum[tid] - neighcnt*Ntrash;
		d_cumsum[tid] = (d_cumsum[tid] >= 0) ? d_cumsum[tid] : 0;
		d_cumsum[tid] = min(255, d_cumsum[tid] * 255 / (neighcnt*Nnontrash));
		if (thid == 0) d_cumsum[tid] = 0;
		if (thid == 255) d_cumsum[tid] = 255;
		tid += stride;
	}
}

__global__ void Enhanced_kernel(int* d_cumsum, unsigned char* d_img_org, unsigned char* d_enhanced, int Nx_org, int Nx_enhanced, int Nx_w, int ws, int xs, int ys, int Nt)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	int stride = blockDim.x * gridDim.x;
	while (tid < Nt)
	{
		int idy = tid / Nx_enhanced;
		int idx = tid - idy*Nx_enhanced;
		int idy_w = idy / ws;
		int idx_w = idx / ws;
		int index_w = idy_w*Nx_w + idx_w;
		int index_pix = (idy + ys)*Nx_org + xs + idx;
		int index_org = (int)d_img_org[(idy + ys)*Nx_org + xs + idx];
		d_enhanced[index_pix] = (unsigned char)d_cumsum[index_w * 256 + (int)d_img_org[index_pix]];
		tid += stride;
	}
}

void mhe_gpu::setupgridblockdim(int dim1_grid, int dim2_grid, int dim3_grid, int dim1_block, int dim2_block, int dim3_block)
{
	griddim.x = dim1_grid;
	griddim.y = dim2_grid;
	griddim.z = dim3_grid;
	blockdim.x = dim1_block;
	blockdim.y = dim2_block;
	blockdim.z = dim3_block;
}

bool mhe_gpu::init(int Nx_org, int Ny_org, int m_mhe_ws, float m_mhe_thresh)
{
	int tmp;
	nr = m_mhe_ws;
	nc = m_mhe_ws;
	thresh = m_mhe_thresh;
	Nx = Nx_org;
	Ny = Ny_org;
	if (Nx_org%m_mhe_ws == 0 && Ny_org%m_mhe_ws == 0){
		ifdiv = true;
		height = Ny_org;
		width = Nx_org;
		xs = 0;
		ys = 0;
	}
	else{
		ifdiv = false;
		tmp = Nx_org / m_mhe_ws;
		width = m_mhe_ws*tmp;
		xs = (Nx_org - width) / 2;
		tmp = Ny_org / m_mhe_ws;
		height = m_mhe_ws*tmp;
		ys = (Ny_org - height) / 2;
	}
	maxI = height / nr;
	maxJ = width / nc;
	Nt_w = maxI*maxJ;
	Nt_pix = height*width;
	N_trash = (int)roundf((float)(nr*nc)*thresh);
	N_nontrash = nr*nc - N_trash;

	if (cudaMalloc((void**)&d_levels, sizeof(int)*(maxI+2)*(maxJ+2)*NUMLEVELS) != cudaSuccess){
		cout << "cudaMalloc ERROR!" << endl;
		return false;
	}

	if (cudaMalloc((void**)&d_transfer, sizeof(int)*maxI*maxJ*NUMLEVELS) != cudaSuccess){
		cout << "cudaMalloc ERROR!" << endl;
		return false;
	}
	if (cudaMalloc((void**)&d_cumsum, sizeof(int)*maxI*maxJ*NUMLEVELS) != cudaSuccess){
		cout << "cudaMalloc ERROR!" << endl;
		return false;
	}
	if (cudaMalloc((void**)&d_enhanced, sizeof(unsigned char)*Nx*Ny) != cudaSuccess){
		cout << "cudaMalloc ERROR!" << endl;
		return false;
	}
	cudaMemset(d_enhanced, 0, sizeof(unsigned char)*Nx*Ny);
	cudaMemset(d_transfer, 0, sizeof(int)*maxI*maxJ*NUMLEVELS);

	if (cudaMalloc((void**)&d_cnt_neigh, sizeof(int)*Nt_w) != cudaSuccess){
		cout << "cudaMalloc ERROR!" << endl;
		return false;
	}
	Calc_cnt_neigh_kernel << <griddim, blockdim >> >(d_cnt_neigh, maxJ, maxI, Nt_w);
	if (cudaPeekAtLastError() != cudaSuccess)
	{
		printf("GPU kernel error at Calc_cnt_neigh_kernel: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
		return false;
	}
	
	return true;
}

bool mhe_gpu::processsing(unsigned char* d_img_org)
{	
	cudaMemset(d_levels, 0, sizeof(int)*(maxI + 2)*(maxJ + 2)*NUMLEVELS);
	
	MakeHist_kernel << <griddim, blockdim >> >(d_levels, d_img_org, Nx, width, nr, maxJ, xs, ys, Nt_pix);
	if (cudaPeekAtLastError() != cudaSuccess)
	{
		printf("GPU kernel error at MakeHist_kernel: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
		return false;
	}

	CreateTransfer_kernel << <griddim, blockdim >> >(d_levels, d_transfer, maxJ, Nt_w * NUMLEVELS);
	if (cudaPeekAtLastError() != cudaSuccess)
	{
		printf("GPU kernel error at CreateTransfer_kernel: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
		return false;
	}

	prescan_kernel << <maxI*maxJ, NUMLEVELS / 2, sizeof(int)*NUMLEVELS*2 >> >(d_transfer, d_cumsum);
	if (cudaPeekAtLastError() != cudaSuccess)
	{
		printf("GPU kernel error at prescan_kernel: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
		return false;
	}

	Calc_Enhanced_value_kernel << <maxI*maxJ, NUMLEVELS >> >(d_cumsum, d_cnt_neigh, N_trash, N_nontrash, Nt_w*NUMLEVELS);
	if (cudaPeekAtLastError() != cudaSuccess)
	{
		printf("GPU kernel error at Calc_Enhanced_value_kernel: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
		return false;
	}

	Enhanced_kernel << <griddim, blockdim >> >(d_cumsum, d_img_org, d_enhanced, Nx, width, maxJ, nr, xs, ys, Nt_pix);
	if (cudaPeekAtLastError() != cudaSuccess)
	{
		printf("GPU kernel error at Enhanced_kernel: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
		return false;
	}
	return true;
}

void mhe_gpu::freemem()
{
	cudaFree(d_transfer);
	cudaFree(d_levels);
	cudaFree(d_enhanced);
	cudaFree(d_cnt_neigh);
	cudaFree(d_cumsum);
}