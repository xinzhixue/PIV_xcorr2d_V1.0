#include "xcorr2d_multi_grid.cuh"
using namespace std;

template <int blockSize>
__device__ void warp_max_index(volatile float* sdata, volatile int* index, int tid) {
	if (blockSize >= 64) {
		index[tid] = (sdata[tid] >= sdata[tid + 32]) ? index[tid] : index[tid + 32];
		sdata[tid] = fmaxf(sdata[tid], sdata[tid + 32]);
	}
	if (blockSize >= 32) {
		index[tid] = (sdata[tid] >= sdata[tid + 16]) ? index[tid] : index[tid + 16];
		sdata[tid] = fmaxf(sdata[tid], sdata[tid + 16]);
	}
	if (blockSize >= 16) {
		index[tid] = (sdata[tid] >= sdata[tid + 8]) ? index[tid] : index[tid + 8];
		sdata[tid] = fmaxf(sdata[tid], sdata[tid + 8]);
	}
	if (blockSize >= 8) {
		index[tid] = (sdata[tid] >= sdata[tid + 4]) ? index[tid] : index[tid + 4];
		sdata[tid] = fmaxf(sdata[tid], sdata[tid + 4]);
	}
	if (blockSize >= 4) {
		index[tid] = (sdata[tid] >= sdata[tid + 2]) ? index[tid] : index[tid + 2];
		sdata[tid] = fmaxf(sdata[tid], sdata[tid + 2]);
	}
	if (blockSize >= 2) {
		index[tid] = (sdata[tid] >= sdata[tid + 1]) ? index[tid] : index[tid + 1];
		sdata[tid] = fmaxf(sdata[tid], sdata[tid + 1]);
	}
}

template <int blockSize>
__global__ void reduce_max_index_general(float* g_idata, int* g_iindex, float* g_odata, int* g_oindex, int ws_quarter) {
	__shared__ float sdata[blockSize * 2];
	__shared__ int index[blockSize * 2];
	int tid = threadIdx.x;
	int idx_base = blockIdx.x*ws_quarter;

	index[tid] = tid;
	__syncthreads();
	if (tid < ws_quarter){
		sdata[tid] = g_idata[idx_base + tid];
	}
	else{
		sdata[tid] = 0;
	}
	__syncthreads();

	if (blockSize >= 1024) {
		if (tid < 512) {
			index[tid] = (sdata[tid] >= sdata[tid + 512]) ? index[tid] : index[tid + 512];
		}
		__syncthreads();
		if (tid < 512) {
			sdata[tid] = fmaxf(sdata[tid], sdata[tid + 512]);
		}
		__syncthreads();
	}

	if (blockSize >= 512) {
		if (tid < 256) {
			index[tid] = (sdata[tid] >= sdata[tid + 256]) ? index[tid] : index[tid + 256];
		}
		__syncthreads();
		if (tid < 256) {
			sdata[tid] = fmaxf(sdata[tid], sdata[tid + 256]);
		}
		__syncthreads();
	}

	if (blockSize >= 256) {
		if (tid < 128) {
			index[tid] = (sdata[tid] >= sdata[tid + 128]) ? index[tid] : index[tid + 128];
		}
		__syncthreads();
		if (tid < 128) {
			sdata[tid] = fmaxf(sdata[tid], sdata[tid + 128]);
		}
		__syncthreads();
	}

	if (blockSize >= 128) {
		if (tid < 64) {
			index[tid] = (sdata[tid] >= sdata[tid + 64]) ? index[tid] : index[tid + 64];
		}
		__syncthreads();
		if (tid < 64) {
			sdata[tid] = fmaxf(sdata[tid], sdata[tid + 64]);
		}
		__syncthreads();
	}

	if (tid < 32) {
		warp_max_index<blockSize>(sdata, index, tid);
	}

	if (tid == 0) {
		g_odata[blockIdx.x] = sdata[0];
		g_oindex[blockIdx.x] = g_iindex[idx_base + min(ws_quarter - 1, index[0])];
	}
}

template <int blockSize>
__global__ void reduce_max_index_final(float* g_idata, int* g_iindex, float* g_odata, int* g_oindex, int ws_quarter, int idx_peak, int Npeak_save) {
	__shared__ float sdata[blockSize * 2];
	__shared__ int index[blockSize * 2];
	int tid = threadIdx.x;
	int idx_base = blockIdx.x*ws_quarter;

	index[tid] = tid;
	__syncthreads();
	if (tid < ws_quarter){
		sdata[tid] = g_idata[idx_base + tid];
	}
	else{
		sdata[tid] = 0;
	}
	__syncthreads();

	if (blockSize >= 1024) {
		if (tid < 512) {
			index[tid] = (sdata[tid] >= sdata[tid + 512]) ? index[tid] : index[tid + 512];
		}
		__syncthreads();
		if (tid < 512) {
			sdata[tid] = fmaxf(sdata[tid], sdata[tid + 512]);
		}
		__syncthreads();
	}

	if (blockSize >= 512) {
		if (tid < 256) {
			index[tid] = (sdata[tid] >= sdata[tid + 256]) ? index[tid] : index[tid + 256];
		}
		__syncthreads();
		if (tid < 256) {
			sdata[tid] = fmaxf(sdata[tid], sdata[tid + 256]);
		}
		__syncthreads();
	}

	if (blockSize >= 256) {
		if (tid < 128) {
			index[tid] = (sdata[tid] >= sdata[tid + 128]) ? index[tid] : index[tid + 128];
		}
		__syncthreads();
		if (tid < 128) {
			sdata[tid] = fmaxf(sdata[tid], sdata[tid + 128]);
		}
		__syncthreads();
	}

	if (blockSize >= 128) {
		if (tid < 64) {
			index[tid] = (sdata[tid] >= sdata[tid + 64]) ? index[tid] : index[tid + 64];
		}
		__syncthreads();
		if (tid < 64) {
			sdata[tid] = fmaxf(sdata[tid], sdata[tid + 64]);
		}
		__syncthreads();
	}

	if (tid < 32) {
		warp_max_index<blockSize>(sdata, index, tid);
	}

	if (tid == 0) {
		g_odata[blockIdx.x * Npeak_save + idx_peak] = sdata[0];
		g_oindex[blockIdx.x * Npeak_save + idx_peak] = g_iindex[idx_base + min(ws_quarter - 1, index[0])];
	}
}

__global__ void Calc_median(float* d_median, bool* d_valid_median, int* d_neighbor_cnt, float* d_data_neigh, int Nt_median, int Nt)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	int stride = blockDim.x * gridDim.x;
	while (tid < Nt){
		int offset = tid*Nt_median;
		int Nneigh = d_neighbor_cnt[tid];
		int cnt = Nneigh;
		float tmp1, tmp2;
		if (d_valid_median[tid]){
			for (int i = 0; i < (Nneigh + 2) / 2; i++){
				cnt -= 1;
				tmp1 = d_data_neigh[offset];
				for (int j = 0; j < cnt; j++){
					tmp2 = d_data_neigh[offset + 1 + j];
					if (tmp1>tmp2){
						d_data_neigh[offset + 1 + j] = tmp1;
						d_data_neigh[offset + j] = tmp2;
					}
					else{
						tmp1 = tmp2;
					}
				}
			}
			if (Nneigh % 2 == 0){
				d_median[tid] = (d_data_neigh[offset + Nneigh / 2] + d_data_neigh[offset + Nneigh / 2 - 1]) / 2;
			}
			else{
				d_median[tid] = d_data_neigh[offset + (Nneigh - 1) / 2];
			}
		}
		tid += stride;
	}
}

__global__ void Calc_devi(float* d_data_neigh, bool* d_valid_median, int* d_neighbor_cnt, float* d_median, float* d_devi_neigh, int Nt_median, int Nt)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	int stride = blockDim.x * gridDim.x;
	while (tid < Nt){
		int id = tid / Nt_median;
		if (d_valid_median[id]){
			int tmp = tid - id*Nt_median;
			if (tmp < d_neighbor_cnt[id]){
				d_devi_neigh[tid] = fabsf(d_data_neigh[tid] - d_median[id]);
			}
		}
		tid += stride;
	}
}

__global__ void median_filtering(float* d_disp_x, float* d_disp_y, float* d_median_x, float* d_median_y, float* d_devimedian_x, float* d_devimedian_y, bool* d_valid_median, bool* d_valid,
	bool* d_valid_filter, float* d_disp_x_filter, float* d_disp_y_filter, float devi_threshold, float epsilon, int Nt, int Npeak_save)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	int stride = blockDim.x * gridDim.x;
	while (tid < Nt){
		d_valid_filter[tid] = false;
		d_disp_x_filter[tid] = 0;
		d_disp_y_filter[tid] = 0;
		float devix, deviy;
		int i;
		if (d_valid_median[tid]){
			for (i = 0; i < Npeak_save; i++){
				if (!d_valid[tid*Npeak_save + i]){
					continue;
				}
				devix = fabsf(d_disp_x[tid * Npeak_save + i] - d_median_x[tid]) / (d_devimedian_x[tid] + epsilon);
				deviy = fabsf(d_disp_y[tid * Npeak_save + i] - d_median_y[tid]) / (d_devimedian_y[tid] + epsilon);
				if (devix < devi_threshold && deviy < devi_threshold){
					d_valid_filter[tid] = true;
					d_disp_x_filter[tid] = d_disp_x[tid * Npeak_save + i];
					d_disp_y_filter[tid] = d_disp_y[tid * Npeak_save + i];
					break;
				}
			}
		}
		tid += stride;
	}
}

__global__ void organize_median_filter_data(float* d_disp_x, float* d_disp_y, bool* d_valid, float* d_disp_x_neigh, float* d_disp_y_neigh, int* d_neighbor_cnt,
	int Nx_vec, int Ny_vec, int wx, int hfwx, int hfwy, int Nt_median, int Nt)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	int stride = blockDim.x * gridDim.x;
	while (tid < Nt)
	{
		int id = tid / Nt_median;
		int idy = id / Nx_vec;
		int idx = id - idy*Nx_vec;
		int tmp = tid - id*Nt_median;
		int idx_win, idy_win, index, tmpidx;
		if (tmp != (Nt_median - 1) / 2){
			idy_win = tmp / wx;
			idx_win = tmp - idy_win*wx;
			idx = idx - hfwx + idx_win;
			idy = idy - hfwy + idy_win;
			if (idx >= 0 && idx <= Nx_vec - 1 && idy >= 0 && idy <= Ny_vec - 1){
				index = idy*Nx_vec + idx;
				if (d_valid[index]){
					tmpidx = atomicAdd(&d_neighbor_cnt[id], 1);
					d_disp_x_neigh[id*Nt_median + tmpidx] = d_disp_x[index];
					d_disp_y_neigh[id*Nt_median + tmpidx] = d_disp_y[index];
				}
			}
		}
		tid += stride;
	}
}

__global__ void median_validity(bool* d_valid_median, int* d_neighbor_cnt, int minNneigh, int Nt)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	int stride = blockDim.x * gridDim.x;
	while (tid < Nt){
		d_valid_median[tid] = false;
		if (d_neighbor_cnt[tid] >= minNneigh){
			d_valid_median[tid] = true;
		}
		tid += stride;
	}
}

__global__ void replace_local_max_2d(float* d_max_local_tot, int* d_index_four_peaks, int peak_idx, int m_Nt, int Npeak_save)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	int stride = blockDim.x * gridDim.x;
	while (tid < m_Nt)
	{
		d_max_local_tot[d_index_four_peaks[tid * Npeak_save + peak_idx]] = -1000000;
		tid += stride;
	}
}

__global__ void peak_subpixel_4peak_2d(bool* valid, float* disp_x, float* disp_y, float* d_disp_x_filter, float* d_disp_y_filter, float* d_max_4peaks, int* d_index_4peaks,
	int* d_index_local_tot, float* d_cc, int ws_x, int ws_y, int Nt_ws, int idw_st, int m_Nt, int Npeak_save, float dxlim1, float dxlim2, float dylim1, float dylim2)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	int stride = blockDim.x * gridDim.x;

	while (tid < m_Nt)
	{
		int idw = tid / Npeak_save;
		int inneridx = tid - idw * Npeak_save;
		int linearidx = d_index_local_tot[d_index_4peaks[tid]];
		int index = linearidx - idw*Nt_ws;
		int idy = index / ws_x;
		int idx = index - idy*ws_x;
		idw = idw + idw_st;
		int inputidx = idw * Npeak_save + inneridx;

		float f0, f1, f2;
		if (idx == 0 || idx == (ws_x - 1) || idy == 0 || idy == (ws_y - 1) || d_max_4peaks[tid] <= 0){
			valid[inputidx] = false;
			disp_x[inputidx] = 0;
			disp_y[inputidx] = 0;
		}
		else{
			valid[inputidx] = true;
			f1 = logf(d_max_4peaks[tid]);
			f0 = logf(d_cc[linearidx - 1]);
			f2 = logf(d_cc[linearidx + 1]);
			disp_x[inputidx] = (float)idx + (f0 - f2) / (f0 + f2 - 2 * f1) / 2 - (float)ws_x / 2 + d_disp_x_filter[idw];
			f0 = logf(d_cc[linearidx - ws_x]);
			f2 = logf(d_cc[linearidx + ws_x]);
			disp_y[inputidx] = (float)idy + (f0 - f2) / (f0 + f2 - 2 * f1) / 2 - (float)ws_y / 2 + d_disp_y_filter[idw];

			if (disp_x[inputidx] < dxlim1 || disp_x[inputidx] > dxlim2 || disp_y[inputidx] < dylim1 || disp_y[inputidx] > dylim2){
				valid[inputidx] = false;
				disp_x[inputidx] = 0;
				disp_y[inputidx] = 0;
			}
		}
		tid += stride;
	}
}

__global__ void fill_blank_vecfield(float* d_disp_x_filter, float* d_disp_y_filter, bool* d_valid_filter, bool* d_fill_index, int* d_fill_cnt, int Nx_vec, int Ny_vec, int hfwx_local,
	int hfwy_local, int Nt)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	int stride = blockDim.x * gridDim.x;
	while (tid < Nt){
		int cnt = 0;
		int idy, idx, j, k, index;
		float sumx = 0;
		float sumy = 0;
		d_fill_index[tid] = false;
		if (!d_valid_filter[tid]){
			idy = tid / Nx_vec;
			idx = tid - idy*Nx_vec;
			for (j = max(0, idy - hfwy_local); j <= min(Ny_vec-1, idy + hfwy_local); j++){
				for (k = max(0, idx - hfwx_local); k <= min(Nx_vec-1, idx + hfwx_local); k++){
					index = j*Nx_vec + k;
					if (d_valid_filter[index]){
						cnt += 1;
						sumx += d_disp_x_filter[index];
						sumy += d_disp_y_filter[index];
					}
				}
			}
			if (cnt > 0){
				atomicAdd(d_fill_cnt, 1);
				d_fill_index[tid] = true;
				d_disp_x_filter[tid] = sumx / (float)cnt;
				d_disp_y_filter[tid] = sumy / (float)cnt;
			}
		}
		tid += stride;
	}
}

__global__ void fill_blank_vecfield_update(bool* d_valid_filter, bool* d_fill_index, int Nt)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	int stride = blockDim.x * gridDim.x;
	while (tid < Nt){
		if (d_fill_index[tid]){
			d_valid_filter[tid] = true;
		}
		tid += stride;
	}
}

__global__ void organize_smooth_filter_data(float* d_disp_x_filter, float* d_disp_y_filter, float* d_disp_x_neigh, float* d_disp_y_neigh, int* d_neighbor_cnt, int Nx_vec, int Ny_vec,
	int wx, int hfwx, int hfwy, int Nt_median, int Nt)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	int stride = blockDim.x * gridDim.x;
	while (tid < Nt)
	{
		int id = tid / Nt_median;
		int idy = id / Nx_vec;
		int idx = id - idy*Nx_vec;
		int tmp = tid - id*Nt_median;
		int index, tmpidx;
		int idy_win = tmp / wx;
		int idx_win = tmp - idy_win*wx;
		idx = idx - hfwx + idx_win;
		idy = idy - hfwy + idy_win;
		if (idx >= 0 && idx <= Nx_vec - 1 && idy >= 0 && idy <= Ny_vec - 1){
			index = idy*Nx_vec + idx;
			tmpidx = atomicAdd(&d_neighbor_cnt[id], 1);
			d_disp_x_neigh[id*Nt_median + tmpidx] = d_disp_x_filter[index];
			d_disp_y_neigh[id*Nt_median + tmpidx] = d_disp_y_filter[index];
		}
		tid += stride;
	}
}

__global__ void smoothing_tophat(float* d_disp_x_filter, float* d_disp_y_filter, float* d_disp_x_neigh, float* d_disp_y_neigh, int* d_neighbor_cnt, int Nt_smooth, int Nt)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	int stride = blockDim.x * gridDim.x;
	while (tid < Nt)
	{
		float sumx = 0;
		float sumy = 0;
		int Nneigh = d_neighbor_cnt[tid];
		int idx = tid*Nt_smooth;
		for (int i = 0; i < Nneigh; i++){
			sumx += d_disp_x_neigh[idx + i];
			sumy += d_disp_y_neigh[idx + i];
		}
		d_disp_x_filter[tid] = sumx / (float)Nneigh;
		d_disp_y_filter[tid] = sumy / (float)Nneigh;
		tid += stride;
	}
}

__global__ void set_float_zero_kernel(float* input, int m_Nt)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	int stride = blockDim.x * gridDim.x;

	while (tid < m_Nt)
	{
		input[tid] = 0;
		tid += stride;
	}
}

__global__ void kernel_remove_nan(float* d_disp_x, float* d_disp_y, bool* d_valid, int m_Nt)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	int stride = blockDim.x * gridDim.x;

	while (tid < m_Nt)
	{
		if (isnan(d_disp_x[tid]) || isnan(d_disp_y[tid]) || isinf(d_disp_x[tid]) || isinf(d_disp_y[tid])){
			d_disp_x[tid] = 0;
			d_disp_y[tid] = 0;
			d_valid[tid] = false;
		}
		tid += stride;
	}
}

__global__ void init_index_inter(int* d_index_inter, int m_Nt)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	int stride = blockDim.x * gridDim.x;
	while (tid < m_Nt)
	{
		d_index_inter[tid] = tid;
		tid += stride;
	}
}

__global__ void median_filter_data_prep(float* d_disp_x, float* d_disp_y, bool* d_valid, float* d_disp_x_filter, float* d_disp_y_filter, bool* d_valid_filter, int Nt, int Npeak_save)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	int stride = blockDim.x * gridDim.x;
	while (tid < Nt)
	{
		d_valid_filter[tid] = d_valid[tid * Npeak_save];
		d_disp_x_filter[tid] = d_disp_x[tid * Npeak_save];
		d_disp_y_filter[tid] = d_disp_y[tid * Npeak_save];
		tid += stride;
	}
}

__global__ void organize_data_fft2d_multi(unsigned char* img_org, cufftComplex* img_sub, int Nx_vec, int ws_x, int spacing_x, int spacing_y, int Npix_w, int border_x, int border_y, int Nx_img,
	int idx_sec_st, int m_Nt)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	int stride = blockDim.x * gridDim.x;
	while (tid < m_Nt)
	{
		int index_window = tid / Npix_w + idx_sec_st;
		int tmp2 = tid%Npix_w;
		int ix_w = index_window%Nx_vec;
		int iy_w = (index_window - ix_w) / Nx_vec;
		int ix_start_w = ix_w*spacing_x;
		int iy_start_w = iy_w*spacing_y;
		int ix = tmp2%ws_x + ix_start_w + border_x;
		int iy = tmp2 / ws_x + iy_start_w + border_y;
		int idx = iy*Nx_img + ix;
		img_sub[tid].x = (float)img_org[idx];
		img_sub[tid].y = 0;
		tid += stride;
	}
}

__global__ void multiply_conjugate(cufftComplex *data1, cufftComplex *data2, int m_Nt)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	int stride = blockDim.x * gridDim.x;

	while (tid < m_Nt)
	{
		float x1 = data1[tid].x;
		float y1 = data1[tid].y;
		float x2 = data2[tid].x;
		float y2 = data2[tid].y;
		data1[tid].x = x1*x2 + y1*y2;
		data1[tid].y = x1*y2 - x2*y1;
		tid += stride;
	}
}

__global__ void complex2amp_xcorr2d(cufftComplex* complex, float* amp, float scale, int m_Nt)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	int stride = blockDim.x * gridDim.x;
	while (tid < m_Nt)
	{
		amp[tid] = sqrtf(complex[tid].x*complex[tid].x + complex[tid].y*complex[tid].y) / scale;
		tid += stride;
	}
}

__global__ void fftshift2d_xcorr2d_row(float* unshifted, int m_Nx, int m_Ny, int Nt_half, int m_Nt)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	int stride = blockDim.x * gridDim.x;
	while (tid < m_Nt / 2)
	{
		int id_w = tid / Nt_half;
		int tmp = tid - id_w*Nt_half;
		int idy = tmp / m_Nx;
		int idx = tmp - m_Nx*idy;
		int idx1 = idx + idy*m_Nx + id_w*Nt_half * 2;
		int idx2 = idx + (idy + m_Ny / 2)*m_Nx + id_w*Nt_half * 2;
		float tmpcc = unshifted[idx1];
		unshifted[idx1] = unshifted[idx2];
		unshifted[idx2] = tmpcc;
		tid += stride;
	}
}

__global__ void fftshift2d_xcorr2d_col(float* unshifted, int m_Nx_half, int Nt_half, int m_Nt)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	int stride = blockDim.x * gridDim.x;
	while (tid < m_Nt / 2)
	{
		int id_w = tid / Nt_half;
		int tmp = tid - id_w*Nt_half;
		int idy = tmp / m_Nx_half;
		int idx = tmp - idy*m_Nx_half;
		int idx1 = idx + idy*m_Nx_half * 2 + id_w*Nt_half * 2;
		int idx2 = idx + m_Nx_half + idy*m_Nx_half * 2 + id_w*Nt_half * 2;
		float tmpcc = unshifted[idx1];
		unshifted[idx1] = unshifted[idx2];
		unshifted[idx2] = tmpcc;
		tid += stride;
	}
}

__global__ void find_local_max_2d(float* d_cc, float* d_max_local, int* d_index_local, int* d_posi_count, int Npeak, int ws_x, int ws_y, int Nt_ws, int m_Nt, int border_x, int border_y)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	int stride = blockDim.x * gridDim.x;
	while (tid < m_Nt)
	{
		int idw = tid / Nt_ws;
		int tmp = tid - idw*Nt_ws;
		int j = tmp / ws_x;
		int i = tmp - j*ws_x;
		int tmpidx;
		if (i >= border_x && i <= ws_x - 1 - border_x && j >= border_y && j <= ws_y - 1 - border_y){
			if (d_cc[tid] > 0 && d_cc[tid] >= d_cc[tid - 1] && d_cc[tid] >= d_cc[tid + 1] && d_cc[tid] >= d_cc[tid - ws_x] && d_cc[tid] >= d_cc[tid + ws_x]){
				tmpidx = atomicAdd(&d_posi_count[idw], 1);
				tmpidx = min(tmpidx, Npeak - 1);
				d_max_local[idw*Npeak + tmpidx] = d_cc[tid];
				d_index_local[idw*Npeak + tmpidx] = tid;
			}
		}
		tid += stride;
	}
}

__global__ void Kernel_Window_Deformation_2d(float* d_img_deform, unsigned char* d_img, int x_st_inten, int y_st_inten, int Nx_inten, float* d_disp_x_ex, float* d_disp_y_ex, float x_st_vec,
	float y_st_vec, float vecspacingx, float vecspacingy, int Nx_vec, int Nx, int m_Nt)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	int stride = blockDim.x * gridDim.x;
	while (tid < m_Nt)
	{
		int idy = tid / Nx;
		int idx = tid - idy*Nx;
		float a_x = ((float)idx - x_st_vec) / vecspacingx;
		float a_y = ((float)idy - y_st_vec) / vecspacingy;
		int i_x = (int)floorf(a_x);
		int i_y = (int)floorf(a_y);
		a_x = a_x - (float)i_x;
		a_y = a_y - (float)i_y;
		float ce1 = (1 - a_x)*(1 - a_y);
		float ce2 = a_x*(1 - a_y);
		float ce3 = (1 - a_x)*a_y;
		float ce4 = a_x*a_y;
		int i1 = i_y*Nx_vec + i_x;
		int i2 = i1 + 1;
		int i3 = i1 + Nx_vec;
		int i4 = i3 + 1;
		a_x = (float)idx + ce1*d_disp_x_ex[i1] + ce2*d_disp_x_ex[i2] + ce3*d_disp_x_ex[i3] + ce4*d_disp_x_ex[i4];
		a_y = (float)idy + ce1*d_disp_y_ex[i1] + ce2*d_disp_y_ex[i2] + ce3*d_disp_y_ex[i3] + ce4*d_disp_y_ex[i4];
		

		i_x = (int)floorf(a_x);
		i_y = (int)floorf(a_y);
		a_x = a_x - (float)i_x;
		a_y = a_y - (float)i_y;
		i_x = i_x - x_st_inten;
		i_y = i_y - y_st_inten;
		i1 = i_y*Nx_inten + i_x;
		i2 = i1 + 1;
		i3 = i1 + Nx_inten;
		i4 = i3 + 1;
		ce1 = (1 - a_x)*(1 - a_y);
		ce2 = a_x*(1 - a_y);
		ce3 = (1 - a_x)*a_y;
		ce4 = a_x*a_y;
		d_img_deform[tid] = ce1*(float)d_img[i1] + ce2*(float)d_img[i2] + ce3*(float)d_img[i3] + ce4*(float)d_img[i4];
		tid += stride;
	}
}

__global__ void organize_data_fft2d_multi_deformed(float* img_org, cufftComplex* img_sub, int Nx_vec, int ws_x, int spacing_x, int spacing_y, int Npix_w, int Nx_img, int idx_sec_st, int m_Nt)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	int stride = blockDim.x * gridDim.x;
	while (tid < m_Nt)
	{
		int index_window = tid / Npix_w + idx_sec_st;
		int tmp2 = tid%Npix_w;
		int ix_w = index_window%Nx_vec;
		int iy_w = (index_window - ix_w) / Nx_vec;
		int ix_start_w = ix_w*spacing_x;
		int iy_start_w = iy_w*spacing_y;
		int ix = tmp2%ws_x + ix_start_w;
		int iy = tmp2 / ws_x + iy_start_w;
		int idx = iy*Nx_img + ix;
		img_sub[tid].x = img_org[idx];
		img_sub[tid].y = 0;
		tid += stride;
	}
}

__global__ void Upsampling_vecfield_kernel_2d(float* d_disp_x_p2, float* d_disp_y_p2, float* d_disp_x_p1_ex, float* d_disp_y_p1_ex, int Nx1_vec, float x_st_vec1, float y_st_vec1, 
	float vecspacingx1, float vecspacingy1, int vecspacingx2, int vecspacingy2, int Nx2_vec, int m_Nt)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	int stride = blockDim.x * gridDim.x;
	while (tid < m_Nt)
	{
		int idy = tid / Nx2_vec;
		int idx = tid - idy*Nx2_vec;
		idx = idx*vecspacingx2 + vecspacingx2 / 2;
		idy = idy*vecspacingy2 + vecspacingy2 / 2;
		float a_x = (float)(idx - x_st_vec1) / vecspacingx1;
		float a_y = (float)(idy - y_st_vec1) / vecspacingy1;
		int i_x = (int)floorf(a_x);
		int i_y = (int)floorf(a_y);
		a_x = a_x - (float)i_x;
		a_y = a_y - (float)i_y;
		float ce1 = (1 - a_x)*(1 - a_y);
		float ce2 = a_x*(1 - a_y);
		float ce3 = (1 - a_x)*a_y;
		float ce4 = a_x*a_y;
		int i1 = i_y*Nx1_vec + i_x;
		int i2 = i1 + 1;
		int i3 = i1 + Nx1_vec;
		int i4 = i3 + 1;
		d_disp_x_p2[tid] = ce1*d_disp_x_p1_ex[i1] + ce2*d_disp_x_p1_ex[i2] + ce3*d_disp_x_p1_ex[i3] + ce4*d_disp_x_p1_ex[i4];
		d_disp_y_p2[tid] = ce1*d_disp_y_p1_ex[i1] + ce2*d_disp_y_p1_ex[i2] + ce3*d_disp_y_p1_ex[i3] + ce4*d_disp_y_p1_ex[i4];
		tid += stride;
	}
}

void xcorr2d_multi::setupgridblockdim(int dim1_grid, int dim2_grid, int dim3_grid, int dim1_block, int dim2_block, int dim3_block)
{
	griddim.x = dim1_grid;
	griddim.y = dim2_grid;
	griddim.z = dim3_grid;
	blockdim.x = dim1_block;
	blockdim.y = dim2_block;
	blockdim.z = dim3_block;
}

bool xcorr2d_multi::init(int m_Nx, int m_Ny, int m_Nsec_cc, int m_ws_x1, int m_ws_y1, int m_spacing_x1, int m_spacing_y1, int m_Npass1, int m_ws_x2, int m_ws_y2, int m_spacing_x2, int m_spacing_y2,
	int m_Npass2, int m_wx1_median, int m_wy1_median, int m_wx2_median, int m_wy2_median, int m_wx1_smooth, int m_wy1_smooth, int m_wx2_smooth, int m_wy2_smooth, int m_Npeak_save,
	float max_dispx_percent1, float max_dispy_percent1, float max_dispx_percent2, float max_dispy_percent2, float m_dispx_lim1, float m_dispx_lim2, float m_dispy_lim1, float m_dispy_lim2)
{
	int tmp1, tmp2, i, j, k;

	Nx = m_Nx;
	Ny = m_Ny;
	Nt = Nx*Ny;
	ws_x1 = m_ws_x1;
	ws_y1 = m_ws_y1;
	spacingx1 = m_spacing_x1;
	spacingy1 = m_spacing_y1;
	Npass1 = m_Npass1;
	ws_x2 = m_ws_x2;
	ws_y2 = m_ws_y2;
	spacingx2 = m_spacing_x2;
	spacingy2 = m_spacing_y2;
	Npass2 = m_Npass2;
	Nsec1_cc = m_Nsec_cc;
	Nsec2_cc = m_Nsec_cc;
	Npeak_save = m_Npeak_save;
	dispx_lim1 = m_dispx_lim1;
	dispx_lim2 = m_dispx_lim2;
	dispy_lim1 = m_dispy_lim1;
	dispy_lim2 = m_dispy_lim2;
	wx1_median = m_wx1_median;
	wy1_median = m_wy1_median;
	wx2_median = m_wx2_median;
	wy2_median = m_wy2_median;
	wx1_smooth = m_wx1_smooth;
	wy1_smooth = m_wy1_smooth;
	wx2_smooth = m_wx2_smooth;
	wy2_smooth = m_wy2_smooth;

	border_x_p1 = max(1, ws_x1 / 2 - (int)roundf((float)ws_x1*max_dispx_percent1));
	border_y_p1 = max(1, ws_y1 / 2 - (int)roundf((float)ws_y1*max_dispy_percent1));
	border_x_p2 = max(1, ws_x2 / 2 - (int)roundf((float)ws_x2*max_dispx_percent2));
	border_y_p2 = max(1, ws_y2 / 2 - (int)roundf((float)ws_y2*max_dispy_percent2));

	Nt1_median = wx1_median*wy1_median;
	Nt2_median = wx2_median*wy2_median;
	Nt1_smooth = wx1_smooth*wy1_smooth;
    Nt2_smooth = wx2_smooth*wy2_smooth;
	
	hfws_x1 = ws_x1 / 2;
	hfws_y1 = ws_y1 / 2;
	hfws_x2 = ws_x2 / 2;
	hfws_y2 = ws_y2 / 2;
	Nt_ws1 = ws_x1*ws_y1;
	Nt_ws2 = ws_x2*ws_y2;
	
	Npeak1 = ws_x1*ws_y1 / 4 / 4;
	Npeak2 = ws_x2*ws_y2 / 4 / 4;

	tmp1 = (Nx - spacingx1 / 2 - 1) / spacingx1;
	tmp2 = Nx - (spacingx1 / 2 + 1 + tmp1*spacingx1);
	while (tmp2 < (spacingx1 / 2 - 1)){
		tmp1 = tmp1 - 1;
		tmp2 = tmp2 + spacingx1;
	}
	Nx1_vec = tmp1 + 1;

	tmp1 = (Ny - spacingy1 / 2 - 1) / spacingy1;
	tmp2 = Ny - (spacingy1 / 2 + 1 + tmp1*spacingy1);
	while (tmp2 < (spacingy1 / 2 - 1)){
		tmp1 = tmp1 - 1;
		tmp2 = tmp2 + spacingy1;
	}
	Ny1_vec = tmp1 + 1;
	Nt1_vec = Nx1_vec*Ny1_vec;

	tmp1 = (Nx - spacingx2 / 2 - 1) / spacingx2;
	tmp2 = Nx - (spacingx2 / 2 + 1 + tmp1*spacingx2);
	while (tmp2 < (spacingx2 / 2 - 1)){
		tmp1 = tmp1 - 1;
		tmp2 = tmp2 + spacingx2;
	}
	Nx2_vec = tmp1 + 1;

	tmp1 = (Ny - spacingy2 / 2 - 1) / spacingy2;
	tmp2 = Ny - (spacingy2 / 2 + 1 + tmp1*spacingy2);
	while (tmp2 < (spacingy2 / 2 - 1)){
		tmp1 = tmp1 - 1;
		tmp2 = tmp2 + spacingy2;
	}
	Ny2_vec = tmp1 + 1;
    Nt2_vec = Nx2_vec*Ny2_vec;

	if (Nt1_vec%Nsec1_cc == 0){
		Neach1 = Nt1_vec / Nsec1_cc;
		Nlast1 = Neach1;
	}
	else{
		Neach1 = Nt1_vec / Nsec1_cc;
		Nlast1 = Nt1_vec - (Nsec1_cc - 1)*Neach1;
		while (Nlast1 > Neach1){
			Nlast1 -= Neach1;
			Nsec1_cc += 1;
		}
	}

	if (Nt2_vec%Nsec2_cc == 0){
		Neach2 = Nt2_vec / Nsec2_cc;
		Nlast2 = Neach2;
	}
	else{
		Neach2 = Nt2_vec / Nsec2_cc;
		Nlast2 = Nt2_vec - (Nsec2_cc - 1)*Neach2;
		while (Nlast2 > Neach2){
			Nlast2 -= Neach2;
			Nsec2_cc += 1;
		}
	}

	if (spacingx1 / 2 < hfws_x1){
		Nx1_left = hfws_x1 - spacingx1 / 2;
	}
	else{
		Nx1_left = 0;
	}
	if (spacingy1 / 2 < hfws_y1){
		Ny1_left = hfws_y1 - spacingy1 / 2;
	}
	else{
		Ny1_left = 0;
	}

	if ((Nx - spacingx1 / 2 - 1 - (Nx1_vec - 1)*spacingx1) < (hfws_x1 - 1)){
		Nx1_right = spacingx1 / 2 + 1 + (Nx1_vec - 1)*spacingx1 + hfws_x1 - 1 - Nx;
	}
	else{
		Nx1_right = 0;
	}
	if ((Ny - spacingy1 / 2 - 1 - (Ny1_vec - 1)*spacingy1) < (hfws_y1 - 1)){
		Ny1_right = spacingy1 / 2 + 1 + (Ny1_vec - 1)*spacingy1 + hfws_y1 - 1 - Ny;
	}
	else{
		Ny1_right = 0;
	}

	if (spacingx2 / 2 < hfws_x2){
		Nx2_left = hfws_x2 - spacingx2 / 2;
	}
	else{
		Nx2_left = 0;
	}
	if (spacingy2 / 2 < hfws_y2){
		Ny2_left = hfws_y2 - spacingy2 / 2;
	}
	else{
		Ny2_left = 0;
	}
	if ((Nx - spacingx2 / 2 - 1 - (Nx2_vec - 1)*spacingx2) < (hfws_x2 - 1)){
		Nx2_right = spacingx2 / 2 + 1 + (Nx2_vec - 1)*spacingx2 + hfws_x2 - 1 - Nx;
	}
	else{
		Nx2_right = 0;
	}
	if ((Ny - spacingy2 / 2 - 1 - (Ny2_vec - 1)*spacingy2) < (hfws_y2 - 1)){
		Ny2_right = spacingy2 / 2 + 1 + (Ny2_vec - 1)*spacingy2 + hfws_y2 - 1 - Ny;
	}
	else{
		Ny2_right = 0;
	}

	Nx1_aug = Nx + Nx1_left + Nx1_right;
	Ny1_aug = Ny + Ny1_left + Ny1_right;
	Nt1_aug = Nx1_aug*Ny1_aug;
	Nx1_aug_ex = Nx1_aug + ws_x1;
	Ny1_aug_ex = Ny1_aug + ws_y1;
	Nt1_aug_ex = Nx1_aug_ex*Ny1_aug_ex;

	Nx2_aug = Nx + Nx2_left + Nx2_right;
	Ny2_aug = Ny + Ny2_left + Ny2_right;
	Nt2_aug = Nx2_aug*Ny2_aug;
	Nx2_aug_ex = Nx2_aug + ws_x2;
	Ny2_aug_ex = Ny2_aug + ws_y2;
	Nt2_aug_ex = Nx2_aug_ex*Ny2_aug_ex;

	if (ws_x1 < 4 || ws_y1 <4 ){
		cout << "The interrogation window size (1st grid) is too small!" << endl;
		return false;
	}
	if (ws_x1 > 1024 || ws_y1 >1024 ){
		cout << "The interrogation window size (1st grid) is too big!" << endl;
		return false;
	}
	if (ws_x2 < 4 || ws_y2 <4 ){
		cout << "The interrogation window size (2nd grid) is too small!" << endl;
		return false;
	}
	if (ws_x2 > 1024 || ws_y2 >1024 ){
		cout << "The interrogation window size (2nd grid) is too big!" << endl;
		return false;
	}

	if (ws_x1 % 4 != 0 || ws_x2 % 4 != 0){
		cout << "The window x-dimension is not divisible by 4!" << endl;
		return false;
	}
	if (ws_y1 % 4 != 0 || ws_y2 % 4 != 0){
		cout << "The window y-dimension is not divisible by 4!" << endl;
		return false;
	}

	int fftdim1[2] = { ws_y1, ws_x1 };
	int fftdim2[2] = { ws_y2, ws_x2 };
	if (cufftPlanMany(&fftplan1, 2, fftdim1, NULL, 0, 0, NULL, 0, 0, CUFFT_C2C, Neach1) != CUFFT_SUCCESS){
		cout << "Error in creating CUDA FFT plan!" << endl;
		return false;
	}
	if (cufftPlanMany(&fftplan2, 2, fftdim2, NULL, 0, 0, NULL, 0, 0, CUFFT_C2C, Neach2) != CUFFT_SUCCESS){
		cout << "Error in creating CUDA FFT plan!" << endl;
		return false;
	}

	cudaHostAlloc((void**)&h_img1_org, sizeof(unsigned char)*Nt, cudaHostAllocDefault);
	cudaHostAlloc((void**)&h_img2_org, sizeof(unsigned char)*Nt, cudaHostAllocDefault);
	if (cudaMalloc((void**)&d_img1_p1_aug_ex, sizeof(unsigned char)*Nt1_aug_ex) != cudaSuccess){
		cout << "cudaMalloc ERROR!" << endl;
		return false;
	}
	if (cudaMalloc((void**)&d_img2_p1_aug_ex, sizeof(unsigned char)*Nt1_aug_ex) != cudaSuccess){
		cout << "cudaMalloc ERROR!" << endl;
		return false;
	}
	if (cudaMalloc((void**)&d_img1_p2_aug_ex, sizeof(unsigned char)*Nt2_aug_ex) != cudaSuccess){
		cout << "cudaMalloc ERROR!" << endl;
		return false;
	}
	if (cudaMalloc((void**)&d_img2_p2_aug_ex, sizeof(unsigned char)*Nt2_aug_ex) != cudaSuccess){
		cout << "cudaMalloc ERROR!" << endl;
		return false;
	}
	cudaMemset(d_img1_p1_aug_ex, 0, sizeof(unsigned char)*Nt1_aug_ex);
	cudaMemset(d_img2_p1_aug_ex, 0, sizeof(unsigned char)*Nt1_aug_ex);
	cudaMemset(d_img1_p2_aug_ex, 0, sizeof(unsigned char)*Nt2_aug_ex);
	cudaMemset(d_img2_p2_aug_ex, 0, sizeof(unsigned char)*Nt2_aug_ex);
	
	tmp1 = max(Neach1*Nt_ws1, Neach2*Nt_ws2);
	if (cudaMalloc((void**)&d_xcorr1, sizeof(cufftComplex)*tmp1) != cudaSuccess){
		cout << "cudaMalloc ERROR!" << endl;
		return false;
	}
	if (cudaMalloc((void**)&d_xcorr2, sizeof(cufftComplex)*tmp1) != cudaSuccess){
		cout << "cudaMalloc ERROR!" << endl;
		return false;
	}
	if (cudaMalloc((void**)&d_cc, sizeof(float)*tmp1) != cudaSuccess){
		cout << "cudaMalloc ERROR!" << endl;
		return false;
	}
	if (cudaMalloc((void**)&d_img_deform_p1, sizeof(float)*Nt1_aug) != cudaSuccess){
		cout << "cudaMalloc ERROR!" << endl;
		return false;
	}
	if (cudaMalloc((void**)&d_img_deform_p2, sizeof(float)*Nt2_aug) != cudaSuccess){
		cout << "cudaMalloc ERROR!" << endl;
		return false;
	}
	if (cudaMalloc((void**)&d_posi_count_p1, sizeof(int)*Neach1) != cudaSuccess){
		cout << "cudaMalloc ERROR!" << endl;
		return false;
	}
	if (cudaMalloc((void**)&d_posi_count_p2, sizeof(int)*Neach2) != cudaSuccess){
		cout << "cudaMalloc ERROR!" << endl;
		return false;
	}
	if (cudaMalloc((void**)&d_index_local_tot1, sizeof(int)*Neach1 * Npeak1) != cudaSuccess){
		cout << "cudaMalloc ERROR!" << endl;
		return false;
	}
	if (cudaMalloc((void**)&d_index_local_tot2, sizeof(int)*Neach2 * Npeak2) != cudaSuccess){
		cout << "cudaMalloc ERROR!" << endl;
		return false;
	}
	if (cudaMalloc((void**)&d_max_local_tot1, sizeof(float)*Neach1 * Npeak1) != cudaSuccess){
		cout << "cudaMalloc ERROR!" << endl;
		return false;
	}
	if (cudaMalloc((void**)&d_max_local_tot2, sizeof(float)*Neach2 * Npeak2) != cudaSuccess){
		cout << "cudaMalloc ERROR!" << endl;
		return false;
	}
	
	gridN1_peak_p1 = ws_y1 / 4;	
	gridN2_peak_p1 = 1;
	blockN1_peak_p1 = (int)powf(2, ceilf(log2f((float)(ws_x1 / 4))));
	blockN2_peak_p1 = (int)powf(2, ceilf(log2f((float)(ws_y1 / 4))));

	gridN1_peak_p2 = ws_y2 / 4;
	gridN2_peak_p2 = 1;
	blockN1_peak_p2 = (int)powf(2, ceilf(log2f((float)(ws_x2 / 4))));
	blockN2_peak_p2 = (int)powf(2, ceilf(log2f((float)(ws_y2 / 4))));

	if (cudaMalloc((void**)&d_index_inter_p1, sizeof(int)*Npeak1*Neach1) != cudaSuccess){
		cout << "cudaMalloc ERROR!" << endl;
		return false;
	}
	if (cudaMalloc((void**)&d_index_inter_p2, sizeof(int)*Npeak2*Neach2) != cudaSuccess){
		cout << "cudaMalloc ERROR!" << endl;
		return false;
	}
	if (cudaMalloc((void**)&d_max_inter1_p1, sizeof(float)*gridN1_peak_p1*Neach1) != cudaSuccess){
		cout << "cudaMalloc ERROR!" << endl;
		return false;
	}
	if (cudaMalloc((void**)&d_max_inter1_p2, sizeof(float)*gridN1_peak_p2*Neach2) != cudaSuccess){
		cout << "cudaMalloc ERROR!" << endl;
		return false;
	}
	if (cudaMalloc((void**)&d_index_inter1_p1, sizeof(int)*gridN1_peak_p1*Neach1) != cudaSuccess){
		cout << "cudaMalloc ERROR!" << endl;
		return false;
	}
	if (cudaMalloc((void**)&d_index_inter1_p2, sizeof(int)*gridN1_peak_p2*Neach2) != cudaSuccess){
		cout << "cudaMalloc ERROR!" << endl;
		return false;
	}
	init_index_inter << <griddim, blockdim >> >(d_index_inter_p1, Npeak1*Neach1);
	if (cudaPeekAtLastError() != cudaSuccess){
		printf("GPU kernel error at init_index_inter: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
		return false;
	}
	init_index_inter << <griddim, blockdim >> >(d_index_inter_p2, Npeak2*Neach2);
	if (cudaPeekAtLastError() != cudaSuccess){
		printf("GPU kernel error at init_index_inter: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
		return false;
	}

	if (cudaMalloc((void**)&d_index_four_peaks_p1, sizeof(int)*Neach1 * Npeak_save) != cudaSuccess){
		cout << "cudaMalloc ERROR!" << endl;
		return false;
	}
	if (cudaMalloc((void**)&d_index_four_peaks_p2, sizeof(int)*Neach2 * Npeak_save) != cudaSuccess){
		cout << "cudaMalloc ERROR!" << endl;
		return false;
	}
	if (cudaMalloc((void**)&d_max_four_peaks_p1, sizeof(float)*Neach1 * Npeak_save) != cudaSuccess){
		cout << "cudaMalloc ERROR!" << endl;
		return false;
	}
	if (cudaMalloc((void**)&d_max_four_peaks_p2, sizeof(float)*Neach2 * Npeak_save) != cudaSuccess){
		cout << "cudaMalloc ERROR!" << endl;
		return false;
	}
	if (cudaMalloc((void**)&d_valid_p1, sizeof(bool)*Nt1_vec * Npeak_save) != cudaSuccess){
		cout << "cudaMalloc ERROR!" << endl;
		return false;
	}
	if (cudaMalloc((void**)&d_disp_x_p1, sizeof(float)*Nt1_vec * Npeak_save) != cudaSuccess){
		cout << "cudaMalloc ERROR!" << endl;
		return false;
	}
	if (cudaMalloc((void**)&d_disp_y_p1, sizeof(float)*Nt1_vec * Npeak_save) != cudaSuccess){
		cout << "cudaMalloc ERROR!" << endl;
		return false;
	}
	if (cudaMalloc((void**)&d_valid_p2, sizeof(bool)*Nt2_vec * Npeak_save) != cudaSuccess){
		cout << "cudaMalloc ERROR!" << endl;
		return false;
	}
	if (cudaMalloc((void**)&d_disp_x_p2, sizeof(float)*Nt2_vec * Npeak_save) != cudaSuccess){
		cout << "cudaMalloc ERROR!" << endl;
		return false;
	}
	if (cudaMalloc((void**)&d_disp_y_p2, sizeof(float)*Nt2_vec * Npeak_save) != cudaSuccess){
		cout << "cudaMalloc ERROR!" << endl;
		return false;
	}
	if (cudaMalloc((void**)&d_disp_x_filter_p1, sizeof(float)*Nt1_vec) != cudaSuccess){
		cout << "cudaMalloc ERROR!" << endl;
		return false;
	}
	if (cudaMalloc((void**)&d_disp_y_filter_p1, sizeof(float)*Nt1_vec) != cudaSuccess){
		cout << "cudaMalloc ERROR!" << endl;
		return false;
	}
	if (cudaMalloc((void**)&d_disp_x_filter_p2, sizeof(float)*Nt2_vec) != cudaSuccess){
		cout << "cudaMalloc ERROR!" << endl;
		return false;
	}
	if (cudaMalloc((void**)&d_disp_y_filter_p2, sizeof(float)*Nt2_vec) != cudaSuccess){
		cout << "cudaMalloc ERROR!" << endl;
		return false;
	}
	if (cudaMalloc((void**)&d_valid_filter_p1, sizeof(bool)*Nt1_vec) != cudaSuccess){
		cout << "cudaMalloc ERROR!" << endl;
		return false;
	}
	if (cudaMalloc((void**)&d_valid_filter_p2, sizeof(bool)*Nt2_vec) != cudaSuccess){
		cout << "cudaMalloc ERROR!" << endl;
		return false;
	}
	if (cudaMalloc((void**)&d_neighbor_cnt_p1, sizeof(int)*Nt1_vec) != cudaSuccess){
		cout << "cudaMalloc ERROR!" << endl;
		return false;
	}
	if (cudaMalloc((void**)&d_neighbor_cnt_p2, sizeof(int)*Nt2_vec) != cudaSuccess){
		cout << "cudaMalloc ERROR!" << endl;
		return false;
	}
	if (cudaMalloc((void**)&d_disp_x_neigh_p1, sizeof(float)*Nt1_median*Nt1_vec) != cudaSuccess){
		cout << "cudaMalloc ERROR!" << endl;
		return false;
	}
	if (cudaMalloc((void**)&d_disp_y_neigh_p1, sizeof(float)*Nt1_median*Nt1_vec) != cudaSuccess){
		cout << "cudaMalloc ERROR!" << endl;
		return false;
	}
	if (cudaMalloc((void**)&d_disp_x_neigh_p2, sizeof(float)*Nt2_median*Nt2_vec) != cudaSuccess){
		cout << "cudaMalloc ERROR!" << endl;
		return false;
	}
	if (cudaMalloc((void**)&d_disp_y_neigh_p2, sizeof(float)*Nt2_median*Nt2_vec) != cudaSuccess){
		cout << "cudaMalloc ERROR!" << endl;
		return false;
	}
	if (cudaMalloc((void**)&d_valid_median_p1, sizeof(bool)*Nt1_vec) != cudaSuccess){
		cout << "cudaMalloc ERROR!" << endl;
		return false;
	}
	if (cudaMalloc((void**)&d_valid_median_p2, sizeof(bool)*Nt2_vec) != cudaSuccess){
		cout << "cudaMalloc ERROR!" << endl;
		return false;
	}
	if (cudaMalloc((void**)&d_median_x_p1, sizeof(float)*Nt1_vec) != cudaSuccess){
		cout << "cudaMalloc ERROR!" << endl;
		return false;
	}
	if (cudaMalloc((void**)&d_median_y_p1, sizeof(float)*Nt1_vec) != cudaSuccess){
		cout << "cudaMalloc ERROR!" << endl;
		return false;
	}
	if (cudaMalloc((void**)&d_median_x_p2, sizeof(float)*Nt2_vec) != cudaSuccess){
		cout << "cudaMalloc ERROR!" << endl;
		return false;
	}
	if (cudaMalloc((void**)&d_median_y_p2, sizeof(float)*Nt2_vec) != cudaSuccess){
		cout << "cudaMalloc ERROR!" << endl;
		return false;
	}
	if (cudaMalloc((void**)&d_devi_x_neigh_p1, sizeof(float)*Nt1_median*Nt1_vec) != cudaSuccess){
		cout << "cudaMalloc ERROR!" << endl;
		return false;
	}
	if (cudaMalloc((void**)&d_devi_y_neigh_p1, sizeof(float)*Nt1_median*Nt1_vec) != cudaSuccess){
		cout << "cudaMalloc ERROR!" << endl;
		return false;
	}
	if (cudaMalloc((void**)&d_devi_x_neigh_p2, sizeof(float)*Nt2_median*Nt2_vec) != cudaSuccess){
		cout << "cudaMalloc ERROR!" << endl;
		return false;
	}
	if (cudaMalloc((void**)&d_devi_y_neigh_p2, sizeof(float)*Nt2_median*Nt2_vec) != cudaSuccess){
		cout << "cudaMalloc ERROR!" << endl;
		return false;
	}
	if (cudaMalloc((void**)&d_devimedian_x_p1, sizeof(float)*Nt1_vec) != cudaSuccess){
		cout << "cudaMalloc ERROR!" << endl;
		return false;
	}
	if (cudaMalloc((void**)&d_devimedian_y_p1, sizeof(float)*Nt1_vec) != cudaSuccess){
		cout << "cudaMalloc ERROR!" << endl;
		return false;
	}
	if (cudaMalloc((void**)&d_devimedian_x_p2, sizeof(float)*Nt2_vec) != cudaSuccess){
		cout << "cudaMalloc ERROR!" << endl;
		return false;
	}
	if (cudaMalloc((void**)&d_devimedian_y_p2, sizeof(float)*Nt2_vec) != cudaSuccess){
		cout << "cudaMalloc ERROR!" << endl;
		return false;
	}
	if (cudaMalloc((void**)&d_fill_index_p1, sizeof(bool)*Nt1_vec) != cudaSuccess){
		cout << "cudaMalloc ERROR!" << endl;
		return false;
	}
	if (cudaMalloc((void**)&d_fill_index_p2, sizeof(bool)*Nt2_vec) != cudaSuccess){
		cout << "cudaMalloc ERROR!" << endl;
		return false;
	}
	if (cudaMalloc((void**)&d_fill_cnt, sizeof(int)) != cudaSuccess){
		cout << "cudaMalloc ERROR!" << endl;
		return false;
	}
	cudaHostAlloc((void**)&h_fill_cnt, sizeof(int), cudaHostAllocDefault);
	if (cudaMalloc((void**)&d_disp_x_neigh_smooth_p1, sizeof(float)*Nt1_smooth*Nt1_vec) != cudaSuccess){
		cout << "cudaMalloc ERROR!" << endl;
		return false;
	}
	if (cudaMalloc((void**)&d_disp_y_neigh_smooth_p1, sizeof(float)*Nt1_smooth*Nt1_vec) != cudaSuccess){
		cout << "cudaMalloc ERROR!" << endl;
		return false;
	}
	if (cudaMalloc((void**)&d_disp_x_neigh_smooth_p2, sizeof(float)*Nt2_smooth*Nt2_vec) != cudaSuccess){
		cout << "cudaMalloc ERROR!" << endl;
		return false;
	}
	if (cudaMalloc((void**)&d_disp_y_neigh_smooth_p2, sizeof(float)*Nt2_smooth*Nt2_vec) != cudaSuccess){
		cout << "cudaMalloc ERROR!" << endl;
		return false;
	}
	if (cudaMalloc((void**)&d_disp_x_filter_p1_ex, sizeof(float)*(Nx1_vec + 2 * vec_ex_border_p1)*(Ny1_vec + 2 * vec_ex_border_p1)) != cudaSuccess){
		cout << "cudaMalloc ERROR!" << endl;
		return false;
	}
	if (cudaMalloc((void**)&d_disp_y_filter_p1_ex, sizeof(float)*(Nx1_vec + 2 * vec_ex_border_p1)*(Ny1_vec + 2 * vec_ex_border_p1)) != cudaSuccess){
		cout << "cudaMalloc ERROR!" << endl;
		return false;
	}
	if (cudaMalloc((void**)&d_disp_x_filter_p2_ex, sizeof(float)*(Nx2_vec + 2 * vec_ex_border_p2)*(Ny2_vec + 2 * vec_ex_border_p2)) != cudaSuccess){
		cout << "cudaMalloc ERROR!" << endl;
		return false;
	}
	if (cudaMalloc((void**)&d_disp_y_filter_p2_ex, sizeof(float)*(Nx2_vec + 2 * vec_ex_border_p2)*(Ny2_vec + 2 * vec_ex_border_p2)) != cudaSuccess){
		cout << "cudaMalloc ERROR!" << endl;
		return false;
	}
	
	cudaHostAlloc((void**)&h_gridx1, sizeof(float)*Nt1_vec, cudaHostAllocDefault);
	cudaHostAlloc((void**)&h_gridy1, sizeof(float)*Nt1_vec, cudaHostAllocDefault);
	for (j = 0; j < Ny1_vec; j++){
		tmp2 = spacingy1 / 2 + j*spacingy1;
		for (k = 0; k < Nx1_vec; k++){
			h_gridy1[j*Nx1_vec + k] = (float)tmp2;
			h_gridx1[j*Nx1_vec + k] = (float)(spacingx1 / 2 + k*spacingx1);
		}
	}
	cudaHostAlloc((void**)&h_gridx2, sizeof(float)*Nt2_vec, cudaHostAllocDefault);
	cudaHostAlloc((void**)&h_gridy2, sizeof(float)*Nt2_vec, cudaHostAllocDefault);
	for (j = 0; j < Ny2_vec; j++){
		tmp2 = spacingy2 / 2 + j*spacingy2;
		for (k = 0; k < Nx2_vec; k++){
			h_gridy2[j*Nx2_vec + k] = (float)tmp2;
			h_gridx2[j*Nx2_vec + k] = (float)(spacingx2 / 2 + k*spacingx2);
		}
	}
	cudaHostAlloc((void**)&h_valid_p1, sizeof(bool)*Nt1_vec * Npeak_save, cudaHostAllocDefault);
	cudaHostAlloc((void**)&h_disp_x_p1, sizeof(float)*Nt1_vec * Npeak_save, cudaHostAllocDefault);
	cudaHostAlloc((void**)&h_disp_y_p1, sizeof(float)*Nt1_vec * Npeak_save, cudaHostAllocDefault);
	cudaHostAlloc((void**)&h_valid_p2, sizeof(bool)*Nt2_vec * Npeak_save, cudaHostAllocDefault);
	cudaHostAlloc((void**)&h_disp_x_p2, sizeof(float)*Nt2_vec * Npeak_save, cudaHostAllocDefault);
	cudaHostAlloc((void**)&h_disp_y_p2, sizeof(float)*Nt2_vec * Npeak_save, cudaHostAllocDefault);
	

	cudaHostAlloc((void**)&h_disp_x_filter_p1, sizeof(float)*Nt1_vec, cudaHostAllocDefault);
	cudaHostAlloc((void**)&h_disp_y_filter_p1, sizeof(float)*Nt1_vec, cudaHostAllocDefault);
	cudaHostAlloc((void**)&h_disp_x_filter_p2, sizeof(float)*Nt2_vec, cudaHostAllocDefault);
	cudaHostAlloc((void**)&h_disp_y_filter_p2, sizeof(float)*Nt2_vec, cudaHostAllocDefault);

	return true;
}

void xcorr2d_multi::prep_img_data()
{
	int j, tmp1, tmp2;
	for (j = 0; j < Ny; j++){
		tmp1 = (j + Ny1_left + hfws_y1)*Nx1_aug_ex;
		tmp2 = (j + Ny2_left + hfws_y2)*Nx2_aug_ex;
		cudaMemcpy(d_img1_p1_aug_ex + tmp1 + Nx1_left + hfws_x1, h_img1_org + j*Nx, sizeof(unsigned char)*Nx, cudaMemcpyHostToDevice);
		cudaMemcpy(d_img2_p1_aug_ex + tmp1 + Nx1_left + hfws_x1, h_img2_org + j*Nx, sizeof(unsigned char)*Nx, cudaMemcpyHostToDevice);
		cudaMemcpy(d_img1_p2_aug_ex + tmp2 + Nx2_left + hfws_x2, h_img1_org + j*Nx, sizeof(unsigned char)*Nx, cudaMemcpyHostToDevice);
		cudaMemcpy(d_img2_p2_aug_ex + tmp2 + Nx2_left + hfws_x2, h_img2_org + j*Nx, sizeof(unsigned char)*Nx, cudaMemcpyHostToDevice);
	}
}

bool xcorr2d_multi::Calc_cc_map_p1(int idx_pass)
{
	int i, j, k, tmpN;
	if (idx_pass == 0){
		set_float_zero_kernel << <griddim, blockdim >> >(d_disp_x_filter_p1, Nt1_vec);
		if (cudaPeekAtLastError() != cudaSuccess){
			printf("GPU kernel error at set_float_zero_kernel: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
			return false;
		}
		set_float_zero_kernel << <griddim, blockdim >> >(d_disp_y_filter_p1, Nt1_vec);
		if (cudaPeekAtLastError() != cudaSuccess){
			printf("GPU kernel error at set_float_zero_kernel: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
			return false;
		}
	}
	else{
		Kernel_Window_Deformation_2d << <griddim, blockdim >> >(d_img_deform_p1, d_img2_p1_aug_ex, -hfws_x1, -hfws_y1, Nx1_aug_ex, d_disp_x_filter_p1_ex, d_disp_y_filter_p1_ex,
			(float)(hfws_x1 - spacingx1*vec_ex_border_p1), (float)(hfws_y1 - spacingy1*vec_ex_border_p1), (float)spacingx1, (float)spacingy1, Nx1_vec + 2 * vec_ex_border_p1, Nx1_aug, Nt1_aug);
		if (cudaPeekAtLastError() != cudaSuccess){
			printf("GPU kernel error at Kernel_Window_Deformation_2d: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
			return false;
		}
	}

	for (i = 0; i < Nsec1_cc; i++){
		if (i == Nsec1_cc - 1){
			tmpN = Nlast1;
		}
		else{
			tmpN = Neach1;
		}
		organize_data_fft2d_multi << <griddim, blockdim >> >(d_img1_p1_aug_ex, d_xcorr1, Nx1_vec, ws_x1, spacingx1, spacingy1, Nt_ws1, hfws_x1, hfws_y1, Nx1_aug_ex, i*Neach1, tmpN*Nt_ws1);
		if (cudaPeekAtLastError() != cudaSuccess){
			printf("GPU kernel error at organize_data_fft2d_multi: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
			return false;
		}
		if (idx_pass == 0){
			organize_data_fft2d_multi << <griddim, blockdim >> >(d_img2_p1_aug_ex, d_xcorr2, Nx1_vec, ws_x1, spacingx1, spacingy1, Nt_ws1, hfws_x1, hfws_y1, Nx1_aug_ex, i*Neach1, tmpN*Nt_ws1);
			if (cudaPeekAtLastError() != cudaSuccess){
				printf("GPU kernel error at organize_data_fft2d_multi: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
				return false;
			}
		}
		else{
			organize_data_fft2d_multi_deformed << <griddim, blockdim >> >(d_img_deform_p1, d_xcorr2, Nx1_vec, ws_x1, spacingx1, spacingy1, Nt_ws1, Nx1_aug, i*Neach1, tmpN*Nt_ws1);
			if (cudaPeekAtLastError() != cudaSuccess){
				printf("GPU kernel error at organize_data_fft2d_multi_deformed: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
				return false;
			}
		}		
		if (cufftExecC2C(fftplan1, d_xcorr1, d_xcorr1, CUFFT_FORWARD) != CUFFT_SUCCESS){
			cout << "Error occurs during execution of cufftExecC2C!" << endl;
			return false;
		}
		if (cufftExecC2C(fftplan1, d_xcorr2, d_xcorr2, CUFFT_FORWARD) != CUFFT_SUCCESS){
			cout << "Error occurs during execution of cufftExecC2C!" << endl;
			return false;
		}
		multiply_conjugate << <griddim, blockdim >> >(d_xcorr1, d_xcorr2, tmpN*Nt_ws1);
		if (cudaPeekAtLastError() != cudaSuccess){
			printf("GPU kernel error at multiply_conjugate: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
			return false;
		}
		if (cufftExecC2C(fftplan1, d_xcorr1, d_xcorr1, CUFFT_INVERSE) != CUFFT_SUCCESS){
			cout << "Error occurs during execution of cufftExecC2C!" << endl;
			return false;
		}
		complex2amp_xcorr2d << <griddim, blockdim >> >(d_xcorr1, d_cc, (float)Nt_ws1, tmpN*Nt_ws1);
		if (cudaPeekAtLastError() != cudaSuccess){
			printf("GPU kernel error at complex2amp_xcorr2d: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
			return false;
		}
		fftshift2d_xcorr2d_row << <griddim, blockdim >> >(d_cc, ws_x1, ws_y1, Nt_ws1 / 2, Nt_ws1*tmpN);
		if (cudaPeekAtLastError() != cudaSuccess){
			printf("GPU kernel error at fftshift2d_xcorr2d_row: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
			return false;
		}
		fftshift2d_xcorr2d_col << <griddim, blockdim >> >(d_cc, ws_x1 / 2, Nt_ws1 / 2, Nt_ws1*tmpN);
		if (cudaPeekAtLastError() != cudaSuccess){
			printf("GPU kernel error at fftshift2d_xcorr2d_col: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
			return false;
		}
		cudaMemset(d_posi_count_p1, 0, sizeof(int)*Neach1);
		cudaMemset(d_index_local_tot1, 0, sizeof(int)*Neach1*Npeak1);
		set_float_zero_kernel << <griddim, blockdim >> >(d_max_local_tot1, Neach1*Npeak1);
		if (cudaPeekAtLastError() != cudaSuccess){
			printf("GPU kernel error at set_float_zero_kernel: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
			return false;
		}
		find_local_max_2d << <griddim, blockdim >> >(d_cc, d_max_local_tot1, d_index_local_tot1, d_posi_count_p1, Npeak1, ws_x1, ws_y1, Nt_ws1, tmpN*Nt_ws1, border_x_p1, border_y_p1);
		if (cudaPeekAtLastError() != cudaSuccess){
			printf("GPU kernel error at find_local_max_2d: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
			return false;
		}
		for (k = 0; k < Npeak_save; k++){
			switch (blockN1_peak_p1){
			case 256:
				reduce_max_index_general<256> << <gridN1_peak_p1*tmpN, 256 >> >(d_max_local_tot1, d_index_inter_p1, d_max_inter1_p1, d_index_inter1_p1, ws_x1 / 4); break;
			case 128:
				reduce_max_index_general<128> << <gridN1_peak_p1*tmpN, 128 >> >(d_max_local_tot1, d_index_inter_p1, d_max_inter1_p1, d_index_inter1_p1, ws_x1 / 4); break;
			case 64:
				reduce_max_index_general<64> << <gridN1_peak_p1*tmpN, 64 >> >(d_max_local_tot1, d_index_inter_p1, d_max_inter1_p1, d_index_inter1_p1, ws_x1 / 4); break;
			case 32:
				reduce_max_index_general<32> << <gridN1_peak_p1*tmpN, 32 >> >(d_max_local_tot1, d_index_inter_p1, d_max_inter1_p1, d_index_inter1_p1, ws_x1 / 4); break;
			case 16:
				reduce_max_index_general<16> << <gridN1_peak_p1*tmpN, 16 >> >(d_max_local_tot1, d_index_inter_p1, d_max_inter1_p1, d_index_inter1_p1, ws_x1 / 4); break;
			case 8:
				reduce_max_index_general<8> << <gridN1_peak_p1*tmpN, 8 >> >(d_max_local_tot1, d_index_inter_p1, d_max_inter1_p1, d_index_inter1_p1, ws_x1 / 4); break;
			case 4:
				reduce_max_index_general<4> << <gridN1_peak_p1*tmpN, 4 >> >(d_max_local_tot1, d_index_inter_p1, d_max_inter1_p1, d_index_inter1_p1, ws_x1 / 4); break;
			case 2:
				reduce_max_index_general<2> << <gridN1_peak_p1*tmpN, 2 >> >(d_max_local_tot1, d_index_inter_p1, d_max_inter1_p1, d_index_inter1_p1, ws_x1 / 4); break;
			case 1:
				reduce_max_index_general<1> << <gridN1_peak_p1*tmpN, 1 >> >(d_max_local_tot1, d_index_inter_p1, d_max_inter1_p1, d_index_inter1_p1, ws_x1 / 4); break;
			}
			if (cudaPeekAtLastError() != cudaSuccess)
			{
				printf("GPU kernel error at reduce_max_index_general: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
				return false;
			}

			switch (blockN2_peak_p1){
			case 256:
				reduce_max_index_final<256> << <gridN2_peak_p1*tmpN, 256 >> >(d_max_inter1_p1, d_index_inter1_p1, d_max_four_peaks_p1, d_index_four_peaks_p1, ws_y1 / 4, k, Npeak_save); break;
			case 128:
				reduce_max_index_final<128> << <gridN2_peak_p1*tmpN, 128 >> >(d_max_inter1_p1, d_index_inter1_p1, d_max_four_peaks_p1, d_index_four_peaks_p1, ws_y1 / 4, k, Npeak_save); break;
			case 64:
				reduce_max_index_final<64> << <gridN2_peak_p1*tmpN, 64 >> >(d_max_inter1_p1, d_index_inter1_p1, d_max_four_peaks_p1, d_index_four_peaks_p1, ws_y1 / 4, k, Npeak_save); break;
			case 32:
				reduce_max_index_final<32> << <gridN2_peak_p1*tmpN, 32 >> >(d_max_inter1_p1, d_index_inter1_p1, d_max_four_peaks_p1, d_index_four_peaks_p1, ws_y1 / 4, k, Npeak_save); break;
			case 16:
				reduce_max_index_final<16> << <gridN2_peak_p1*tmpN, 16 >> >(d_max_inter1_p1, d_index_inter1_p1, d_max_four_peaks_p1, d_index_four_peaks_p1, ws_y1 / 4, k, Npeak_save); break;
			case 8:
				reduce_max_index_final<8> << <gridN2_peak_p1*tmpN, 8 >> >(d_max_inter1_p1, d_index_inter1_p1, d_max_four_peaks_p1, d_index_four_peaks_p1, ws_y1 / 4, k, Npeak_save); break;
			case 4:
				reduce_max_index_final<4> << <gridN2_peak_p1*tmpN, 4 >> >(d_max_inter1_p1, d_index_inter1_p1, d_max_four_peaks_p1, d_index_four_peaks_p1, ws_y1 / 4, k, Npeak_save); break;
			case 2:
				reduce_max_index_final<2> << <gridN2_peak_p1*tmpN, 2 >> >(d_max_inter1_p1, d_index_inter1_p1, d_max_four_peaks_p1, d_index_four_peaks_p1, ws_y1 / 4, k, Npeak_save); break;
			case 1:
				reduce_max_index_final<1> << <gridN2_peak_p1*tmpN, 1 >> >(d_max_inter1_p1, d_index_inter1_p1, d_max_four_peaks_p1, d_index_four_peaks_p1, ws_y1 / 4, k, Npeak_save); break;
			}
			if (cudaPeekAtLastError() != cudaSuccess)
			{
				printf("GPU kernel error at reduce_max_index_final: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
				return false;
			}
			if (k < Npeak_save-1){
				replace_local_max_2d << <griddim, blockdim >> >(d_max_local_tot1, d_index_four_peaks_p1, k, tmpN, Npeak_save);
				if (cudaPeekAtLastError() != cudaSuccess)
				{
					printf("GPU kernel error at replace_local_max_2d: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
					return false;
				}
			}
		}
		peak_subpixel_4peak_2d << <griddim, blockdim >> >(d_valid_p1, d_disp_x_p1, d_disp_y_p1, d_disp_x_filter_p1, d_disp_y_filter_p1, d_max_four_peaks_p1, d_index_four_peaks_p1,
			d_index_local_tot1, d_cc, ws_x1, ws_y1, Nt_ws1, i*Neach1, tmpN * Npeak_save, Npeak_save, dispx_lim1, dispx_lim2, dispy_lim1, dispy_lim2);
	}
	kernel_remove_nan << <griddim, blockdim >> >(d_disp_x_p1, d_disp_y_p1, d_valid_p1, Nt1_vec * Npeak_save);
	if (cudaPeekAtLastError() != cudaSuccess)
	{
		printf("GPU kernel error at kernel_remove_nan: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
		return false;
	}
	return true;
}

bool xcorr2d_multi::MedianFilter_Smoothing_p1(int Npass1_median, int minNneigh_p1, float residual_threshold_p1, float insert_threshold_p1, int Npass1_smooth)
{
	int i;
	int hfwx_median = (wx1_median - 1) / 2;
	int hfwy_median = (wy1_median - 1) / 2;
	int hfwx_smooth = (wx1_smooth - 1) / 2;
	int hfwy_smooth = (wy1_smooth - 1) / 2;

	median_filter_data_prep << <griddim, blockdim >> >(d_disp_x_p1, d_disp_y_p1, d_valid_p1, d_disp_x_filter_p1, d_disp_y_filter_p1, d_valid_filter_p1, Nt1_vec, Npeak_save);
	if (cudaPeekAtLastError() != cudaSuccess){
		printf("GPU kernel error at median_filter_data_prep: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
		return false;
	}
	for (i = 0; i < Npass1_median; i++){
		cudaMemset(d_neighbor_cnt_p1, 0, sizeof(int)*Nt1_vec);
		organize_median_filter_data << <griddim, blockdim >> >(d_disp_x_filter_p1, d_disp_y_filter_p1, d_valid_filter_p1, d_disp_x_neigh_p1, d_disp_y_neigh_p1, d_neighbor_cnt_p1, Nx1_vec,
			Ny1_vec, wx1_median, hfwx_median, hfwy_median, Nt1_median, Nt1_median*Nt1_vec);
		if (cudaPeekAtLastError() != cudaSuccess){
			printf("GPU kernel error at organize_median_filter_data: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
			return false;
		}
		median_validity << < griddim, blockdim >> > (d_valid_median_p1, d_neighbor_cnt_p1, minNneigh_p1, Nt1_vec);
		if (cudaPeekAtLastError() != cudaSuccess){
			printf("GPU kernel error at median_validity: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
			return false;
		}
		Calc_median << <griddim, blockdim >> >(d_median_x_p1, d_valid_median_p1, d_neighbor_cnt_p1, d_disp_x_neigh_p1, Nt1_median, Nt1_vec);
		if (cudaPeekAtLastError() != cudaSuccess){
			printf("GPU kernel error at Calc_median: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
			return false;
		}
		Calc_median << <griddim, blockdim >> >(d_median_y_p1, d_valid_median_p1, d_neighbor_cnt_p1, d_disp_y_neigh_p1, Nt1_median, Nt1_vec);
		if (cudaPeekAtLastError() != cudaSuccess){
			printf("GPU kernel error at Calc_median: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
			return false;
		}
		Calc_devi << <griddim, blockdim >> >(d_disp_x_neigh_p1, d_valid_median_p1, d_neighbor_cnt_p1, d_median_x_p1, d_devi_x_neigh_p1, Nt1_median, Nt1_median*Nt1_vec);
		if (cudaPeekAtLastError() != cudaSuccess){
			printf("GPU kernel error at Calc_devi: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
			return false;
		}
		Calc_devi << <griddim, blockdim >> >(d_disp_y_neigh_p1, d_valid_median_p1, d_neighbor_cnt_p1, d_median_y_p1, d_devi_y_neigh_p1, Nt1_median, Nt1_median*Nt1_vec);
		if (cudaPeekAtLastError() != cudaSuccess){
			printf("GPU kernel error at Calc_devi: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
			return false;
		}
		Calc_median << <griddim, blockdim >> >(d_devimedian_x_p1, d_valid_median_p1, d_neighbor_cnt_p1, d_devi_x_neigh_p1, Nt1_median, Nt1_vec);
		if (cudaPeekAtLastError() != cudaSuccess){
			printf("GPU kernel error at Calc_median: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
			return false;
		}
		Calc_median << <griddim, blockdim >> >(d_devimedian_y_p1, d_valid_median_p1, d_neighbor_cnt_p1, d_devi_y_neigh_p1, Nt1_median, Nt1_vec);
		if (cudaPeekAtLastError() != cudaSuccess){
			printf("GPU kernel error at Calc_median: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
			return false;
		}
		median_filtering << <griddim, blockdim >> >(d_disp_x_p1, d_disp_y_p1, d_median_x_p1, d_median_y_p1, d_devimedian_x_p1, d_devimedian_y_p1, d_valid_median_p1, d_valid_p1,
			d_valid_filter_p1, d_disp_x_filter_p1, d_disp_y_filter_p1, insert_threshold_p1, epsilon, Nt1_vec, Npeak_save);
		if (cudaPeekAtLastError() != cudaSuccess){
			printf("GPU kernel error at median_filtering: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
			return false;
		}
	}
	memset(h_fill_cnt, 1, sizeof(int));
	while (h_fill_cnt[0] > 0){
		cudaMemset(d_fill_cnt, 0, sizeof(int));
		fill_blank_vecfield << <griddim, blockdim >> >(d_disp_x_filter_p1, d_disp_y_filter_p1, d_valid_filter_p1, d_fill_index_p1, d_fill_cnt, Nx1_vec, Ny1_vec, 1, 1, Nt1_vec);
		if (cudaPeekAtLastError() != cudaSuccess){
			printf("GPU kernel error at fill_blank_vecfield: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
			return false;
		}
		fill_blank_vecfield_update << <griddim, blockdim >> >(d_valid_filter_p1, d_fill_index_p1, Nt1_vec);
		if (cudaPeekAtLastError() != cudaSuccess){
			printf("GPU kernel error at fill_blank_vecfield_update: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
			return false;
		}
		cudaMemcpy(h_fill_cnt, d_fill_cnt, sizeof(int), cudaMemcpyDeviceToHost);
	}

	for (i = 0; i < Npass1_smooth; i++){
		cudaMemset(d_neighbor_cnt_p1, 0, sizeof(int)*Nt1_vec);
		organize_smooth_filter_data << <griddim, blockdim >> >(d_disp_x_filter_p1, d_disp_y_filter_p1, d_disp_x_neigh_smooth_p1, d_disp_y_neigh_smooth_p1, d_neighbor_cnt_p1, Nx1_vec, Ny1_vec,
			wx1_smooth, hfwx_smooth, hfwy_smooth, Nt1_smooth, Nt1_vec*Nt1_smooth);
		if (cudaPeekAtLastError() != cudaSuccess){
			printf("GPU kernel error at organize_smooth_filter_data: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
			return false;
		}
		smoothing_tophat << <griddim, blockdim >> >(d_disp_x_filter_p1, d_disp_y_filter_p1, d_disp_x_neigh_smooth_p1, d_disp_y_neigh_smooth_p1, d_neighbor_cnt_p1, Nt1_smooth, Nt1_vec);
		if (cudaPeekAtLastError() != cudaSuccess){
			printf("GPU kernel error at smoothing_tophat: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
			return false;
		}
	}
	return true;
}

void xcorr2d_multi::extend_filtered_vecfield_p1()
{
	int j, k, tmp1, tmp2;
	int Nx_ex = Nx1_vec + 2 * vec_ex_border_p1;
	int Ny_ex = Ny1_vec + 2 * vec_ex_border_p1;

	for (j = 0; j < Ny1_vec; j++){
		tmp1 = (j + vec_ex_border_p1)*Nx_ex;
		tmp2 = j*Nx1_vec;
		cudaMemcpy(d_disp_x_filter_p1_ex + tmp1 + vec_ex_border_p1, d_disp_x_filter_p1 + tmp2, sizeof(float)*Nx1_vec, cudaMemcpyDeviceToDevice);
		cudaMemcpy(d_disp_y_filter_p1_ex + tmp1 + vec_ex_border_p1, d_disp_y_filter_p1 + tmp2, sizeof(float)*Nx1_vec, cudaMemcpyDeviceToDevice);
	}

	for (j = 0; j < Ny1_vec; j++){
		tmp1 = (j + vec_ex_border_p1)*Nx_ex;
		for (k = 0; k < vec_ex_border_p1; k++){
			cudaMemcpy(d_disp_x_filter_p1_ex + tmp1 + k, d_disp_x_filter_p1_ex + tmp1 + vec_ex_border_p1, sizeof(float), cudaMemcpyDeviceToDevice);
			cudaMemcpy(d_disp_x_filter_p1_ex + tmp1 + Nx_ex - 1 - k, d_disp_x_filter_p1_ex + tmp1 + vec_ex_border_p1 + Nx1_vec - 1, sizeof(float), cudaMemcpyDeviceToDevice);
			cudaMemcpy(d_disp_y_filter_p1_ex + tmp1 + k, d_disp_y_filter_p1_ex + tmp1 + vec_ex_border_p1, sizeof(float), cudaMemcpyDeviceToDevice);
			cudaMemcpy(d_disp_y_filter_p1_ex + tmp1 + Nx_ex - 1 - k, d_disp_y_filter_p1_ex + tmp1 + vec_ex_border_p1 + Nx1_vec - 1, sizeof(float), cudaMemcpyDeviceToDevice);
		}
	}

	for (j = 0; j < vec_ex_border_p1; j++){
		cudaMemcpy(d_disp_x_filter_p1_ex + j*Nx_ex, d_disp_x_filter_p1_ex + vec_ex_border_p1*Nx_ex, sizeof(float)*Nx_ex, cudaMemcpyDeviceToDevice);
		cudaMemcpy(d_disp_x_filter_p1_ex + (Ny_ex - 1 - j)*Nx_ex, d_disp_x_filter_p1_ex + (Ny1_vec + vec_ex_border_p1 - 1)*Nx_ex, sizeof(float)*Nx_ex, cudaMemcpyDeviceToDevice);
		cudaMemcpy(d_disp_y_filter_p1_ex + j*Nx_ex, d_disp_y_filter_p1_ex + vec_ex_border_p1*Nx_ex, sizeof(float)*Nx_ex, cudaMemcpyDeviceToDevice);
		cudaMemcpy(d_disp_y_filter_p1_ex + (Ny_ex - 1 - j)*Nx_ex, d_disp_y_filter_p1_ex + (Ny1_vec + vec_ex_border_p1 - 1)*Nx_ex, sizeof(float)*Nx_ex, cudaMemcpyDeviceToDevice);
	}
}

bool xcorr2d_multi::Upsampling_vector_field()
{
	Upsampling_vecfield_kernel_2d << <griddim, blockdim >> >(d_disp_x_filter_p2, d_disp_y_filter_p2, d_disp_x_filter_p1_ex, d_disp_y_filter_p1_ex, Nx1_vec + 2 * vec_ex_border_p1,
		spacingx1 / 2 - vec_ex_border_p1*spacingx1, spacingy1 / 2 - vec_ex_border_p1*spacingy1, (float)spacingx1, (float)spacingy1, spacingx2, spacingy2, Nx2_vec, Nt2_vec);
	if (cudaPeekAtLastError() != cudaSuccess){
		printf("GPU kernel error at Upsampling_vecfield_kernel_2d: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
		return false;
	}
	return true;
}

void xcorr2d_multi::extend_filtered_vecfield_p2()
{
	int j, k, tmp1, tmp2;
	int Nx_ex = Nx2_vec + 2 * vec_ex_border_p2;
	int Ny_ex = Ny2_vec + 2 * vec_ex_border_p2;

	for (j = 0; j < Ny2_vec; j++){
		tmp1 = (j + vec_ex_border_p2)*Nx_ex;
		tmp2 = j*Nx2_vec;
		cudaMemcpy(d_disp_x_filter_p2_ex + tmp1 + vec_ex_border_p2, d_disp_x_filter_p2 + tmp2, sizeof(float)*Nx2_vec, cudaMemcpyDeviceToDevice);
		cudaMemcpy(d_disp_y_filter_p2_ex + tmp1 + vec_ex_border_p2, d_disp_y_filter_p2 + tmp2, sizeof(float)*Nx2_vec, cudaMemcpyDeviceToDevice);
	}

	for (j = 0; j < Ny2_vec; j++){
		tmp1 = (j + vec_ex_border_p2)*Nx_ex;
		for (k = 0; k < vec_ex_border_p2; k++){
			cudaMemcpy(d_disp_x_filter_p2_ex + tmp1 + k, d_disp_x_filter_p2_ex + tmp1 + vec_ex_border_p2, sizeof(float), cudaMemcpyDeviceToDevice);
			cudaMemcpy(d_disp_x_filter_p2_ex + tmp1 + Nx_ex - 1 - k, d_disp_x_filter_p2_ex + tmp1 + vec_ex_border_p2 + Nx2_vec - 1, sizeof(float), cudaMemcpyDeviceToDevice);
			cudaMemcpy(d_disp_y_filter_p2_ex + tmp1 + k, d_disp_y_filter_p2_ex + tmp1 + vec_ex_border_p2, sizeof(float), cudaMemcpyDeviceToDevice);
			cudaMemcpy(d_disp_y_filter_p2_ex + tmp1 + Nx_ex - 1 - k, d_disp_y_filter_p2_ex + tmp1 + vec_ex_border_p2 + Nx2_vec - 1, sizeof(float), cudaMemcpyDeviceToDevice);
		}
	}

	for (j = 0; j < vec_ex_border_p2; j++){
		cudaMemcpy(d_disp_x_filter_p2_ex + j*Nx_ex, d_disp_x_filter_p2_ex + vec_ex_border_p2*Nx_ex, sizeof(float)*Nx_ex, cudaMemcpyDeviceToDevice);
		cudaMemcpy(d_disp_x_filter_p2_ex + (Ny_ex - 1 - j)*Nx_ex, d_disp_x_filter_p2_ex + (Ny2_vec + vec_ex_border_p2 - 1)*Nx_ex, sizeof(float)*Nx_ex, cudaMemcpyDeviceToDevice);
		cudaMemcpy(d_disp_y_filter_p2_ex + j*Nx_ex, d_disp_y_filter_p2_ex + vec_ex_border_p2*Nx_ex, sizeof(float)*Nx_ex, cudaMemcpyDeviceToDevice);
		cudaMemcpy(d_disp_y_filter_p2_ex + (Ny_ex - 1 - j)*Nx_ex, d_disp_y_filter_p2_ex + (Ny2_vec + vec_ex_border_p2 - 1)*Nx_ex, sizeof(float)*Nx_ex, cudaMemcpyDeviceToDevice);
	}
}

bool xcorr2d_multi::Calc_cc_map_p2()
{
	int i, j, k, tmpN;
	
	Kernel_Window_Deformation_2d << <griddim, blockdim >> >(d_img_deform_p2, d_img2_p2_aug_ex, -hfws_x2, -hfws_y2, Nx2_aug_ex, d_disp_x_filter_p2_ex, d_disp_y_filter_p2_ex,
		(float)(hfws_x2 - spacingx2*vec_ex_border_p2), (float)(hfws_y2 - spacingy2*vec_ex_border_p2), (float)spacingx2, (float)spacingy2, Nx2_vec + 2 * vec_ex_border_p2, Nx2_aug, Nt2_aug);
	if (cudaPeekAtLastError() != cudaSuccess){
		printf("GPU kernel error at Kernel_Window_Deformation_2d: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
		return false;
	}

	for (i = 0; i < Nsec2_cc; i++){
		if (i == Nsec2_cc - 1){
			tmpN = Nlast2;
		}
		else{
			tmpN = Neach2;
		}
		organize_data_fft2d_multi << <griddim, blockdim >> >(d_img1_p2_aug_ex, d_xcorr1, Nx2_vec, ws_x2, spacingx2, spacingy2, Nt_ws2, hfws_x2, hfws_y2, Nx2_aug_ex, i*Neach2, tmpN*Nt_ws2);
		if (cudaPeekAtLastError() != cudaSuccess){
			printf("GPU kernel error at organize_data_fft2d_multi: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
			return false;
		}
		organize_data_fft2d_multi_deformed << <griddim, blockdim >> >(d_img_deform_p2, d_xcorr2, Nx2_vec, ws_x2, spacingx2, spacingy2, Nt_ws2, Nx2_aug, i*Neach2, tmpN*Nt_ws2);
		if (cudaPeekAtLastError() != cudaSuccess){
			printf("GPU kernel error at organize_data_fft2d_multi_deformed: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
			return false;
		}
	
		if (cufftExecC2C(fftplan2, d_xcorr1, d_xcorr1, CUFFT_FORWARD) != CUFFT_SUCCESS){
			cout << "Error occurs during execution of cufftExecC2C!" << endl;
			return false;
		}
		if (cufftExecC2C(fftplan2, d_xcorr2, d_xcorr2, CUFFT_FORWARD) != CUFFT_SUCCESS){
			cout << "Error occurs during execution of cufftExecC2C!" << endl;
			return false;
		}
		multiply_conjugate << <griddim, blockdim >> >(d_xcorr1, d_xcorr2, tmpN*Nt_ws2);
		if (cudaPeekAtLastError() != cudaSuccess){
			printf("GPU kernel error at multiply_conjugate: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
			return false;
		}
		if (cufftExecC2C(fftplan2, d_xcorr1, d_xcorr1, CUFFT_INVERSE) != CUFFT_SUCCESS){
			cout << "Error occurs during execution of cufftExecC2C!" << endl;
			return false;
		}
		complex2amp_xcorr2d << <griddim, blockdim >> >(d_xcorr1, d_cc, (float)Nt_ws2, tmpN*Nt_ws2);
		if (cudaPeekAtLastError() != cudaSuccess){
			printf("GPU kernel error at complex2amp_xcorr2d: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
			return false;
		}
		fftshift2d_xcorr2d_row << <griddim, blockdim >> >(d_cc, ws_x2, ws_y2, Nt_ws2 / 2, Nt_ws2*tmpN);
		if (cudaPeekAtLastError() != cudaSuccess){
			printf("GPU kernel error at fftshift2d_xcorr2d_row: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
			return false;
		}
		fftshift2d_xcorr2d_col << <griddim, blockdim >> >(d_cc, ws_x2 / 2, Nt_ws2 / 2, Nt_ws2*tmpN);
		if (cudaPeekAtLastError() != cudaSuccess){
			printf("GPU kernel error at fftshift2d_xcorr2d_col: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
			return false;
		}
		cudaMemset(d_posi_count_p2, 0, sizeof(int)*Neach2);
		cudaMemset(d_index_local_tot2, 0, sizeof(int)*Neach2*Npeak2);
		set_float_zero_kernel << <griddim, blockdim >> >(d_max_local_tot2, Neach2*Npeak2);
		if (cudaPeekAtLastError() != cudaSuccess){
			printf("GPU kernel error at set_float_zero_kernel: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
			return false;
		}
		find_local_max_2d << <griddim, blockdim >> >(d_cc, d_max_local_tot2, d_index_local_tot2, d_posi_count_p2, Npeak2, ws_x2, ws_y2, Nt_ws2, tmpN*Nt_ws2, border_x_p2, border_y_p2);
		if (cudaPeekAtLastError() != cudaSuccess){
			printf("GPU kernel error at find_local_max_2d: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
			return false;
		}
		for (k = 0; k < Npeak_save; k++){
			switch (blockN1_peak_p2){
			case 256:
				reduce_max_index_general<256> << <gridN1_peak_p2*tmpN, 256 >> >(d_max_local_tot2, d_index_inter_p2, d_max_inter1_p2, d_index_inter1_p2, ws_x2 / 4); break;
			case 128:
				reduce_max_index_general<128> << <gridN1_peak_p2*tmpN, 128 >> >(d_max_local_tot2, d_index_inter_p2, d_max_inter1_p2, d_index_inter1_p2, ws_x2 / 4); break;
			case 64:
				reduce_max_index_general<64> << <gridN1_peak_p2*tmpN, 64 >> >(d_max_local_tot2, d_index_inter_p2, d_max_inter1_p2, d_index_inter1_p2, ws_x2 / 4); break;
			case 32:
				reduce_max_index_general<32> << <gridN1_peak_p2*tmpN, 32 >> >(d_max_local_tot2, d_index_inter_p2, d_max_inter1_p2, d_index_inter1_p2, ws_x2 / 4); break;
			case 16:
				reduce_max_index_general<16> << <gridN1_peak_p2*tmpN, 16 >> >(d_max_local_tot2, d_index_inter_p2, d_max_inter1_p2, d_index_inter1_p2, ws_x2 / 4); break;
			case 8:
				reduce_max_index_general<8> << <gridN1_peak_p2*tmpN, 8 >> >(d_max_local_tot2, d_index_inter_p2, d_max_inter1_p2, d_index_inter1_p2, ws_x2 / 4); break;
			case 4:
				reduce_max_index_general<4> << <gridN1_peak_p2*tmpN, 4 >> >(d_max_local_tot2, d_index_inter_p2, d_max_inter1_p2, d_index_inter1_p2, ws_x2 / 4); break;
			case 2:
				reduce_max_index_general<2> << <gridN1_peak_p2*tmpN, 2 >> >(d_max_local_tot2, d_index_inter_p2, d_max_inter1_p2, d_index_inter1_p2, ws_x2 / 4); break;
			case 1:
				reduce_max_index_general<1> << <gridN1_peak_p2*tmpN, 1 >> >(d_max_local_tot2, d_index_inter_p2, d_max_inter1_p2, d_index_inter1_p2, ws_x2 / 4); break;
			}
			if (cudaPeekAtLastError() != cudaSuccess)
			{
				printf("GPU kernel error at reduce_max_index_general: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
				return false;
			}

			switch (blockN2_peak_p2){
			case 256:
				reduce_max_index_final<256> << <gridN2_peak_p2*tmpN, 256 >> >(d_max_inter1_p2, d_index_inter1_p2, d_max_four_peaks_p2, d_index_four_peaks_p2, ws_y2 / 4, k, Npeak_save); break;
			case 128:
				reduce_max_index_final<128> << <gridN2_peak_p2*tmpN, 128 >> >(d_max_inter1_p2, d_index_inter1_p2, d_max_four_peaks_p2, d_index_four_peaks_p2, ws_y2 / 4, k, Npeak_save); break;
			case 64:
				reduce_max_index_final<64> << <gridN2_peak_p2*tmpN, 64 >> >(d_max_inter1_p2, d_index_inter1_p2, d_max_four_peaks_p2, d_index_four_peaks_p2, ws_y2 / 4, k, Npeak_save); break;
			case 32:
				reduce_max_index_final<32> << <gridN2_peak_p2*tmpN, 32 >> >(d_max_inter1_p2, d_index_inter1_p2, d_max_four_peaks_p2, d_index_four_peaks_p2, ws_y2 / 4, k, Npeak_save); break;
			case 16:
				reduce_max_index_final<16> << <gridN2_peak_p2*tmpN, 16 >> >(d_max_inter1_p2, d_index_inter1_p2, d_max_four_peaks_p2, d_index_four_peaks_p2, ws_y2 / 4, k, Npeak_save); break;
			case 8:
				reduce_max_index_final<8> << <gridN2_peak_p2*tmpN, 8 >> >(d_max_inter1_p2, d_index_inter1_p2, d_max_four_peaks_p2, d_index_four_peaks_p2, ws_y2 / 4, k, Npeak_save); break;
			case 4:
				reduce_max_index_final<4> << <gridN2_peak_p2*tmpN, 4 >> >(d_max_inter1_p2, d_index_inter1_p2, d_max_four_peaks_p2, d_index_four_peaks_p2, ws_y2 / 4, k, Npeak_save); break;
			case 2:
				reduce_max_index_final<2> << <gridN2_peak_p2*tmpN, 2 >> >(d_max_inter1_p2, d_index_inter1_p2, d_max_four_peaks_p2, d_index_four_peaks_p2, ws_y2 / 4, k, Npeak_save); break;
			case 1:
				reduce_max_index_final<1> << <gridN2_peak_p2*tmpN, 1 >> >(d_max_inter1_p2, d_index_inter1_p2, d_max_four_peaks_p2, d_index_four_peaks_p2, ws_y2 / 4, k, Npeak_save); break;
			}
			if (cudaPeekAtLastError() != cudaSuccess)
			{
				printf("GPU kernel error at reduce_max_index_final: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
				return false;
			}
			if (k < Npeak_save -1){
				replace_local_max_2d << <griddim, blockdim >> >(d_max_local_tot2, d_index_four_peaks_p2, k, tmpN, Npeak_save);
				if (cudaPeekAtLastError() != cudaSuccess)
				{
					printf("GPU kernel error at replace_local_max_2d: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
					return false;
				}
			}
		}
		peak_subpixel_4peak_2d << <griddim, blockdim >> >(d_valid_p2, d_disp_x_p2, d_disp_y_p2, d_disp_x_filter_p2, d_disp_y_filter_p2, d_max_four_peaks_p2, d_index_four_peaks_p2,
			d_index_local_tot2, d_cc, ws_x2, ws_y2, Nt_ws2, i*Neach2, tmpN * Npeak_save, Npeak_save, dispx_lim1, dispx_lim2, dispy_lim1, dispy_lim2);
	}
	kernel_remove_nan << <griddim, blockdim >> >(d_disp_x_p2, d_disp_y_p2, d_valid_p2, Nt2_vec * Npeak_save);
	if (cudaPeekAtLastError() != cudaSuccess)
	{
		printf("GPU kernel error at kernel_remove_nan: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
		return false;
	}
	return true;
}

bool xcorr2d_multi::MedianFilter_Smoothing_p2(int Npass2_median, int minNneigh_p2, float residual_threshold_p2, float insert_threshold_p2, int Npass2_smooth, int iffill_p2)
{
	int i;
	int hfwx_median = (wx2_median - 1) / 2;
	int hfwy_median = (wy2_median - 1) / 2;
	int hfwx_smooth = (wx2_smooth - 1) / 2;
	int hfwy_smooth = (wy2_smooth - 1) / 2;

	median_filter_data_prep << <griddim, blockdim >> >(d_disp_x_p2, d_disp_y_p2, d_valid_p2, d_disp_x_filter_p2, d_disp_y_filter_p2, d_valid_filter_p2, Nt2_vec, Npeak_save);
	if (cudaPeekAtLastError() != cudaSuccess){
		printf("GPU kernel error at median_filter_data_prep: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
		return false;
	}
	for (i = 0; i < Npass2_median; i++){
		cudaMemset(d_neighbor_cnt_p2, 0, sizeof(int)*Nt2_vec);
		organize_median_filter_data << <griddim, blockdim >> >(d_disp_x_filter_p2, d_disp_y_filter_p2, d_valid_filter_p2, d_disp_x_neigh_p2, d_disp_y_neigh_p2, d_neighbor_cnt_p2, Nx2_vec,
			Ny2_vec, wx2_median, hfwx_median, hfwy_median, Nt2_median, Nt2_median*Nt2_vec);
		if (cudaPeekAtLastError() != cudaSuccess){
			printf("GPU kernel error at organize_median_filter_data: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
			return false;
		}
		median_validity << < griddim, blockdim >> > (d_valid_median_p2, d_neighbor_cnt_p2, minNneigh_p2, Nt2_vec);
		if (cudaPeekAtLastError() != cudaSuccess){
			printf("GPU kernel error at median_validity: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
			return false;
		}
		Calc_median << <griddim, blockdim >> >(d_median_x_p2, d_valid_median_p2, d_neighbor_cnt_p2, d_disp_x_neigh_p2, Nt2_median, Nt2_vec);
		if (cudaPeekAtLastError() != cudaSuccess){
			printf("GPU kernel error at Calc_median: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
			return false;
		}
		Calc_median << <griddim, blockdim >> >(d_median_y_p2, d_valid_median_p2, d_neighbor_cnt_p2, d_disp_y_neigh_p2, Nt2_median, Nt2_vec);
		if (cudaPeekAtLastError() != cudaSuccess){
			printf("GPU kernel error at Calc_median: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
			return false;
		}
		Calc_devi << <griddim, blockdim >> >(d_disp_x_neigh_p2, d_valid_median_p2, d_neighbor_cnt_p2, d_median_x_p2, d_devi_x_neigh_p2, Nt2_median, Nt2_median*Nt2_vec);
		if (cudaPeekAtLastError() != cudaSuccess){
			printf("GPU kernel error at Calc_devi: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
			return false;
		}
		Calc_devi << <griddim, blockdim >> >(d_disp_y_neigh_p2, d_valid_median_p2, d_neighbor_cnt_p2, d_median_y_p2, d_devi_y_neigh_p2, Nt2_median, Nt2_median*Nt2_vec);
		if (cudaPeekAtLastError() != cudaSuccess){
			printf("GPU kernel error at Calc_devi: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
			return false;
		}
		Calc_median << <griddim, blockdim >> >(d_devimedian_x_p2, d_valid_median_p2, d_neighbor_cnt_p2, d_devi_x_neigh_p2, Nt2_median, Nt2_vec);
		if (cudaPeekAtLastError() != cudaSuccess){
			printf("GPU kernel error at Calc_median: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
			return false;
		}
		Calc_median << <griddim, blockdim >> >(d_devimedian_y_p2, d_valid_median_p2, d_neighbor_cnt_p2, d_devi_y_neigh_p2, Nt2_median, Nt2_vec);
		if (cudaPeekAtLastError() != cudaSuccess){
			printf("GPU kernel error at Calc_median: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
			return false;
		}
		median_filtering << <griddim, blockdim >> >(d_disp_x_p2, d_disp_y_p2, d_median_x_p2, d_median_y_p2, d_devimedian_x_p2, d_devimedian_y_p2, d_valid_median_p2, d_valid_p2,
			d_valid_filter_p2, d_disp_x_filter_p2, d_disp_y_filter_p2, insert_threshold_p2, epsilon, Nt2_vec, Npeak_save);
		if (cudaPeekAtLastError() != cudaSuccess){
			printf("GPU kernel error at median_filtering: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
			return false;
		}
	}

	if (iffill_p2 == 1){
		memset(h_fill_cnt, 1, sizeof(int));
		while (h_fill_cnt[0] > 0){
			cudaMemset(d_fill_cnt, 0, sizeof(int));
			fill_blank_vecfield << <griddim, blockdim >> >(d_disp_x_filter_p2, d_disp_y_filter_p2, d_valid_filter_p2, d_fill_index_p2, d_fill_cnt, Nx2_vec, Ny2_vec, 1, 1, Nt2_vec);
			if (cudaPeekAtLastError() != cudaSuccess){
				printf("GPU kernel error at fill_blank_vecfield: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
				return false;
			}
			fill_blank_vecfield_update << <griddim, blockdim >> >(d_valid_filter_p2, d_fill_index_p2, Nt2_vec);
			if (cudaPeekAtLastError() != cudaSuccess){
				printf("GPU kernel error at fill_blank_vecfield_update: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
				return false;
			}
			cudaMemcpy(h_fill_cnt, d_fill_cnt, sizeof(int), cudaMemcpyDeviceToHost);
		}
	}

	for (i = 0; i < Npass2_smooth; i++){
		cudaMemset(d_neighbor_cnt_p2, 0, sizeof(int)*Nt2_vec);
		organize_smooth_filter_data << <griddim, blockdim >> >(d_disp_x_filter_p2, d_disp_y_filter_p2, d_disp_x_neigh_smooth_p2, d_disp_y_neigh_smooth_p2, d_neighbor_cnt_p2, Nx2_vec, Ny2_vec,
			wx2_smooth, hfwx_smooth, hfwy_smooth, Nt2_smooth, Nt2_vec*Nt2_smooth);
		if (cudaPeekAtLastError() != cudaSuccess){
			printf("GPU kernel error at organize_smooth_filter_data: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
			return false;
		}
		smoothing_tophat << <griddim, blockdim >> >(d_disp_x_filter_p2, d_disp_y_filter_p2, d_disp_x_neigh_smooth_p2, d_disp_y_neigh_smooth_p2, d_neighbor_cnt_p2, Nt2_smooth, Nt2_vec);
		if (cudaPeekAtLastError() != cudaSuccess){
			printf("GPU kernel error at smoothing_tophat: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
			return false;
		}
	}
	return true;
}

bool xcorr2d_multi::write_vec_data_general(CString dir_vec, int selection)
{
	int i;
	switch (selection){
	case 0:
		cudaMemcpy(h_disp_x_p1, d_disp_x_p1, sizeof(float)*Nt1_vec * Npeak_save, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_disp_y_p1, d_disp_y_p1, sizeof(float)*Nt1_vec * Npeak_save, cudaMemcpyDeviceToHost);
		outputfile.open(dir_vec);
		if (outputfile.is_open())
		{
			outputfile << setprecision(2);
			outputfile << fixed;
			outputfile << "TITLE = \"B00001\"\n";
			outputfile << "VARIABLES = \"x\", \"y\", \"dx1\", \"dy1\", \"dx2\", \"dy2\", \"dx3\", \"dy3\", \"dx4\", \"dy4\"\n";
			outputfile << "ZONE T=\"Frame 0\", I=" << Nx1_vec << ", J=" << Ny1_vec << "\n";
			for (i = 0; i < Nt1_vec; ++i)
			{
				outputfile << h_gridx1[i] << "\t" << h_gridy1[i] << "\t"
					<< h_disp_x_p1[i * Npeak_save] << "\t" << h_disp_y_p1[i * Npeak_save] << "\t"
					<< h_disp_x_p1[i * Npeak_save + 1] << "\t" << h_disp_y_p1[i * Npeak_save + 1] << "\t"
					<< h_disp_x_p1[i * Npeak_save + 2] << "\t" << h_disp_y_p1[i * Npeak_save + 2] << "\t"
					<< h_disp_x_p1[i * Npeak_save + 3] << "\t" << h_disp_y_p1[i * Npeak_save + 3] << "\n";
			}
			outputfile.close();
		}
		else{
			printf("Error occurs while the vector file is being written.");
			outputfile.close();
			return false;
		}
		break;
	case 1:
		cudaMemcpy(h_disp_x_filter_p1, d_disp_x_filter_p1, sizeof(float)*Nt1_vec, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_disp_y_filter_p1, d_disp_y_filter_p1, sizeof(float)*Nt1_vec, cudaMemcpyDeviceToHost);
		outputfile.open(dir_vec);
		if (outputfile.is_open())
		{
			outputfile << setprecision(2);
			outputfile << fixed;
			outputfile << "TITLE = \"B00001\"\n";
			outputfile << "VARIABLES = \"x\", \"y\", \"displacement-x\", \"displacement-y\"\n";
			outputfile << "ZONE T=\"Frame 0\", I=" << Nx1_vec << ", J=" << Ny1_vec << "\n";
			for (i = 0; i < Nt1_vec; ++i)
			{
				outputfile << h_gridx1[i] << "\t" << h_gridy1[i] << "\t"
					<< h_disp_x_filter_p1[i] << "\t" << h_disp_y_filter_p1[i] << "\n";
			}
			outputfile.close();
		}
		else{
			printf("Error occurs while the vector file is being written.");
			outputfile.close();
			return false;
		}
		break;
	case 2:
		cudaMemcpy(h_disp_x_p2, d_disp_x_p2, sizeof(float)*Nt2_vec * Npeak_save, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_disp_y_p2, d_disp_y_p2, sizeof(float)*Nt2_vec * Npeak_save, cudaMemcpyDeviceToHost);
		outputfile.open(dir_vec);
		if (outputfile.is_open())
		{
			outputfile << setprecision(2);
			outputfile << fixed;
			outputfile << "TITLE = \"B00001\"\n";
			outputfile << "VARIABLES = \"x\", \"y\", \"dx1\", \"dy1\", \"dx2\", \"dy2\", \"dx3\", \"dy3\", \"dx4\", \"dy4\"\n";
			outputfile << "ZONE T=\"Frame 0\", I=" << Nx2_vec << ", J=" << Ny2_vec << "\n";
			for (i = 0; i < Nt2_vec; ++i)
			{
				outputfile << h_gridx2[i] << "\t" << h_gridy2[i] << "\t"
					<< h_disp_x_p2[i * Npeak_save] << "\t" << h_disp_y_p2[i * Npeak_save] << "\t"
					<< h_disp_x_p2[i * Npeak_save + 1] << "\t" << h_disp_y_p2[i * Npeak_save + 1] << "\t"
					<< h_disp_x_p2[i * Npeak_save + 2] << "\t" << h_disp_y_p2[i * Npeak_save + 2] << "\t"
					<< h_disp_x_p2[i * Npeak_save + 3] << "\t" << h_disp_y_p2[i * Npeak_save + 3] << "\n";
			}
			outputfile.close();
		}
		else{
			printf("Error occurs while the vector file is being written.");
			outputfile.close();
			return false;
		}
		break;
	case 3:
		cudaMemcpy(h_disp_x_filter_p2, d_disp_x_filter_p2, sizeof(float)*Nt2_vec, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_disp_y_filter_p2, d_disp_y_filter_p2, sizeof(float)*Nt2_vec, cudaMemcpyDeviceToHost);
		outputfile.open(dir_vec);
		if (outputfile.is_open())
		{
			outputfile << setprecision(2);
			outputfile << fixed;
			outputfile << "TITLE = \"B00001\"\n";
			outputfile << "VARIABLES = \"x\", \"y\", \"displacement-x\", \"displacement-y\"\n";
			outputfile << "ZONE T=\"Frame 0\", I=" << Nx2_vec << ", J=" << Ny2_vec << "\n";
			for (i = 0; i < Nt2_vec; ++i)
			{
				outputfile << h_gridx2[i] << "\t" << h_gridy2[i] << "\t"
					<< h_disp_x_filter_p2[i] << "\t" << h_disp_y_filter_p2[i] << "\n";
			}
			outputfile.close();
		}
		else{
			printf("Error occurs while the vector file is being written.");
			outputfile.close();
			return false;
		}
		break;
	}
	return true;
}


void xcorr2d_multi::freemem()
{
	cudaFreeHost(h_img1_org);
	cudaFreeHost(h_img2_org);
	cudaFree(d_img1_p1_aug_ex);
	cudaFree(d_img2_p1_aug_ex);
	cudaFree(d_img1_p2_aug_ex);
	cudaFree(d_img2_p2_aug_ex);
	cudaFree(d_xcorr1);
	cudaFree(d_xcorr2);
	cudaFree(d_cc);
	cudaFree(d_img_deform_p1);
	cudaFree(d_img_deform_p2);
	cudaFree(d_posi_count_p1);
	cudaFree(d_posi_count_p2);
	cudaFree(d_index_local_tot1);
	cudaFree(d_index_local_tot2);
	cudaFree(d_max_local_tot1);
	cudaFree(d_max_local_tot2);
	cudaFree(d_index_inter_p1);
	cudaFree(d_index_inter_p2);
	cudaFree(d_max_inter1_p1);
	cudaFree(d_max_inter1_p2);
	cudaFree(d_index_inter1_p1);
	cudaFree(d_index_inter1_p2);
	cudaFree(d_index_four_peaks_p1);
	cudaFree(d_index_four_peaks_p2);
	cudaFree(d_max_four_peaks_p1);
	cudaFree(d_max_four_peaks_p2);
	cudaFree(d_valid_p1);
	cudaFree(d_valid_p2);
	cudaFree(d_disp_x_p1);
	cudaFree(d_disp_y_p1);
	cudaFree(d_disp_x_p2);
	cudaFree(d_disp_y_p2);
	cudaFree(d_disp_x_filter_p1);
	cudaFree(d_disp_y_filter_p1);
	cudaFree(d_disp_x_filter_p2);
	cudaFree(d_disp_y_filter_p2);
	cudaFree(d_valid_filter_p1);
	cudaFree(d_valid_filter_p2);
	cudaFree(d_neighbor_cnt_p1);
	cudaFree(d_neighbor_cnt_p2);
	cudaFree(d_disp_x_neigh_p1);
	cudaFree(d_disp_y_neigh_p1);
	cudaFree(d_disp_x_neigh_p2);
	cudaFree(d_disp_y_neigh_p2);
	cudaFree(d_valid_median_p1);
	cudaFree(d_valid_median_p2);
	cudaFree(d_median_x_p1);
	cudaFree(d_median_y_p1);
	cudaFree(d_median_x_p2);
	cudaFree(d_median_y_p2);
	cudaFree(d_devi_x_neigh_p1);
	cudaFree(d_devi_y_neigh_p1);
	cudaFree(d_devi_x_neigh_p2);
	cudaFree(d_devi_y_neigh_p2);
	cudaFree(d_devimedian_x_p1);
	cudaFree(d_devimedian_y_p1);
	cudaFree(d_devimedian_x_p2);
	cudaFree(d_devimedian_y_p2);
	cudaFree(d_fill_index_p1);
	cudaFree(d_fill_index_p2);
	cudaFree(d_fill_cnt);
	cudaFreeHost(h_fill_cnt);
	cudaFree(d_disp_x_neigh_smooth_p1);
	cudaFree(d_disp_y_neigh_smooth_p1);
	cudaFree(d_disp_x_neigh_smooth_p2);
	cudaFree(d_disp_y_neigh_smooth_p2);
	cudaFree(d_disp_x_filter_p1_ex);
	cudaFree(d_disp_y_filter_p1_ex);
	cudaFree(d_disp_x_filter_p2_ex);
	cudaFree(d_disp_y_filter_p2_ex);
	cudaFreeHost(h_gridx1);
	cudaFreeHost(h_gridy1);
	cudaFreeHost(h_gridx2);
	cudaFreeHost(h_gridy2);
	cudaFreeHost(h_disp_x_p1);
	cudaFreeHost(h_disp_y_p1);
	cudaFreeHost(h_valid_p1);
	cudaFreeHost(h_disp_x_p2);
	cudaFreeHost(h_disp_y_p2);
	cudaFreeHost(h_valid_p2);
	cudaFreeHost(h_disp_x_filter_p1);
	cudaFreeHost(h_disp_y_filter_p1);
	cudaFreeHost(h_disp_x_filter_p2);
	cudaFreeHost(h_disp_y_filter_p2);
}