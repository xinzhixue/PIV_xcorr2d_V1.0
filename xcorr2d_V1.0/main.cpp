#include "ImageIOPkg.h"
#include "gpuinfo.h"
#include "xcorr2d_multi_grid.cuh"
#include "FolderFileOp.h"
#include "MHE7_4_gpu.h"
typedef unsigned short WORD;
typedef unsigned long  DWORD;
using namespace std;

int main(int argc, char **argv)
{
	CString m_prefix_f1;
	CString m_suffix_f1;
	CString m_prefix_f2;
	CString m_suffix_f2;
	CString m_dir_index;
	CString m_folder_result;
	int m_device_id;
	int m_ifmhe;
	int m_mhe_ws;
	float m_mhe_thresh;
	int m_Nx;
	int m_Ny;
	int m_ws_x1;
	int m_ws_y1;
	int m_spacingx1;
	int m_spacingy1;
	int m_Npass1;
	int m_ws_x2;
	int m_ws_y2;
	int m_spacingx2;
	int m_spacingy2;
	int m_Npass2;
	int m_Nsec_cc;
	int m_wx1_median;
	int m_wy1_median;
	int m_Npass1_median;
	int m_minNneigh_p1;
	float m_residual_threshold_p1;
	float m_insert_threshold_p1;
	int m_wx1_smooth;
	int m_wy1_smooth;
	int m_Npass1_smooth;
	int m_wx2_median;
	int m_wy2_median;
	int m_Npass2_median;
	int m_minNneigh_p2;
	float m_residual_threshold_p2;
	float m_insert_threshold_p2;
	int m_wx2_smooth;
	int m_wy2_smooth;
	int m_Npass2_smooth;
	float m_max_dispx_win_percent1;
	float m_max_dispy_win_percent1;
	float m_max_dispx_win_percent2;
	float m_max_dispy_win_percent2;
	int m_Npeak_save;
	float m_dispx_lim1;
	float m_dispx_lim2;
	float m_dispy_lim1;
	float m_dispy_lim2;


	CString dir_para;
	CString tmpCstring;
	bool statusflag = true;
	int dim1_grid = 10000;
	int dim2_grid = 1;
	int dim3_grid = 1;
	int dim1_block = 512;
	int dim2_block = 1;
	int dim3_block = 1;
	CString dir_img1;
	CString dir_img2;
	int i, j;
	CString vec_name;
	CString dir_result;

	unsigned char* d_img_org;

	//CString dir_img_out1;
	//CString dir_img_out2;

	clock_t begin, end;
	double elapsed_secs;

	if (argc == 1){
		dir_para = _T("Parameter.dat");
	}
	else{
		dir_para = CString(argv[1], strlen(argv[1]) + 1);
	}
	CStdioFile fin;
	if (fin.Open(dir_para, CFile::modeRead))
	{
		fin.SeekToBegin();
		statusflag = statusflag && fin.ReadString(m_prefix_f1);
		statusflag = statusflag && fin.ReadString(m_suffix_f1);
		statusflag = statusflag && fin.ReadString(m_prefix_f2);
		statusflag = statusflag && fin.ReadString(m_suffix_f2);
		statusflag = statusflag && fin.ReadString(m_dir_index);
		statusflag = statusflag && fin.ReadString(m_folder_result);
		statusflag = statusflag && fin.ReadString(tmpCstring); m_device_id = _wtoi(tmpCstring);
		statusflag = statusflag && fin.ReadString(tmpCstring); m_ifmhe = _wtoi(tmpCstring);
		statusflag = statusflag && fin.ReadString(tmpCstring); m_mhe_ws = _wtoi(tmpCstring);
		statusflag = statusflag && fin.ReadString(tmpCstring); m_mhe_thresh = _wtof(tmpCstring);
		statusflag = statusflag && fin.ReadString(tmpCstring); m_Nx = _wtoi(tmpCstring);
		statusflag = statusflag && fin.ReadString(tmpCstring); m_Ny = _wtoi(tmpCstring);
		statusflag = statusflag && fin.ReadString(tmpCstring); m_ws_x1 = _wtoi(tmpCstring);
		statusflag = statusflag && fin.ReadString(tmpCstring); m_ws_y1 = _wtoi(tmpCstring);
		statusflag = statusflag && fin.ReadString(tmpCstring); m_spacingx1 = _wtoi(tmpCstring);
		statusflag = statusflag && fin.ReadString(tmpCstring); m_spacingy1 = _wtoi(tmpCstring);
		statusflag = statusflag && fin.ReadString(tmpCstring); m_Npass1 = _wtoi(tmpCstring);
		statusflag = statusflag && fin.ReadString(tmpCstring); m_ws_x2 = _wtoi(tmpCstring);
		statusflag = statusflag && fin.ReadString(tmpCstring); m_ws_y2 = _wtoi(tmpCstring);
		statusflag = statusflag && fin.ReadString(tmpCstring); m_spacingx2 = _wtoi(tmpCstring);
		statusflag = statusflag && fin.ReadString(tmpCstring); m_spacingy2 = _wtoi(tmpCstring);
		statusflag = statusflag && fin.ReadString(tmpCstring); m_Npass2 = _wtoi(tmpCstring);
		statusflag = statusflag && fin.ReadString(tmpCstring); m_Nsec_cc = _wtoi(tmpCstring);
		statusflag = statusflag && fin.ReadString(tmpCstring); m_wx1_median = _wtoi(tmpCstring);
		statusflag = statusflag && fin.ReadString(tmpCstring); m_wy1_median = _wtoi(tmpCstring);
		statusflag = statusflag && fin.ReadString(tmpCstring); m_Npass1_median = _wtoi(tmpCstring);
		statusflag = statusflag && fin.ReadString(tmpCstring); m_minNneigh_p1 = _wtoi(tmpCstring);
		statusflag = statusflag && fin.ReadString(tmpCstring); m_residual_threshold_p1 = _wtof(tmpCstring);
		statusflag = statusflag && fin.ReadString(tmpCstring); m_insert_threshold_p1 = _wtof(tmpCstring);
		statusflag = statusflag && fin.ReadString(tmpCstring); m_wx1_smooth = _wtoi(tmpCstring);
		statusflag = statusflag && fin.ReadString(tmpCstring); m_wy1_smooth = _wtoi(tmpCstring);
		statusflag = statusflag && fin.ReadString(tmpCstring); m_Npass1_smooth = _wtoi(tmpCstring);
		statusflag = statusflag && fin.ReadString(tmpCstring); m_wx2_median = _wtoi(tmpCstring);
		statusflag = statusflag && fin.ReadString(tmpCstring); m_wy2_median = _wtoi(tmpCstring);
		statusflag = statusflag && fin.ReadString(tmpCstring); m_Npass2_median = _wtoi(tmpCstring);
		statusflag = statusflag && fin.ReadString(tmpCstring); m_minNneigh_p2 = _wtoi(tmpCstring);
		statusflag = statusflag && fin.ReadString(tmpCstring); m_residual_threshold_p2 = _wtof(tmpCstring);
		statusflag = statusflag && fin.ReadString(tmpCstring); m_insert_threshold_p2 = _wtof(tmpCstring);
		statusflag = statusflag && fin.ReadString(tmpCstring); m_wx2_smooth = _wtoi(tmpCstring);
		statusflag = statusflag && fin.ReadString(tmpCstring); m_wy2_smooth = _wtoi(tmpCstring);
		statusflag = statusflag && fin.ReadString(tmpCstring); m_Npass2_smooth = _wtoi(tmpCstring);
		statusflag = statusflag && fin.ReadString(tmpCstring); m_max_dispx_win_percent1 = _wtof(tmpCstring);
		statusflag = statusflag && fin.ReadString(tmpCstring); m_max_dispy_win_percent1 = _wtof(tmpCstring);
		statusflag = statusflag && fin.ReadString(tmpCstring); m_max_dispx_win_percent2 = _wtof(tmpCstring);
		statusflag = statusflag && fin.ReadString(tmpCstring); m_max_dispy_win_percent2 = _wtof(tmpCstring);
		statusflag = statusflag && fin.ReadString(tmpCstring); m_Npeak_save = _wtoi(tmpCstring);
		statusflag = statusflag && fin.ReadString(tmpCstring); m_dispx_lim1 = _wtof(tmpCstring);
		statusflag = statusflag && fin.ReadString(tmpCstring); m_dispx_lim2 = _wtof(tmpCstring);
		statusflag = statusflag && fin.ReadString(tmpCstring); m_dispy_lim1 = _wtof(tmpCstring);
		statusflag = statusflag && fin.ReadString(tmpCstring); m_dispy_lim2 = _wtof(tmpCstring);
		fin.Close();
		if (!statusflag)
		{
			MessageBox(NULL, _T("Parameter missing in Parameter.dat!"), _T("Message"), MB_OK);
			return 1;
		}
	}
	else
	{
		MessageBox(NULL, _T("Can't open parameter file!"), _T("Message"), MB_OK);
		return 1;
	}

	gpusetup gpu;
	if (!gpu.GetNofGpu())
	{
		MessageBox(NULL, _T("No GPU on this machine!"), _T("Message"), MB_OK);
		return 1;
	}
	gpu.AssignGpu(m_device_id);
	gpu.PrintGpuProperties();
	gpu.meminfo();

	ffop fios;
	vector<int> pairidx;
	if (fin.Open(m_dir_index, CFile::modeRead))
	{
		fin.SeekToBegin();
		while (fin.ReadString(tmpCstring))
		{
			pairidx.push_back(_wtoi(tmpCstring));
		}
		fin.Close();
	}
	else
	{
		MessageBox(NULL, _T("Can't open image pair index file!"), _T("Message"), MB_OK);
		return 1;
	}
	if (pairidx.empty())
	{
		MessageBox(NULL, _T("No image pair index detected!"), _T("Message"), MB_OK);
		return 1;
	}

	if (!fios.FolderExist(m_folder_result))
	{
		if (!fios.CreateFolder(m_folder_result))
		{
			MessageBox(NULL, _T("Folder creation not successful!"), _T("Message"), MB_OK);
			return 1;
		}
	}

	CImageIO ImgIO;
	ImgIO.SetImageSize(m_Nx, m_Ny);
	ImgIO.m_nImType = CImageIO::imfTIFF;
	tmpCstring.Format(_T("%04d"), pairidx[0]);
	dir_img1 = m_prefix_f1 + tmpCstring + m_suffix_f1;
	if (!fios.FileExist(dir_img1)){
		MessageBox(NULL, _T("An image for crosss-correlation is missing!"), _T("Message"), MB_OK);
		return 1;
	}
	ImgIO.m_szFileName = dir_img1;
	ImgIO.getImHeadTiff();

	/*CImageIO imgout;
	imgout.SetImageSize(m_Nx, m_Ny);
	imgout.m_nImType = CImageIO::imfTIFF;
	imgout.m_szFileName = dir_img1;
	imgout.getImHeadTiff();*/

	mhe_gpu mhegpu;
	mhegpu.setupgridblockdim(dim1_grid, dim2_grid, dim3_grid, dim1_block, dim2_block, dim3_block);
	if (!mhegpu.init(m_Nx, m_Ny, m_mhe_ws, m_mhe_thresh)){
		mhegpu.freemem();
		MessageBox(NULL, _T("Initialization of mhe_gpu class failed!"), _T("Message"), MB_OK);
		return 1;
	}

	xcorr2d_multi xcor;
	xcor.setupgridblockdim(dim1_grid, dim2_grid, dim3_grid, dim1_block, dim2_block, dim3_block);
	if (!xcor.init(m_Nx, m_Ny, m_Nsec_cc, m_ws_x1, m_ws_y1, m_spacingx1, m_spacingy1, m_Npass1, m_ws_x2, m_ws_y2, m_spacingx2, m_spacingy2, m_Npass2, m_wx1_median, m_wy1_median,
		m_wx2_median, m_wy2_median, m_wx1_smooth, m_wy1_smooth, m_wx2_smooth, m_wy2_smooth, m_Npeak_save, m_max_dispx_win_percent1, m_max_dispy_win_percent1, m_max_dispx_win_percent2,
		m_max_dispy_win_percent2, m_dispx_lim1, m_dispx_lim2, m_dispy_lim1, m_dispy_lim2)){
		xcor.freemem();
		mhegpu.freemem();
		MessageBox(NULL, _T("Initialization of xcorr2d_multi class failed!"), _T("Message"), MB_OK);
		return 1;
	}
	cout << "After allocating GPU memory for cross-correlation:" << endl;
	gpu.meminfo();

	if (cudaMalloc((void**)&d_img_org, sizeof(unsigned char)*m_Nx*m_Ny) != cudaSuccess){
		cudaFree(d_img_org);
		xcor.freemem();
		mhegpu.freemem();
		MessageBox(NULL, _T("Error in allocating d_img_org!"), _T("Message"), MB_OK);
		return 1;
	}

	for (i = 0; i < pairidx.size(); i++){
		begin = clock();
		cout << "Processing pair " << pairidx[i] << ": " << i + 1 << " out of " << pairidx.size() << endl;
		tmpCstring.Format(_T("%04d"), pairidx[i]);
		dir_img1 = m_prefix_f1 + tmpCstring + m_suffix_f1;
		dir_img2 = m_prefix_f2 + tmpCstring + m_suffix_f2;
		if (!fios.FileExist(dir_img1) || !fios.FileExist(dir_img2)){
			statusflag = false;
			MessageBox(NULL, _T("An image for crosss-correlation is missing!"), _T("Message"), MB_OK);
			break;
		}

		//dir_img_out1 = m_folder_result + _T("1_") + tmpCstring + _T(".tif");
		//dir_img_out2 = m_folder_result + _T("2_") + tmpCstring + _T(".tif");
		
		ImgIO.m_szFileName = dir_img1;
		ImgIO.ImRead(xcor.h_img1_org);
		ImgIO.m_szFileName = dir_img2;
		ImgIO.ImRead(xcor.h_img2_org);
		if (m_ifmhe == 1){
			cudaMemcpy(d_img_org, xcor.h_img1_org, sizeof(unsigned char)*xcor.Nt, cudaMemcpyHostToDevice);
			if (!mhegpu.processsing(d_img_org)){
				statusflag = false;
				MessageBox(NULL, _T("mhegpu.processing failed!"), _T("Message"), MB_OK);
				break;
			}
			cudaMemcpy(xcor.h_img1_org, mhegpu.d_enhanced, sizeof(unsigned char)*xcor.Nt, cudaMemcpyDeviceToHost);

			cudaMemcpy(d_img_org, xcor.h_img2_org, sizeof(unsigned char)*xcor.Nt, cudaMemcpyHostToDevice);
			if (!mhegpu.processsing(d_img_org)){
				statusflag = false;
				MessageBox(NULL, _T("mhegpu.processing failed!"), _T("Message"), MB_OK);
				break;
			}
			cudaMemcpy(xcor.h_img2_org, mhegpu.d_enhanced, sizeof(unsigned char)*xcor.Nt, cudaMemcpyDeviceToHost);
		}
		/*imgout.m_szFileName = dir_img_out1;
		imgout.ImWrite(xcor.h_img1_org);
		imgout.m_szFileName = dir_img_out2;
		imgout.ImWrite(xcor.h_img2_org);*/
		xcor.prep_img_data();
		for (j = 0; j < xcor.Npass1; j++){
			if (j > 0){
				xcor.extend_filtered_vecfield_p1();
			}
			if (!xcor.Calc_cc_map_p1(j)){
				statusflag = false;
				MessageBox(NULL, _T("xcor.Calc_cc_map_p1 failed!"), _T("Message"), MB_OK);
				break;
			}
			if (!xcor.MedianFilter_Smoothing_p1(m_Npass1_median, m_minNneigh_p1, m_residual_threshold_p1, m_insert_threshold_p1, m_Npass1_smooth)){
				statusflag = false;
				MessageBox(NULL, _T("xcor.MedianFilter_Smoothing_p1 failed!"), _T("Message"), MB_OK);
				break;
			}
		}
		if (!statusflag) break;
		if (xcor.Npass2 > 0){
			xcor.extend_filtered_vecfield_p1();
			if (!xcor.Upsampling_vector_field()){
				statusflag = false;
				MessageBox(NULL, _T("xcor.Upsampling_vector_field failed!"), _T("Message"), MB_OK);
				break;
			}
			for (j = 0; j < xcor.Npass2; j++){
				xcor.extend_filtered_vecfield_p2();
				if (!xcor.Calc_cc_map_p2()){
					statusflag = false;
					MessageBox(NULL, _T("xcor.Calc_cc_map_p2 failed!"), _T("Message"), MB_OK);
					break;
				}
				if (!xcor.MedianFilter_Smoothing_p2(m_Npass2_median, m_minNneigh_p2, m_residual_threshold_p2, m_insert_threshold_p2, m_Npass2_smooth, 1)){
					statusflag = false;
					MessageBox(NULL, _T("xcor.MedianFilter_Smoothing_p2 failed!"), _T("Message"), MB_OK);
					break;
				}
			}
			if (!statusflag) break;
		}
		tmpCstring.Format(_T("%04d"), pairidx[i]);
		vec_name.Format(_T("_vector_GPU_Grid1_%d-peak-candidate_3DCC-%d-%d-%dpass_4peaks_raw.dat"), m_Npeak_save, m_ws_x1, m_ws_y1, m_Npass1);
		dir_result = m_folder_result + tmpCstring + vec_name;
		if (!xcor.write_vec_data_general(dir_result, 0)){
			statusflag = false;
			MessageBox(NULL, _T("xcor.write_vec_data_general failed!"), _T("Message"), MB_OK);
			break;
		}

		vec_name.Format(_T("_vector_GPU_Grid1_%d-peak-candidate_3DCC-%d-%d-%dpass_median-%d-%d-%.1f-%dpass_smooth-%d-%d-%dpass.dat"),
			m_Npeak_save, m_ws_x1, m_ws_y1, m_Npass1, m_wx1_median, m_wy1_median, m_insert_threshold_p1, m_Npass1_median, m_wx1_smooth, m_wy1_smooth, m_Npass1_smooth);
		dir_result = m_folder_result + tmpCstring + vec_name;
		if (!xcor.write_vec_data_general(dir_result, 1)){
			statusflag = false;
			MessageBox(NULL, _T("xcor.write_vec_data_general failed!"), _T("Message"), MB_OK);
			break;
		}

		if (m_Npass2 > 0){
			vec_name.Format(_T("_vector_GPU_Grid2_%d-peak-candidate_3DCC-%d-%d-%dpass_4peaks_raw.dat"), m_Npeak_save, m_ws_x2, m_ws_y2, m_Npass2);
			dir_result = m_folder_result + tmpCstring + vec_name;
			if (!xcor.write_vec_data_general(dir_result, 2)){
				statusflag = false;
				MessageBox(NULL, _T("xcor.write_vec_data_general failed!"), _T("Message"), MB_OK);
				break;
			}

			vec_name.Format(_T("_vector_GPU_Grid2_%d-peak-candidate_3DCC-%d-%d-%dpass_median-%d-%d-%.1f-%dpass_smooth-%d-%d-%dpass.dat"),
				m_Npeak_save, m_ws_x2, m_ws_y2, m_Npass2, m_wx2_median, m_wy2_median, m_insert_threshold_p2, m_Npass2_median, m_wx2_smooth, m_wy2_smooth, m_Npass2_smooth);
			dir_result = m_folder_result + tmpCstring + vec_name;
			if (!xcor.write_vec_data_general(dir_result, 3)){
				statusflag = false;
				MessageBox(NULL, _T("xcor.write_vec_data_general failed!"), _T("Message"), MB_OK);
				break;
			}

			if (!xcor.MedianFilter_Smoothing_p2(m_Npass2_median, m_minNneigh_p2, m_residual_threshold_p2, m_insert_threshold_p2, 0, 0)){
				statusflag = false;
				MessageBox(NULL, _T("xcor.MedianFilter_Smoothing_p2 failed!"), _T("Message"), MB_OK);
				break;
			}
			vec_name.Format(_T("_vector_GPU_Grid2_%d-peak-candidate_3DCC-%d-%d-%dpass_median-%d-%d-%.1f-%dpass.dat"),
				m_Npeak_save, m_ws_x2, m_ws_y2, m_Npass2, m_wx2_median, m_wy2_median, m_insert_threshold_p2, m_Npass2_median);
			dir_result = m_folder_result + tmpCstring + vec_name;
			if (!xcor.write_vec_data_general(dir_result, 3)){
				statusflag = false;
				MessageBox(NULL, _T("xcor.write_vec_data_general failed!"), _T("Message"), MB_OK);
				break;
			}
		}
		end = clock();
		elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
		cout << "Done! " << elapsed_secs << " seconds" << endl;
	}
	cudaFree(d_img_org);
	mhegpu.freemem();
	xcor.freemem();
	cudaDeviceReset();
	if (statusflag){
		cout << "NO ERROR!";
		getchar();
		return 0;
	}
	else{
		cout << "ERROR! Press Enter to Exit:";
		getchar();
		return 1;
	}
}