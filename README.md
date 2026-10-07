# PIV_xcorr2d V1.0: GPU-Accelerated Multi-Grid PIV Cross-Correlation

A CUDA/C++ engine for batch-processing **Particle Image Velocimetry (PIV)** image pairs on an NVIDIA GPU. It turns pairs of particle images into 2D displacement (velocity) fields using FFT-based cross-correlation, coarse-to-fine multi-grid refinement, iterative window deformation, and DaVis-style outlier validation, with almost every stage running on the GPU.

*Originally written circa 2013.*

---

## What it does

PIV measures fluid flow by seeding it with tracer particles and imaging them twice in quick succession. Each image is cut into small **interrogation windows**. For each window, the code asks: "how far did this patch of particles move between frame 1 and frame 2?" The answer is the location of the peak in the 2D cross-correlation of the two patches. Doing this for thousands of overlapping windows per image, over multiple passes, is expensive. This code runs every window in parallel on the GPU.

## Processing pipeline

```
 For each image pair (index list):
   ┌──────────────────────────────────────────────────────────────┐
   │ 1. Read frame 1 and frame 2 (8-bit TIFF) into pinned memory  │
   │ 2. Optional MHE contrast enhancement (GPU)                   │
   │ 3. Pad images to fit all windows; upload to GPU              │
   ├──────────────── Grid 1 (coarse, e.g. 64x64) ─────────────────┤
   │   repeat Npass1:                                             │
   │     a. Deform frame 2 using current vector field (pass > 1)  │
   │     b. Batched FFT cross-correlation of all windows          │
   │     c. Find top-N correlation peaks per window               │
   │     d. Sub-pixel Gaussian peak fit                           │
   │     e. Universal outlier detection + peak replacement        │
   │     f. Fill holes, top-hat smoothing                         │
   ├──────────────── Grid 2 (fine, e.g. 32x32) ───────────────────┤
   │   Bilinear upsampling of grid-1 field onto grid 2            │
   │   repeat Npass2: steps a–f at the finer resolution           │
   ├──────────────────────────────────────────────────────────────┤
   │ 4. Write raw and filtered vector fields (Tecplot ASCII)      │
   └──────────────────────────────────────────────────────────────┘
```

---

## Technical highlights

### 1. Batched FFT cross-correlation (cuFFT)
- All interrogation windows of a grid are gathered into one contiguous complex buffer (`organize_data_fft2d_multi`) and transformed with a **single batched 2D plan** (`cufftPlanMany`, `CUFFT_C2C`). The GPU does thousands of small FFTs in one call instead of looping over windows.
- Correlation is computed in the frequency domain, `IFFT( FFT(A) · conj(FFT(B)) )`, with a custom element-wise `multiply_conjugate` kernel, followed by magnitude extraction and a two-step in-place `fftshift` (row swap, then column swap) so that zero displacement sits at the window center.
- Window sizes may be non-square (each dimension a multiple of 4, up to 1024).

### 2. Memory sectioning for large images
- High-resolution cameras (the sample configuration is **6600 × 4400 px**) can produce more correlation planes than fit in GPU memory. The `Nsec_cc` parameter splits the windows into sections that are processed one after another through the same buffers. The FFT plan batch size (`Neach`) and the remainder section (`Nlast`) are computed at init time.

### 3. Multi-peak detection with parallel reductions
- `find_local_max_2d` marks every pixel that is a positive 4-neighbor local maximum inside an allowed search region. Candidates are appended per window using `atomicAdd` on a per-window counter.
- The top-N peaks (default 4, matching DaVis) are then extracted by repeating a **two-stage, templated shared-memory arg-max reduction** (`reduce_max_index_general` → `reduce_max_index_final`). Each round finds the strongest remaining peak, then suppresses it (`replace_local_max_2d`) so the next round finds the runner-up.
- The reduction uses a **warp-synchronous unrolled tail** (`warp_max_index` on `volatile` shared memory). The block size is a template parameter, chosen at runtime through a `switch`, so the compiler can fully unroll each variant.
- The peak search region is limited by a user-defined fraction of the window size (`max_disp_win_percent`). This caps the largest displacement that can be detected and rejects edge artifacts.

### 4. Sub-pixel accuracy: 3-point Gaussian fit
- Each candidate peak is refined independently in x and y with the classic three-point Gaussian estimator:

  ```
  δ = (ln f₋₁ − ln f₊₁) / (2 · (ln f₋₁ + ln f₊₁ − 2 ln f₀))
  ```

  The previous pass's displacement is added back (predictor-corrector). Peaks on the window border, non-positive peaks, NaN/Inf results, and vectors outside the user's physical limits are flagged invalid.

### 5. Iterative window deformation
- On passes after the first, frame 2 is **warped** by the current filtered vector field (`Kernel_Window_Deformation_2d`). For every pixel, the displacement is bilinearly interpolated from the vector grid, and the image intensity is bilinearly resampled at the shifted position.
- This removes in-window shear and gradient bias, so later passes measure only a small residual displacement. That improves accuracy in flows with strong velocity gradients.
- The vector field is first padded by replicating its border (`extend_filtered_vecfield_*`), so interpolation near image edges stays in bounds.

### 6. Coarse-to-fine multi-grid
- Grid 1 uses large windows (robust, handles large displacements). Its final field is **bilinearly upsampled** onto the finer Grid 2 (`Upsampling_vecfield_kernel_2d`) as the predictor. Grid 2 then runs its own deformation passes with smaller windows and higher overlap (e.g. 32 × 32 at 50% overlap) for higher spatial resolution.

### 7. Universal outlier detection with peak-candidate replacement
- Implements the **normalized median test** (Westerweel & Scarano, 2005), the same method as DaVis's "Universal outlier detection/removal/insert":
  1. Gather valid neighbors in an `wx × wy` window (atomic append).
  2. Compute the neighborhood median (`Calc_median`, a per-thread partial bubble sort that only sorts up to the middle element).
  3. Compute the median of absolute deviations from that median (residual).
  4. Normalize: `|u − median| / (residual_median + ε)`, with ε = 0.1 px.
- **Key feature:** if the strongest peak fails the test, the code tries the 2nd, 3rd, and 4th correlation peaks in order. The first candidate that passes replaces the vector. This recovers many vectors that a simple reject-and-interpolate scheme would lose.
- Vectors with too few valid neighbors (`minNneigh`) are left unchanged. The test runs for a configurable number of passes.

### 8. Hole filling and smoothing
- Remaining gaps are filled iteratively (`fill_blank_vecfield`): each empty vector takes the mean of its valid 3 × 3 neighbors. This repeats until a device-side counter reports nothing left to fill, growing inward from the valid region.
- A configurable **top-hat (box) filter** smooths the field for a set number of passes.

### 9. GPU MHE image enhancement
- An optional pre-processing step (`MHE7_4_gpu.cu`) applies local, histogram-based contrast enhancement to raw particle images. It suppresses uneven background and boosts particle contrast before correlation:
  - The image is tiled into `ws × ws` blocks. 256-bin per-block histograms are built with `atomicAdd` (`MakeHist_kernel`).
  - Each block's transfer function is computed from the combined histogram of the block and its 3 × 3 neighbors, which smooths transitions between tiles.
  - Cumulative histograms use a **work-efficient Blelloch prefix scan** in shared memory (`prescan_kernel`, an up-sweep then down-sweep).
  - The lowest `thresh` fraction of pixels (e.g. 91%, mostly background) is mapped to 0. The remaining top fraction (mostly particles) is stretched across the full 0–255 range.

### 10. General CUDA design patterns
- **Grid-stride loops** in every element-wise kernel, so one fixed launch configuration handles any problem size.
- **Pinned host memory** (`cudaHostAlloc`) for faster host↔device image transfer.
- All intermediate data (correlation planes, peak lists, vector fields, median buffers) **stays on the GPU** across passes. Only the input images and final vector fields cross the PCIe bus.
- Every kernel launch is followed by a `cudaPeekAtLastError` check with a named error message.
- Device query and memory reporting at startup (`gpuinfo.cu`), with a selectable GPU on multi-GPU machines.

---

## Repository layout

```
xcorr2d_V1.0.sln                     Visual Studio solution
xcorr2d_V1.0/
  main.cpp                           Driver: parameter parsing, batch loop, output naming
  xcorr2d_multi_grid.cu / .cuh       Core PIV engine (class xcorr2d_multi, ~25 kernels)
  MHE7_4_gpu.cu / .h                 GPU MHE image enhancement (class mhe_gpu)
  gpuinfo.cu / .h                    GPU enumeration, selection, memory reporting
  ImageIOPkg.cpp / .h                TIFF/RAW image I/O (CImageIO, MFC-based)
  FolderFileOp.cpp / .h              File/folder existence and creation helpers
  Parameter.dat                      Example parameter file
  Parameter_Instructions_xcorr2d_V1.0.txt   Line-by-line parameter documentation
data/
  data_organized/                    Sample image pairs (1_000N.tif / 2_000N.tif) + pair index
  data_organized/result/             Sample output vector fields
  2DPIV_1.2ms_40us_1/                Additional raw PIV sample sequence
x64/Debug/                           Prebuilt debug executable (Windows x64)
```

## Input

- **Images:** 8-bit grayscale TIFF, one file per frame. File names are built as `prefix + %04d index + suffix`, for example `data_organized\1_0001.tif` and `data_organized\2_0001.tif`.
- **Pair index file:** a plain-text column of integers listing which pairs to process.
- **Parameter file:** `Parameter.dat` (or a path passed as the first command-line argument). It has one value per line: file paths, GPU id, MHE settings, image size, window size, spacing and pass count per grid, median/smoothing settings per grid, peak-search limits, number of peak candidates, and physical displacement limits. See `Parameter_Instructions_xcorr2d_V1.0.txt` for the full reference.

## Output

Tecplot-compatible ASCII `.dat` files per pair, written to the result folder. File names encode the processing settings, for example:

```
0001_vector_GPU_Grid1_4-peak-candidate_3DCC-64-64-2pass_4peaks_raw.dat
0001_vector_GPU_Grid1_4-peak-candidate_3DCC-64-64-2pass_median-5-5-3.0-2pass_smooth-3-3-1pass.dat
0001_vector_GPU_Grid2_4-peak-candidate_3DCC-32-32-2pass_4peaks_raw.dat
0001_vector_GPU_Grid2_..._median-5-5-3.0-2pass_smooth-3-3-1pass.dat
0001_vector_GPU_Grid2_..._median-5-5-3.0-2pass.dat     (median-filtered, no fill/smooth)
```

- `*_raw.dat`: `x, y, dx1, dy1, …, dx4, dy4`, the displacement for each of the 4 peak candidates.
- Filtered files: `x, y, displacement-x, displacement-y`, in pixels, on the vector grid.

## Building and running

**Environment (as configured in the project):**
- Windows, x64
- Visual Studio 2013 (platform toolset `v120`) with **MFC** (`CString`, `CStdioFile`, `CFile`)
- NVIDIA CUDA Toolkit with cuFFT. The project file currently references the CUDA 8.0 build customization and `compute_50, sm_50`. Retarget both to match your installed toolkit and GPU.

**Run:**
```
xcorr2d_V1.0.exe                     (reads Parameter.dat from the working directory)
xcorr2d_V1.0.exe path\to\params.dat
```
The program prints GPU properties and memory usage, then the per-pair processing time.

## Known limitations

- Windows/MFC only. Not portable to Linux or macOS without replacing the MFC I/O layer.
- Only 8-bit TIFF input is used by the driver.
- The raw-output writer always writes 4 peak columns, so set the peak-candidate count to at least 4.
- The **Remove threshold** parameter is read but not used. Only the **Insert threshold** drives the median test.
- No window weighting (noted as future work in the original instructions).
- Build artifacts (`.sdf`, `.suo`, `.pdb`, `.obj`, `x64/Debug`) are committed alongside the source.

## Acknowledgments

`ImageIOPkg` (the `CImageIO` TIFF/RAW I/O class) is a reused utility originally written by Jian Sheng (2001), as credited in its file header.
