// preprocess_kernel.cu — Zero-Copy Image Preprocessing CUDA Kernel
//
// Performs:
//   1. uint8 -> float32 conversion (/ 255.0)
//   2. BGR -> RGB channel swap
//   3. HWC (Interleaved) -> CHW (Planar) memory layout transpose
//   4. ImageNet Normalization: output = (color - mean) / std
//
// Target Hardware: Jetson Nano (sm_53)

#include <cuda_runtime.h>
#include <stdio.h>

// ImageNet normalization constants stored in GPU Constant Memory (64KB cached fast access)
// RGB order
__constant__ float d_mean[3] = {0.485f, 0.456f, 0.406f};
__constant__ float d_std[3]  = {0.229f, 0.224f, 0.225f};

/*
 * CUDA Kernel: preprocess_image_kernel
 *
 * Each thread handles ONE pixel (x, y).
 * For a 224x224 image, we launch 224x224 = 50,176 threads across a 2D grid of 16x16 blocks.
 *
 * Input:  const unsigned char* input  (uint8 BGR, HWC layout: height x width x 3)
 * Output: float* output               (float32 RGB, CHW layout: 3 x height x width)
 */
__global__ void preprocess_image_kernel(
    const unsigned char* __restrict__ input,
    float* __restrict__ output,
    int width,
    int height
) {
    // Calculate 2D pixel coordinates for this thread
    int x = blockIdx.x * blockDim.x + threadIdx.x;  // column (0 to width-1)
    int y = blockIdx.y * blockDim.y + threadIdx.y;  // row (0 to height-1)

    // Bounds check
    if (x >= width || y >= height) return;

    // 1. Calculate input index in HWC (Height-Width-Channels) interleaved layout
    // Offset for pixel (y, x): (y * width + x) * 3
    int hwc_idx = (y * width + x) * 3;

    // Read raw BGR uint8 bytes and normalize to [0.0, 1.0] float32
    float b = (float)input[hwc_idx + 0] / 255.0f;
    float g = (float)input[hwc_idx + 1] / 255.0f;
    float r = (float)input[hwc_idx + 2] / 255.0f;

    // 2. Channel swap (BGR -> RGB) & ImageNet normalization
    float r_norm = (r - d_mean[0]) / d_std[0];
    float g_norm = (g - d_mean[1]) / d_std[1];
    float b_norm = (b - d_mean[2]) / d_std[2];

    // 3. Write to CHW (Channel-Height-Width) planar layout
    // Spatial offset within one channel plane (224x224 = 50,176 floats)
    int chw_offset = y * width + x;
    int plane_size = height * width;

    // Plane 0: Red channel
    output[0 * plane_size + chw_offset] = r_norm;
    // Plane 1: Green channel
    output[1 * plane_size + chw_offset] = g_norm;
    // Plane 2: Blue channel
    output[2 * plane_size + chw_offset] = b_norm;
}


/*
 * Host C Wrapper Function (Exported via extern "C" for ctypes)
 */
extern "C" {

int cuda_preprocess(
    const unsigned char* d_input,   // Pointer to GPU/Unified memory input BGR uint8
    float* d_output,                // Pointer to GPU/Unified memory output RGB float32
    int width,
    int height
) {
    if (!d_input || !d_output || width <= 0 || height <= 0) {
        return -1;
    }

    // 2D Block configuration: 16x16 = 256 threads per block
    dim3 block(16, 16);

    // 2D Grid configuration: Cover entire image width and height
    dim3 grid(
        (width  + block.x - 1) / block.x,
        (height + block.y - 1) / block.y
    );

    // Launch kernel
    preprocess_image_kernel<<<grid, block>>>(d_input, d_output, width, height);

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Preprocess Kernel Error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Wait for GPU execution to complete
    cudaDeviceSynchronize();
    return 0;
}

} // extern "C"
