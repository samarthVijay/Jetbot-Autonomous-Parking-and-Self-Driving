// test_preprocess.cu — Standalone CUDA C Image Preprocessing Verification Harness
//
// Creates a synthetic BGR image in GPU memory, runs preprocess_image_kernel,
// and compares the output element-by-element against CPU computation.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

// Forward declaration of exported C function
extern "C" int cuda_preprocess(const unsigned char* d_input, float* d_output, int width, int height);

// ImageNet normalization reference arrays
static const float REF_MEAN[3] = {0.485f, 0.456f, 0.406f};  // R, G, B
static const float REF_STD[3]  = {0.229f, 0.224f, 0.225f};  // R, G, B

void cpu_preprocess(const unsigned char* input, float* output, int width, int height) {
    int plane_size = width * height;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int hwc_idx = (y * width + x) * 3;
            float b = (float)input[hwc_idx + 0] / 255.0f;
            float g = (float)input[hwc_idx + 1] / 255.0f;
            float r = (float)input[hwc_idx + 2] / 255.0f;

            int chw_offset = y * width + x;
            output[0 * plane_size + chw_offset] = (r - REF_MEAN[0]) / REF_STD[0];
            output[1 * plane_size + chw_offset] = (g - REF_MEAN[1]) / REF_STD[1];
            output[2 * plane_size + chw_offset] = (b - REF_MEAN[2]) / REF_STD[2];
        }
    }
}

int main() {
    int width = 224, height = 224;
    int num_pixels = width * height;
    size_t input_bytes = num_pixels * 3 * sizeof(unsigned char);
    size_t output_bytes = num_pixels * 3 * sizeof(float);

    printf("==========================================================");
    printf("\n  CUDA Image Preprocessing Verification Suite (C Native)  \n");
    printf("  Dimensions: %dx%d x 3 (%.2f KB raw input)              \n", width, height, input_bytes / 1024.0);
    printf("==========================================================\n\n");

    // 1. Allocate Host Memory
    unsigned char* h_input = (unsigned char*)malloc(input_bytes);
    float* h_output_cpu = (float*)malloc(output_bytes);
    float* h_output_gpu = (float*)malloc(output_bytes);

    // Fill synthetic image with deterministic pattern
    for (int i = 0; i < num_pixels * 3; i++) {
        h_input[i] = (unsigned char)((i * 17 + 3) % 256);
    }

    // 2. CPU Reference Preprocessing
    printf("[1/3] Computing CPU Reference ImageNet Preprocessing...\n");
    cpu_preprocess(h_input, h_output_cpu, width, height);

    // 3. Device Allocation & Memory Copy
    printf("[2/3] Executing CUDA Preprocessing Kernel...\n");
    unsigned char* d_input;
    float* d_output;
    cudaMalloc(&d_input, input_bytes);
    cudaMalloc(&d_output, output_bytes);

    cudaMemcpy(d_input, h_input, input_bytes, cudaMemcpyHostToDevice);

    // Benchmark GPU Execution Time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Run kernel wrapper
    int res = cuda_preprocess(d_input, d_output, width, height);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_ms = 0;
    cudaEventElapsedTime(&gpu_ms, start, stop);
    printf("  --> CUDA Kernel Execution Time: %.3f ms (res=%d)\n\n", gpu_ms, res);

    // 4. Copy Back & Verify
    printf("[3/3] Verifying Output Tensors Element-by-Element...\n");
    cudaMemcpy(h_output_gpu, d_output, output_bytes, cudaMemcpyDeviceToHost);

    int errors = 0;
    float max_diff = 0.0f;
    for (int i = 0; i < num_pixels * 3; i++) {
        float diff = fabsf(h_output_gpu[i] - h_output_cpu[i]);
        if (diff > max_diff) max_diff = diff;
        if (diff > 1e-4f) {
            if (errors < 5) {
                printf("  [MISMATCH] idx=%d: GPU=%.6f, CPU=%.6f, diff=%.6f\n",
                       i, h_output_gpu[i], h_output_cpu[i], diff);
            }
            errors++;
        }
    }

    printf("==========================================================\n");
    printf("  Max Absolute Difference: %.8f\n", max_diff);
    if (errors == 0) {
        printf("  VERIFICATION: \033[1;32m[PASS]\033[0m CUDA output EXACTLY matches CPU!\n");
        printf("  PERFORMANCE:  \033[1;36m%.3f ms per frame (%.1f FPS)\033[0m\n", gpu_ms, 1000.0 / gpu_ms);
    } else {
        printf("  VERIFICATION: \033[1;31m[FAIL]\033[0m %d / %d elements mismatch!\n", errors, num_pixels * 3);
    }
    printf("==========================================================\n");

    // Cleanup
    cudaEventDestroy(start); cudaEventDestroy(stop);
    cudaFree(d_input); cudaFree(d_output);
    free(h_input); free(h_output_cpu); free(h_output_gpu);

    return errors > 0 ? 1 : 0;
}
