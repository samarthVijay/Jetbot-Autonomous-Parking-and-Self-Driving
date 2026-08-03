// vector_add.cu — Practice 1: 1D Vector Addition & CPU vs GPU Benchmark
//
// TASK FOR YOU:
// Fill in the __global__ kernel function `vector_add` below using what you learned!
//
// Syntax hints:
//  - Calculate 1D global thread index: int idx = blockIdx.x * blockDim.x + threadIdx.x;
//  - Always add a bounds check: if (idx >= n) return;
//  - Write output element: C[idx] = A[idx] + B[idx];

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <chrono>

// ═══════════════════════════════════════════════════════════════════
// STEP 1: WRITE YOUR CUDA KERNEL HERE
// ═══════════════════════════════════════════════════════════════════
__global__ void vector_add(const float* A, const float* B, float* C, int n) {
    // TODO: Calculate 1D global index 'idx'
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // TODO: Add bounds check
    if (idx >= n) return;

    // TODO: Perform vector addition for index 'idx'
    C[idx] = A[idx] + B[idx];
}


// ═══════════════════════════════════════════════════════════════════
// CPU REFERENCE IMPLEMENTATION (For benchmarking & verification)
// ═══════════════════════════════════════════════════════════════════
void vector_add_cpu(const float* A, const float* B, float* C, int n) {
    for (int i = 0; i < n; i++) {
        C[i] = A[i] + B[i];
    }
}


// ═══════════════════════════════════════════════════════════════════
// HARNESS & BENCHMARK SUITE
// ═══════════════════════════════════════════════════════════════════
int main() {
    int n = 5000000;  // 5 Million elements
    size_t bytes = n * sizeof(float);

    printf("=====================================================\n");
    printf("  CUDA Practice 1: Vector Addition & Benchmarking   \n");
    printf("  Array Size: %d elements (%.2f MB)\n", n, bytes / 1e6);
    printf("=====================================================\n\n");

    // 1. Allocate Host Memory
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C_gpu = (float*)malloc(bytes);
    float *h_C_cpu = (float*)malloc(bytes);

    for (int i = 0; i < n; i++) {
        h_A[i] = (float)i * 0.5f;
        h_B[i] = (float)(i * 2) * 0.25f;
    }

    // 2. Benchmark CPU Execution Time
    printf("[CPU] Running vector_add on CPU...\n");
    auto start_cpu = std::chrono::high_resolution_clock::now();
    
    vector_add_cpu(h_A, h_B, h_C_cpu, n);
    
    auto end_cpu = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(end_cpu - start_cpu).count();
    printf("  --> CPU Execution Time: %.3f ms\n\n", cpu_ms);

    // 3. Allocate Device Memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // 4. Configure Grid & Launch Kernel
    int threadsPerBlock = 256;
    int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;

    printf("[GPU] Launching CUDA Kernel (%d blocks x %d threads)...\n", numBlocks, threadsPerBlock);
    
    // Create CUDA Events for precise GPU timing
    cudaEvent_t start_gpu, stop_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);

    cudaEventRecord(start_gpu);
    
    // KERNEL LAUNCH
    vector_add<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, n);
    
    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);

    float gpu_ms = 0;
    cudaEventElapsedTime(&gpu_ms, start_gpu, stop_gpu);
    printf("  --> GPU Kernel Execution Time: %.3f ms\n\n", gpu_ms);

    // 5. Copy Result Back & Verify Correctness
    cudaMemcpy(h_C_gpu, d_C, bytes, cudaMemcpyDeviceToHost);

    int errors = 0;
    for (int i = 0; i < n; i++) {
        if (fabsf(h_C_gpu[i] - h_C_cpu[i]) > 1e-4) {
            if (errors < 5) {
                printf("  [ERROR] Mismatch at index %d: GPU=%.4f, CPU=%.4f\n", i, h_C_gpu[i], h_C_cpu[i]);
            }
            errors++;
        }
    }

    printf("=====================================================\n");
    if (errors == 0) {
        printf("  VERIFICATION: \033[1;32m[PASS]\033[0m All %d results match!\n", n);
        printf("  SPEEDUP:      \033[1;36m%.2fx Faster on GPU\033[0m\n", cpu_ms / gpu_ms);
    } else {
        printf("  VERIFICATION: \033[1;31m[FAIL]\033[0m %d errors detected.\n", errors);
    }
    printf("=====================================================\n");

    // Cleanup
    cudaEventDestroy(start_gpu); cudaEventDestroy(stop_gpu);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C_gpu); free(h_C_cpu);

    return 0;
}
